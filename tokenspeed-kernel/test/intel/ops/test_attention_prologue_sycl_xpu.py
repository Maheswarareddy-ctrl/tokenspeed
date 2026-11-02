# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU attention prologue tests require an Intel XPU",
)

from tokenspeed_kernel.ops.attention.prologue import (  # noqa: E402
    HeadKVCache,
    HeadNorm,
    RopeStyle,
    Rotary,
    _select,
    gqa_prologue,
)

BF16 = torch.bfloat16
# Qwen3-like GQA layer: 16 query heads share 8 KV heads of width 128.
_TOKENS, _Q_HEADS, _KV_HEADS, _HEAD_DIM, _CACHE_SLOTS = 7, 16, 8, 128, 64
_EPS = 1e-6


def _cos_sin_cache(rotary_dim: int, max_pos: int = 4096) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to("xpu")


def _norm_then_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Step-by-step reference: fp32 RMSNorm rounded to bf16, then fp32 NEOX RoPE
    rounded to bf16, as SGLang's XPU path computes."""
    heads = x.unflatten(-1, (-1, _HEAD_DIM)).float()
    rms = torch.rsqrt(heads.square().mean(-1, keepdim=True) + _EPS)
    normed = (heads * rms * weight.float()).to(BF16).float()
    cos, sin = cos_sin[positions].chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    x1, x2 = normed.chunk(2, dim=-1)
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return rotated.to(BF16).flatten(1)


@pytest.mark.parametrize("return_kv", [True, False], ids=["extend", "decode"])
def test_sycl_gqa_prologue_matches_step_by_step_reference(return_kv: bool) -> None:
    torch.manual_seed(0)
    qkv = torch.randn(
        _TOKENS, (_Q_HEADS + 2 * _KV_HEADS) * _HEAD_DIM, device="xpu", dtype=BF16
    )
    # Strided views of one packed QKV row, as the QKV projection produces.
    q, k, v = qkv.split(
        [_Q_HEADS * _HEAD_DIM, _KV_HEADS * _HEAD_DIM, _KV_HEADS * _HEAD_DIM], dim=-1
    )
    norm = HeadNorm(
        (torch.rand(_HEAD_DIM, device="xpu") + 0.5).to(BF16),
        (torch.rand(_HEAD_DIM, device="xpu") + 0.5).to(BF16),
        0.0,
        _EPS,
    )
    cos_sin = _cos_sin_cache(_HEAD_DIM)
    positions = torch.randint(0, 4096, (_TOKENS,), device="xpu")
    rotary = Rotary(
        cos_sin_cache=cos_sin, positions=positions, style=RopeStyle.NEOX, mrope=None
    )
    slots = torch.randperm(_CACHE_SLOTS, device="xpu")[:_TOKENS]
    k_cache = torch.zeros(_CACHE_SLOTS, _KV_HEADS, _HEAD_DIM, device="xpu", dtype=BF16)
    v_cache = torch.zeros_like(k_cache)
    ref_q = _norm_then_rope(q, norm.q_weight, cos_sin, positions)
    ref_k = _norm_then_rope(k, norm.k_weight, cos_sin, positions)
    ref_v = v.clone()

    out = gqa_prologue(
        q,
        k,
        v,
        norm=norm,
        rotary=rotary,
        cache=HeadKVCache(k_cache=k_cache, v_cache=v_cache, scales=None, slots=slots),
        return_kv=return_kv,
        solution=None,
        override="sycl_gqa_prologue",
    )

    assert (out.k is not None) == return_kv
    # One bf16 unit in the last place: the kernels may order fp32 math differently.
    torch.testing.assert_close(out.q, ref_q, atol=1e-2, rtol=8e-3)
    torch.testing.assert_close(k_cache[slots].flatten(1), ref_k, atol=1e-2, rtol=8e-3)
    assert torch.equal(v_cache[slots].flatten(1), ref_v)


def test_triton_stays_the_default_gqa_prologue() -> None:
    traits = {
        "head_dim": _HEAD_DIM,
        "token_heads": _TOKENS * _Q_HEADS,
        "has_norm": True,
        "kv_format": "native",
        "kv_convert": False,
        "mrope": False,
        "partial_rotary": False,
        "return_kv": False,
        "rope_style": "neox",
    }
    kernel = _select("gqa_prologue", BF16, tuple(traits.items()), None, None)
    assert kernel.name == "triton_gqa_prologue"
