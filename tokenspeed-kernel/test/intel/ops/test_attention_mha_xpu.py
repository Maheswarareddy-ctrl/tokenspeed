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
    reason="XPU attention dispatch tests require an Intel XPU",
)

xpu_mha = pytest.importorskip("tokenspeed_kernel_xpu.ops.mha")

import tokenspeed_kernel.ops.attention.mha as mha  # noqa: E402
from tokenspeed_kernel.selection import NoKernelFoundError  # noqa: E402

BF16 = torch.bfloat16


def _decode(page_size: int) -> torch.Tensor:
    k_cache = torch.randn(4, page_size, 8, 128, device="xpu", dtype=BF16)
    q = torch.randn(2, 16, 128, device="xpu", dtype=BF16)
    return mha.mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=torch.randn_like(k_cache),
        page_table=torch.tensor([[0, 1], [2, 3]], device="xpu", dtype=torch.int32),
        cache_seqlens=torch.tensor([page_size + 1, 3], device="xpu", dtype=torch.int32),
        max_seqlen_k=page_size + 1,
        max_seqlen_q=1,
    )


def test_xpu_dispatches_mha_decode_to_tokenspeed_kernel_xpu(monkeypatch) -> None:
    calls = []
    real = xpu_mha.mha_decode_with_kvcache

    def spy(**kwargs):
        calls.append(kwargs["k_cache"].shape[1])
        return real(**kwargs)

    sycl = pytest.importorskip("tokenspeed_kernel.ops.attention.mha.sycl")
    monkeypatch.setattr(sycl, "_mha_decode_with_kvcache", spy)

    out = _decode(page_size=64)

    assert calls == [64]
    assert out.shape == (2, 16, 128)


def test_unsupported_page_size_is_not_selected() -> None:
    with pytest.raises(NoKernelFoundError):
        _decode(page_size=32)


def test_xpu_dispatches_mha_extend_to_tokenspeed_kernel_xpu(monkeypatch) -> None:
    calls = []
    real = xpu_mha.mha_extend_with_kvcache

    def spy(**kwargs):
        calls.append(kwargs["is_causal"])
        return real(**kwargs)

    sycl = pytest.importorskip("tokenspeed_kernel.ops.attention.mha.sycl")
    monkeypatch.setattr(sycl, "_mha_extend_with_kvcache", spy)
    k_cache = torch.randn(4, 64, 8, 128, device="xpu", dtype=BF16)
    q = torch.randn(5, 16, 128, device="xpu", dtype=BF16)

    out = mha.mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=torch.tensor([0, 2, 5], device="xpu", dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 66, 69], device="xpu", dtype=torch.int32),
        k_cache=k_cache,
        v_cache=torch.randn_like(k_cache),
        page_table=torch.tensor([[0, 1], [2, 3]], device="xpu", dtype=torch.int32),
        cache_seqlens=torch.tensor([66, 3], device="xpu", dtype=torch.int32),
        max_seqlen_q=3,
        max_seqlen_k=66,
        is_causal=True,
    )

    assert calls == [True]
    assert out.shape == (5, 16, 128)
