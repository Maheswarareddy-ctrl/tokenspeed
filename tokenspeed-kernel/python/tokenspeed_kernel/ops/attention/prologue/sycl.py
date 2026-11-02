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

"""Registration shim for the Intel XPU GQA prologue (tokenspeed-kernel-xpu)."""

import torch
from tokenspeed_kernel.ops.attention.prologue.types import (
    BOOLS,
    GQAPrologueOutput,
    HeadKVCache,
    HeadNorm,
    RopeStyle,
    Rotary,
)
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_xpu:
    from tokenspeed_kernel_xpu.ops.prologue import gqa_prologue as _gqa_prologue

    @register_kernel(
        "attention",
        "gqa_prologue",
        name="sycl_gqa_prologue",
        solution="sycl",
        capability=CapabilityRequirement(vendors=frozenset({"intel"})),
        signatures=format_signatures(("q",), "dense", {torch.bfloat16}),
        # Ranked below triton_gqa_prologue: this path rounds between steps (like
        # the composite), so it serves only when selected explicitly.
        priority=Priority.PORTABLE,
        traits={
            # sgl-kernel-xpu's fused RoPE supports only these head widths.
            "head_dim": frozenset({128, 256, 512}),
            "has_norm": BOOLS,
            "kv_format": frozenset({"native"}),
            "kv_convert": frozenset({False}),
            "mrope": frozenset({False}),
            "partial_rotary": frozenset({False}),
            "return_kv": BOOLS,
            "rope_style": frozenset({"none", "neox", "gptj"}),
        },
    )
    def sycl_gqa_prologue(
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        norm: HeadNorm | None,
        rotary: Rotary | None,
        cache: HeadKVCache,
        return_kv: bool,
        enable_pdl: bool,
    ) -> GQAPrologueOutput:
        del enable_pdl  # PDL is NVIDIA-only.
        if norm is not None and norm.weight_offset != 0.0:
            raise NotImplementedError(
                "sycl_gqa_prologue does not support norm weight offsets"
            )
        q, k = _gqa_prologue(
            q.flatten(1),
            k,
            v,
            head_dim=cache.k_cache.shape[-1],
            q_weight=None if norm is None else norm.q_weight,
            k_weight=None if norm is None else norm.k_weight,
            eps=0.0 if norm is None else norm.eps,
            cos_sin_cache=None if rotary is None else rotary.cos_sin_cache,
            positions=None if rotary is None else rotary.positions,
            is_neox=rotary is not None and rotary.style is RopeStyle.NEOX,
            k_cache=cache.k_cache,
            v_cache=cache.v_cache,
            slots=cache.slots,
        )
        return GQAPrologueOutput(
            q=q, k=k if return_kv else None, v=v if return_kv else None
        )
