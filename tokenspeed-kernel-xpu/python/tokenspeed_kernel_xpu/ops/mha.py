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

"""Intel XPU multi-head attention backed by sgl-kernel-xpu SYCL kernels."""

from __future__ import annotations

import math

import torch
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


def _scale(q: torch.Tensor, softmax_scale: float | None) -> float:
    return softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])


def _check_options(
    *,
    window_left: int,
    logit_cap: float,
    sinks: torch.Tensor | None,
    return_lse: bool,
) -> None:
    # Only the options validated against a reference on XPU are accepted.
    if window_left >= 0:
        raise NotImplementedError("XPU MHA does not support sliding windows yet")
    if logit_cap:
        raise NotImplementedError("XPU MHA does not support logit caps yet")
    if sinks is not None:
        raise NotImplementedError("XPU MHA does not support attention sinks yet")
    if return_lse:
        raise NotImplementedError("XPU MHA does not return LSE yet")


def _check_unscaled(*scales: torch.Tensor | None) -> None:
    # Block-scaled (MXFP8) KV caches are not supported on XPU yet.
    if any(scale is not None for scale in scales):
        raise NotImplementedError("XPU MHA does not support block-scaled KV caches yet")


def mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run causal variable-length MHA over uncached K/V."""
    del cu_seqlens_cpu  # The SYCL kernel reads the device-side cu_seqlens.
    _check_options(
        window_left=window_left, logit_cap=logit_cap, sinks=sinks, return_lse=return_lse
    )
    return flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=_scale(q, softmax_scale),
        causal=True,
    )


def mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Attend new query rows to a paged KV cache that already holds them."""
    del max_seqlen_k, enable_pdl  # Read from cache_seqlens; PDL is NVIDIA-only.
    _check_options(
        window_left=window_left, logit_cap=logit_cap, sinks=sinks, return_lse=return_lse
    )
    _check_unscaled(q_scale, k_scale, v_scale)

    return flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=_scale(q, softmax_scale),
        causal=is_causal,
    )


def mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Decode ``max_seqlen_q`` query rows per request against a paged KV cache."""
    del max_seqlen_k, enable_pdl  # Read from cache_seqlens; PDL is NVIDIA-only.
    _check_options(
        window_left=window_left, logit_cap=logit_cap, sinks=sinks, return_lse=return_lse
    )
    _check_unscaled(q_scale, k_scale, v_scale)

    batch_size = cache_seqlens.shape[0]
    out = flash_attn_with_kvcache(
        q=q.view(batch_size, max_seqlen_q, q.shape[1], q.shape[2]),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        softmax_scale=_scale(q, softmax_scale),
        causal=max_seqlen_q > 1,
    )
    return out.view_as(q)


__all__ = ["mha_decode_with_kvcache", "mha_extend_with_kvcache", "mha_prefill"]
