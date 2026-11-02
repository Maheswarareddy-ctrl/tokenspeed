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

"""Intel XPU GQA attention prologue chained from sgl-kernel-xpu SYCL kernels."""

from __future__ import annotations

import torch
from sgl_kernel import fused_qk_rope_with_cos_sin_cache_inplace as _sgl_rope
from sgl_kernel import rmsnorm as _sgl_rmsnorm
from sgl_kernel import store_cache_xpu as _sgl_store_cache


def gqa_prologue(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    head_dim: int,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    eps: float,
    cos_sin_cache: torch.Tensor | None,
    positions: torch.Tensor | None,
    is_neox: bool,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head RMSNorm, RoPE and the K/V store, one sgl-kernel-xpu kernel per step.

    The steps round to the activation dtype between them, as SGLang's XPU path does.

    Args:
        q: Query ``[num_tokens, num_q_heads * head_dim]``; rows may be strided.
        k: Key ``[num_tokens, num_kv_heads * head_dim]``; rows may be strided.
        v: Value ``[num_tokens, num_kv_heads * head_dim]``; rows may be strided.
        head_dim: Width of one head.
        q_weight: Per-head RMSNorm weight for the query, or ``None`` for no norm.
        k_weight: Per-head RMSNorm weight for the key, or ``None`` for no norm.
        eps: RMSNorm epsilon.
        cos_sin_cache: fp32 cos/sin table, or ``None`` for no RoPE.
        positions: Token positions ``[num_tokens]``, or ``None`` for no RoPE.
        is_neox: NEOX (True) or GPT-J (False) rotation.
        k_cache: Key cache ``[num_slots, num_kv_heads, head_dim]``.
        v_cache: Value cache ``[num_slots, num_kv_heads, head_dim]``.
        slots: Cache slot of each stored row; the first ``slots.numel()`` rows are stored.

    Returns:
        The prepared query and key, ``[num_tokens, heads * head_dim]``.
    """
    num_tokens = q.shape[0]
    if q_weight is not None:
        # The norm reads one head per row, so the strided QKV views are packed first.
        q = _sgl_rmsnorm(q.reshape(-1, head_dim), q_weight, eps).view(num_tokens, -1)
        k = _sgl_rmsnorm(k.reshape(-1, head_dim), k_weight, eps).view(num_tokens, -1)
    if cos_sin_cache is not None:
        _sgl_rope(
            q.view(num_tokens, -1, head_dim),
            k.view(num_tokens, -1, head_dim),
            cos_sin_cache,
            positions.long(),
            head_dim,
            is_neox,
        )
    rows = slots.numel()
    if rows:
        row_dim = k_cache.shape[1] * head_dim
        _sgl_store_cache(
            k[:rows].view(rows, row_dim),
            v[:rows].view(rows, row_dim),
            k_cache.view(-1, row_dim),
            v_cache.view(-1, row_dim),
            slots,
        )
    return q, k
