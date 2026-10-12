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

"""Intel XPU activation operators backed by sgl-kernel-xpu SYCL kernels."""

from __future__ import annotations

import torch
from sgl_kernel import silu_and_mul as _sgl_silu_and_mul
from sgl_kernel import silu_and_mul_clamp as _sgl_silu_and_mul_clamp


def silu_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    limit: float | None = None,
) -> torch.Tensor:
    """Fused ``SiLU(x[..., :D]) * x[..., D:]``, optionally clamped by ``limit``."""
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last dimension must be even, got {x.shape[-1]}")
    if not x.is_contiguous():
        x = x.contiguous()
    output_shape = (*x.shape[:-1], x.shape[-1] // 2)
    if out is None:
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    elif tuple(out.shape) != output_shape:
        raise ValueError(f"out shape must be {output_shape}, got {tuple(out.shape)}")
    # The SYCL op writes into out.contiguous(); a non-contiguous out would get
    # a temporary copy and never see the result.
    if not out.is_contiguous():
        raise ValueError("out must be contiguous for the SYCL silu_and_mul")
    if out.numel() == 0:
        return out
    if limit is None:
        return _sgl_silu_and_mul(x, out)
    # The clamp kernel rounds inputs to bfloat16 before clamping (DeepGEMM
    # semantics), so float16 inputs get bfloat16 accuracy on this path.
    _sgl_silu_and_mul_clamp(x, out, limit)
    return out


__all__ = ["silu_and_mul"]
