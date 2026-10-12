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

"""Intel XPU normalization operators backed by sgl-kernel-xpu SYCL kernels."""

from __future__ import annotations

import torch
from sgl_kernel import fused_add_rmsnorm as _sgl_fused_add_rmsnorm
from sgl_kernel import rmsnorm as _sgl_rmsnorm


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm, optionally adding and updating a residual tensor."""
    # An empty batch has nothing to normalize; skip the kernel launch.
    if x.shape[0] == 0:
        result = x if out is None else out
        return (result, residual) if residual is not None else result
    if residual is not None:
        # In place: residual += x, then x = rmsnorm(residual) * weight.
        _sgl_fused_add_rmsnorm(x, residual, weight, eps)
        return x, residual
    return _sgl_rmsnorm(x, weight, eps, out=out)


__all__ = ["rmsnorm"]
