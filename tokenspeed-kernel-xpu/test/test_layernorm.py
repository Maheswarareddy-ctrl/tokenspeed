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
    reason="SYCL RMSNorm tests require an Intel XPU",
)

pytest.importorskip("sgl_kernel")

from tokenspeed_kernel_xpu.ops.layernorm import rmsnorm  # noqa: E402

# bfloat16 has few mantissa bits; float32 should match almost exactly.
_TOLERANCE = {torch.bfloat16: 2e-2, torch.float32: 1e-5}


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Float32 RMSNorm on CPU, independent of the device under test."""
    x_float = x.cpu().float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps) * weight.cpu().float()).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rmsnorm(dtype: torch.dtype) -> None:
    x = torch.randn(7, 256, device="xpu", dtype=dtype)
    weight = torch.randn(256, device="xpu", dtype=dtype)
    expected = _reference(x, weight, 1e-6)

    out = torch.empty_like(x)
    result = rmsnorm(x, weight, 1e-6, out=out)

    assert result is out
    tol = _TOLERANCE[dtype]
    torch.testing.assert_close(result.cpu(), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_add_rmsnorm_updates_residual(dtype: torch.dtype) -> None:
    x = torch.randn(7, 256, device="xpu", dtype=dtype)
    residual = torch.randn_like(x)
    expected_residual = (x + residual).cpu()
    weight = torch.randn(256, device="xpu", dtype=dtype)
    expected = _reference(expected_residual, weight, 1e-6)

    result, residual_out = rmsnorm(x, weight, 1e-6, residual=residual)

    assert residual_out is residual
    tol = _TOLERANCE[dtype]
    torch.testing.assert_close(residual.cpu(), expected_residual, atol=tol, rtol=tol)
    torch.testing.assert_close(result.cpu(), expected, atol=tol, rtol=tol)


def test_empty_batch_skips_kernel() -> None:
    x = torch.empty(0, 256, device="xpu", dtype=torch.bfloat16)
    weight = torch.ones(256, device="xpu", dtype=torch.bfloat16)

    assert rmsnorm(x, weight, 1e-6) is x
