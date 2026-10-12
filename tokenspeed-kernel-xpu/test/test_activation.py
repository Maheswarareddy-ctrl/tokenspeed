# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="SYCL silu_and_mul tests require an Intel XPU",
)

pytest.importorskip("sgl_kernel")

from tokenspeed_kernel_xpu.ops.activation import silu_and_mul  # noqa: E402

# bfloat16 has few mantissa bits; float16 should match more closely.
_TOLERANCE = {torch.bfloat16: 2e-2, torch.float16: 2e-3}
_ROWS, _HIDDEN = 4, 128


def _reference(x: torch.Tensor, limit: float | None) -> torch.Tensor:
    """Float32 SwiGLU on CPU, independent of the device under test."""
    gate, up = x.cpu().float().chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return (F.silu(gate) * up).to(x.dtype)


@pytest.mark.parametrize("limit", [None, 7.0], ids=["plain", "clamped"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_silu_and_mul_matches_reference(
    dtype: torch.dtype, limit: float | None
) -> None:
    # Scaled so that a limit of 7 clamps some values.
    x = torch.randn(_ROWS, 2 * _HIDDEN, device="xpu", dtype=dtype) * 4
    expected = _reference(x, limit)
    out = torch.empty(_ROWS, _HIDDEN, device="xpu", dtype=dtype)

    result = silu_and_mul(x, out, limit=limit)

    assert result is out
    tol = _TOLERANCE[dtype]
    if limit is not None:
        # The SYCL clamp kernel rounds inputs to bfloat16 before clamping
        # (DeepGEMM semantics), so the clamped form has bfloat16 accuracy.
        tol = _TOLERANCE[torch.bfloat16]
    torch.testing.assert_close(result.cpu(), expected, atol=tol, rtol=tol)


def test_non_contiguous_input_is_accepted() -> None:
    wide = torch.randn(_ROWS, 4 * _HIDDEN, device="xpu", dtype=torch.bfloat16)
    x = wide[:, : 2 * _HIDDEN]

    result = silu_and_mul(x, limit=7.0)

    tol = _TOLERANCE[torch.bfloat16]
    torch.testing.assert_close(result.cpu(), _reference(x, 7.0), atol=tol, rtol=tol)


def test_empty_batch_skips_kernel() -> None:
    x = torch.empty(0, 2 * _HIDDEN, device="xpu", dtype=torch.bfloat16)

    assert silu_and_mul(x).shape == (0, _HIDDEN)


def test_non_contiguous_out_is_rejected() -> None:
    x = torch.randn(_ROWS, 2 * _HIDDEN, device="xpu", dtype=torch.bfloat16)
    wide = torch.empty(_ROWS, 2 * _HIDDEN, device="xpu", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="contiguous"):
        silu_and_mul(x, wide[:, :_HIDDEN])
