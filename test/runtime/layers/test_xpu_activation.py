# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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
import torch.nn.functional as F

from tokenspeed.runtime.layers import activation as activation_module
from tokenspeed.runtime.layers.activation import SiluAndMul

_XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()

pytestmark = pytest.mark.skipif(not _XPU_AVAILABLE, reason="Intel XPU is unavailable")


def _reference(x: torch.Tensor, limit: float | None) -> torch.Tensor:
    """Float32 SwiGLU on CPU, independent of the device under test."""
    gate, up = x.cpu().float().chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return (F.silu(gate) * up).to(x.dtype)


@pytest.mark.parametrize("limit", [None, 7.0], ids=["plain", "clamped"])
def test_xpu_silu_and_mul_layer_calls_kernel(monkeypatch, limit) -> None:
    calls = []
    kernel = activation_module.silu_and_mul

    def spy(x, out, *, limit):
        calls.append(limit)
        return kernel(x, out, limit=limit)

    monkeypatch.setattr(activation_module, "silu_and_mul", spy)
    x = torch.randn(4, 256, device="xpu", dtype=torch.bfloat16) * 4

    result = SiluAndMul(swiglu_limit=limit)(x)

    assert calls == [limit]
    torch.testing.assert_close(result.cpu(), _reference(x, limit), atol=2e-2, rtol=2e-2)


def test_xpu_silu_and_mul_layer_rejects_fp8_out() -> None:
    x = torch.randn(4, 256, device="xpu", dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="XPU fp8_out"):
        SiluAndMul()(x, fp8_out=True)
