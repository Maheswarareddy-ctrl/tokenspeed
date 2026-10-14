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

import argparse

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU runtime tests require an Intel XPU",
)

import tokenspeed_kernel._triton as kernel_triton  # noqa: E402

from tokenspeed.runtime.configs.device_config import DeviceConfig  # noqa: E402
from tokenspeed.runtime.execution.device import _NoDeviceWork  # noqa: E402
from tokenspeed.runtime.utils import triton as runtime_triton  # noqa: E402
from tokenspeed.runtime.utils.common import get_available_gpu_memory  # noqa: E402
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def test_device_config_accepts_xpu() -> None:
    assert DeviceConfig("xpu").device == torch.device("xpu")


def test_cli_accepts_xpu_device() -> None:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    assert parser.parse_args(["--model", "x", "--device", "xpu"]).device == "xpu"


def test_available_gpu_memory_on_xpu() -> None:
    total_gb = torch.xpu.get_device_properties(0).total_memory / (1 << 30)

    assert 0 < get_available_gpu_memory("xpu", 0) <= total_gb


def test_the_guard_rejects_xpu_work_on_the_control_plane() -> None:
    xpu = torch.zeros(2, device="xpu")
    with _NoDeviceWork():
        with pytest.raises(RuntimeError, match="XPU factory"):
            torch.empty(2, device="xpu")
        with pytest.raises(RuntimeError, match="XPU tensor"):
            xpu.add_(1)
        assert xpu.view(1, 2).shape == (1, 2)  # views do no device work


def test_runtime_triton_is_the_kernel_platform_triton() -> None:
    assert runtime_triton.triton is kernel_triton.triton
    assert runtime_triton.tl is kernel_triton.tl
