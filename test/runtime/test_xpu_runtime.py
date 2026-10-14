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
import contextlib
from unittest import mock

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU runtime tests require an Intel XPU",
)

import tokenspeed_kernel._triton as kernel_triton  # noqa: E402

from tokenspeed.runtime.configs.device_config import DeviceConfig  # noqa: E402
from tokenspeed.runtime.execution.breakable_cuda_graph import (  # noqa: E402
    BreakableCapture,
    break_here,
)
from tokenspeed.runtime.execution.device import _NoDeviceWork  # noqa: E402
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner  # noqa: E402
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.triton import (  # noqa: E402
    TritonSamplingBackend,
)
from tokenspeed.runtime.utils import triton as runtime_triton  # noqa: E402
from tokenspeed.runtime.utils.common import (  # noqa: E402
    device_mem_get_info,
    get_available_gpu_memory,
)
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


def test_triton_sampling_backend_builds_on_xpu() -> None:
    config = SamplingBackendConfig(
        enable_speculative_sampling=False,
        sampling_stream="batch",
        logprob_order="torch",
        max_bs=4,
        max_draft_tokens_per_req=4,
        max_req_pool_size=8,
        vocab_size=1024,
        device="xpu",
    )
    TritonSamplingBackend(config)  # used to crash on torch.cuda


def test_decode_graph_capture_records_an_xpu_graph() -> None:
    runner = ForwardStepRunner.__new__(ForwardStepRunner)
    runner.device = "xpu"
    runner.device_module = torch.xpu

    class _Recorded(Exception):
        pass

    def _xpu_graph():
        raise _Recorded

    with mock.patch.object(torch.xpu, "XPUGraph", _xpu_graph):
        with pytest.raises(_Recorded):
            runner._capture_one(1, observer=contextlib.nullcontext())


def test_free_memory_counts_torch_allocations_on_xpu() -> None:
    torch.xpu.empty_cache()
    before, _ = device_mem_get_info(torch.xpu, 0)
    held = torch.empty(1 << 30, dtype=torch.uint8, device="xpu")
    after, _ = device_mem_get_info(torch.xpu, 0)
    assert before - after >= held.numel()


def test_breakable_capture_replays_on_xpu() -> None:
    x = torch.ones(4, device="xpu")
    out = torch.empty(4, device="xpu")
    cap = BreakableCapture()
    with cap:
        y = x * 2
        z = break_here(lambda t: t + 1, out, y)
        w = z * 10
    assert cap.num_segments == 3  # graph, eager break, graph

    x.fill_(3)
    cap.replay()
    torch.xpu.synchronize()
    assert w.tolist() == [70.0] * 4
