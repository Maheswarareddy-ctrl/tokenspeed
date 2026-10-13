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

"""Registration shim for Intel XPU multi-head attention (tokenspeed-kernel-xpu)."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_xpu:
    from tokenspeed_kernel_xpu.ops.mha import (
        mha_decode_with_kvcache as _mha_decode_with_kvcache,
    )
    from tokenspeed_kernel_xpu.ops.mha import (
        mha_extend_with_kvcache as _mha_extend_with_kvcache,
    )
    from tokenspeed_kernel_xpu.ops.mha import mha_prefill as _mha_prefill

    _XPU_CAPABILITY = CapabilityRequirement(vendors=frozenset({"intel"}))
    _XPU_DTYPES = {torch.float16, torch.bfloat16}
    # Options validated against a reference on XPU; the rest raise in
    # tokenspeed-kernel-xpu.
    _XPU_OPTIONS = {
        "logit_cap": frozenset({False}),
        "return_lse": frozenset({False}),
        "sinks": frozenset({False}),
        "skip_softmax": frozenset({False}),
        "sliding_window": frozenset({False}),
    }
    # sgl-kernel-xpu paged attention supports only these page sizes.
    _XPU_PAGE_SIZES = frozenset({64, 128})

    @register_kernel(
        "attention",
        "mha_prefill",
        name="sycl_mha_prefill",
        solution="sycl",
        capability=_XPU_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _XPU_DTYPES),
        priority=Priority.PERFORMANT,
        traits=_XPU_OPTIONS,
    )
    def sycl_mha_prefill(**kwargs):
        return _mha_prefill(**kwargs)

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="sycl_mha_extend_with_kvcache",
        solution="sycl",
        capability=_XPU_CAPABILITY,
        signatures=format_signatures(("q", "k_cache", "v_cache"), "dense", _XPU_DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            **_XPU_OPTIONS,
            "page_size": _XPU_PAGE_SIZES,
            "is_causal": frozenset({False, True}),
        },
    )
    def sycl_mha_extend_with_kvcache(**kwargs):
        return _mha_extend_with_kvcache(**kwargs)

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="sycl_mha_decode_with_kvcache",
        solution="sycl",
        capability=_XPU_CAPABILITY,
        signatures=format_signatures(("q", "k_cache", "v_cache"), "dense", _XPU_DTYPES),
        priority=Priority.PERFORMANT,
        traits={**_XPU_OPTIONS, "page_size": _XPU_PAGE_SIZES},
    )
    def sycl_mha_decode_with_kvcache(**kwargs):
        return _mha_decode_with_kvcache(**kwargs)
