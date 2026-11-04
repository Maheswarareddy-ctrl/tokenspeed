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

"""Run the fused accepted-frontier advance on Intel XPU."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU accepted-frontier tests require an Intel XPU",
)

import tokenspeed_kernel.ops.metadata.accepted_frontier as accepted_frontier  # noqa: E402
from tokenspeed_kernel.ops.metadata import advance_accepted_frontier  # noqa: E402


def _batch(batch, num_extends, pool, tokens, context, padding_rows):
    torch.manual_seed(batch)
    dev = "xpu"
    slots = torch.randperm(pool - 1, device=dev)[:batch].to(torch.int64)
    slots[batch - padding_rows :] = pool - 1
    input_lengths = torch.randint(1, 6, (batch,), device=dev, dtype=torch.int32)
    input_lengths[num_extends:] = 6
    accept_lengths = torch.randint(1, 7, (batch,), device=dev, dtype=torch.int32)
    accept_lengths[:num_extends] = 0
    valid = torch.randint(0, 500, (pool,), device=dev, dtype=torch.int32)
    tail = torch.randint(-1, 100, (pool, context), device=dev, dtype=torch.int64)
    previous = torch.randint(-1, 100, (tokens, context), device=dev, dtype=torch.int64)
    mask = torch.rand(tokens, device=dev) > 0.3
    ids = torch.randint(0, 100, (tokens,), device=dev, dtype=torch.int32)
    return slots, input_lengths, accept_lengths, valid, tail, previous, mask, ids


@pytest.mark.parametrize(
    ("batch", "num_extends", "padding_rows", "context"),
    ((7, 3, 0, 3), (16, 0, 5, 3), (33, 1, 2, 4), (5000, 1700, 9, 3)),
)
def test_xpu_advance_runs_the_triton_kernel(
    monkeypatch, batch, num_extends, padding_rows, context
):
    pool, tokens = max(64, batch + 1), max(256, batch * 6)
    slots, inputs, accepts, valid, tail, previous, mask, ids = _batch(
        batch, num_extends, pool, tokens, context, padding_rows
    )
    expected_valid, expected_tail = valid.clone(), tail.clone()
    accepted_frontier._advance_accepted_frontier_torch(
        slots,
        inputs,
        accepts,
        expected_valid,
        num_extends,
        pool - 1,
        expected_tail,
        previous,
        mask,
        ids,
    )

    def fail(*args):
        raise AssertionError("XPU fell back to the torch reference path")

    monkeypatch.setattr(accepted_frontier, "_advance_accepted_frontier_torch", fail)

    advance_accepted_frontier(
        slots,
        inputs,
        accepts,
        valid,
        num_extends,
        pool - 1,
        ngram_tail=tail,
        ngram_previous_tokens=previous,
        ngram_token_mask=mask,
        input_ids=ids,
    )
    torch.xpu.synchronize()

    assert torch.equal(valid, expected_valid)
    assert torch.equal(tail, expected_tail)
