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

"""Run the portable Triton sampling kernel tests on Intel XPU."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU sampling tests require an Intel XPU",
)

from ops.test_sampling_triton import (  # noqa: E402,F401
    _qrita_gumbel_scratch,
    _top_k_topp_allowed_ids,
    test_gumbel_generic_mixed_batch_samples_allowed_set,
    test_gumbel_min_p_samples_allowed_set,
    test_gumbel_sample_from_pools_takes_tail_token,
    test_gumbel_top_k_top_p_qrita_verify_idx_mapping_matches_expanded_rows,
    test_gumbel_top_k_top_p_samples_allowed_set,
    test_gumbel_top_p_parallel_samples_allowed_set,
)
from tokenspeed_kernel.ops.sampling.triton import (  # noqa: E402
    gumbel_sample_top_k_top_p_qrita_from_pools,
)


@pytest.fixture
def device() -> str:
    return "xpu"


def test_qrita_sizes_its_grid_on_xpu(device: str) -> None:
    """Without an explicit program count the qrita sampler sizes its grid from
    the platform's compute-unit count, which XPU must provide."""
    torch.manual_seed(5030)
    rows, vocab_size, pool_rows = 4, 1025, 4
    logits = torch.randn((rows, vocab_size), dtype=torch.float32, device=device)
    req_pool_indices = torch.arange(rows, dtype=torch.int32, device=device)
    temperature_pool = torch.ones(pool_rows, dtype=torch.float32, device=device)
    top_k_pool = torch.full((pool_rows,), 64, dtype=torch.int32, device=device)
    top_p_pool = torch.full((pool_rows,), 0.85, dtype=torch.float32, device=device)
    seed_pool = torch.arange(701, 701 + pool_rows, dtype=torch.int64, device=device)
    offsets_pool = torch.zeros(pool_rows, dtype=torch.int64, device=device)
    buffer, table, out = _qrita_gumbel_scratch(rows, vocab_size, device)

    sampled = gumbel_sample_top_k_top_p_qrita_from_pools(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        buffer,
        table,
        out,
    )

    for row, token_id in enumerate(sampled.cpu().tolist()):
        assert token_id in _top_k_topp_allowed_ids(logits[row], 1.0, 64, 0.85)
