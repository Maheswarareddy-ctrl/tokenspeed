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

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="SYCL MHA tests require an Intel XPU",
)

pytest.importorskip("sgl_kernel")

from tokenspeed_kernel_xpu.ops.mha import (  # noqa: E402
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)

BF16 = torch.bfloat16
# Qwen3-like GQA: 16 query heads share 8 KV heads of width 128.
_Q_HEADS, _KV_HEADS, _HEAD_DIM = 16, 8, 128
_TOL = 2e-2


def _int32(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, device="xpu", dtype=torch.int32)


def _cumsum(lengths: list[int]) -> list[int]:
    out = [0]
    for length in lengths:
        out.append(out[-1] + length)
    return out


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_start: int
) -> torch.Tensor:
    """Causal GQA attention in float32 on CPU; query row i sits at q_start + i."""
    q, k, v = q.cpu().float(), k.cpu().float(), v.cpu().float()
    groups = _Q_HEADS // _KV_HEADS
    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(_HEAD_DIM)
    q_pos = torch.arange(q.shape[0])[:, None] + q_start
    scores = scores.masked_fill(torch.arange(k.shape[0])[None, :] > q_pos, -math.inf)
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


def _paged_cache(lengths: list[int], page_size: int):
    """Random K/V for each sequence, stored in shuffled pages."""
    pages = [math.ceil(length / page_size) for length in lengths]
    total = sum(pages) + 3
    order = torch.randperm(total)
    shape = (total, page_size, _KV_HEADS, _HEAD_DIM)
    k_cache = torch.randn(shape, device="xpu", dtype=BF16)
    v_cache = torch.randn_like(k_cache)
    table = torch.zeros(len(lengths), max(pages), dtype=torch.int32)
    start = 0
    for row, count in enumerate(pages):
        table[row, :count] = order[start : start + count]
        start += count
    return k_cache, v_cache, table.to("xpu")


def _rows(cache: torch.Tensor, table: torch.Tensor, row: int, length: int):
    return cache[table[row].long()].reshape(-1, _KV_HEADS, _HEAD_DIM)[:length]


def test_prefill_matches_reference() -> None:
    lengths = [37, 130, 5]
    cu = _cumsum(lengths)
    q = torch.randn(cu[-1], _Q_HEADS, _HEAD_DIM, device="xpu", dtype=BF16)
    k = torch.randn(cu[-1], _KV_HEADS, _HEAD_DIM, device="xpu", dtype=BF16)
    v = torch.randn_like(k)

    out = mha_prefill(q, k, v, _int32(cu), cu, max(lengths))

    expected = torch.cat(
        [_reference(q[a:b], k[a:b], v[a:b], 0) for a, b in zip(cu[:-1], cu[1:])]
    )
    torch.testing.assert_close(out.cpu().float(), expected, atol=_TOL, rtol=_TOL)


@pytest.mark.parametrize("page_size", [64, 128])
def test_extend_matches_reference(page_size: int) -> None:
    prefix, new = [70, 0, 200], [9, 33, 1]
    total = [p + n for p, n in zip(prefix, new)]
    k_cache, v_cache, table = _paged_cache(total, page_size)
    cu_q = _cumsum(new)
    q = torch.randn(cu_q[-1], _Q_HEADS, _HEAD_DIM, device="xpu", dtype=BF16)

    out = mha_extend_with_kvcache(
        q,
        _int32(cu_q),
        _int32(_cumsum(total)),
        k_cache,
        v_cache,
        table,
        _int32(total),
        max(new),
        max(total),
        is_causal=True,
    )

    expected = torch.cat(
        [
            _reference(
                q[cu_q[row] : cu_q[row + 1]],
                _rows(k_cache, table, row, total[row]),
                _rows(v_cache, table, row, total[row]),
                prefix[row],
            )
            for row in range(len(total))
        ]
    )
    torch.testing.assert_close(out.cpu().float(), expected, atol=_TOL, rtol=_TOL)


@pytest.mark.parametrize("q_len", [1, 3], ids=["decode", "multi_query"])
@pytest.mark.parametrize("page_size", [64, 128])
def test_decode_matches_reference(page_size: int, q_len: int) -> None:
    lengths = [5, 100, 513, 64]
    k_cache, v_cache, table = _paged_cache(lengths, page_size)
    q = torch.randn(len(lengths) * q_len, _Q_HEADS, _HEAD_DIM, device="xpu", dtype=BF16)

    out = mha_decode_with_kvcache(
        q, k_cache, v_cache, table, _int32(lengths), max(lengths), q_len
    )

    expected = torch.cat(
        [
            _reference(
                q[row * q_len : (row + 1) * q_len],
                _rows(k_cache, table, row, length),
                _rows(v_cache, table, row, length),
                length - q_len,
            )
            for row, length in enumerate(lengths)
        ]
    )
    assert out.shape == q.shape
    torch.testing.assert_close(out.cpu().float(), expected, atol=_TOL, rtol=_TOL)


def test_unsupported_options_are_rejected() -> None:
    q = torch.randn(4, _Q_HEADS, _HEAD_DIM, device="xpu", dtype=BF16)
    k = torch.randn(4, _KV_HEADS, _HEAD_DIM, device="xpu", dtype=BF16)

    with pytest.raises(NotImplementedError, match="sliding windows"):
        mha_prefill(q, k, k, _int32([0, 4]), [0, 4], 4, window_left=2)
