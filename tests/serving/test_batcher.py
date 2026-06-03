"""Batcher: concurrent submits collapse to a single batched predict call;
each future gets its row's result; closed batcher refuses new submits."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest


class _CountingRuntime:
    """Stand-in for ONNXRuntime; counts how many predict() invocations happen."""

    def __init__(self):
        self.calls = 0
        self.batch_sizes: list[int] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.batch_sizes.append(len(X))
        # Trivial: return sum of each row
        return np.asarray(X).sum(axis=1)


@pytest.mark.asyncio
async def test_batcher_collapses_concurrent_submits_to_one_call() -> None:
    from deepCab.serving.batcher import Batcher

    rt = _CountingRuntime()
    b = Batcher(rt, max_batch=16, max_wait_ms=20)
    await b.start()

    rows = [np.array([float(i), float(i + 1)], dtype=np.float32) for i in range(8)]
    results = await asyncio.gather(*(b.submit(r) for r in rows))

    expected = [r.sum() for r in rows]
    assert results == pytest.approx(expected)
    # All 8 rows should land in one batched call (well under the 20ms window)
    assert rt.calls == 1
    assert rt.batch_sizes == [8]
    await b.stop()


@pytest.mark.asyncio
async def test_batcher_caps_at_max_batch() -> None:
    from deepCab.serving.batcher import Batcher

    rt = _CountingRuntime()
    b = Batcher(rt, max_batch=4, max_wait_ms=50)
    await b.start()

    rows = [np.array([float(i)], dtype=np.float32) for i in range(10)]
    results = await asyncio.gather(*(b.submit(r) for r in rows))

    assert len(results) == 10
    # 10 items / batch of 4 -> at least 3 batches (4, 4, 2)
    assert rt.calls >= 3
    assert max(rt.batch_sizes) == 4
    await b.stop()


@pytest.mark.asyncio
async def test_batcher_stop_rejects_new_submits() -> None:
    from deepCab.serving.batcher import Batcher

    rt = _CountingRuntime()
    b = Batcher(rt)
    await b.start()
    await b.stop()
    with pytest.raises(RuntimeError, match="closed"):
        await b.submit(np.array([1.0], dtype=np.float32))
