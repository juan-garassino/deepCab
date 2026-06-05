"""Async request coalescer for the predict endpoint.

Pattern (classic ML-serving lesson):
    `submit(x)` queues a row and returns a Future. A single background worker
    pulls items, accumulating up to `max_batch` rows OR until `max_wait_ms`
    elapses since the first item arrived, then runs ONE batched session.run
    and resolves every queued future with its row of the output.

Why: a single 64-row inference is ~30× cheaper than 64 single-row calls on
ONNX runtime, especially for tree models where per-call overhead dominates.

Lifecycle: `await batcher.start()` from FastAPI lifespan, `await batcher.stop()`
on shutdown. The worker drains in-flight items on stop."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np

from deepCab.obs.log import get_logger
from deepCab.serving.runtime import ONNXRuntime

log = get_logger(__name__)


@dataclass
class _Item:
    fut: asyncio.Future
    row: np.ndarray


class Batcher:
    def __init__(
        self,
        runtime: ONNXRuntime,
        max_batch: int = 32,
        max_wait_ms: int = 10,
    ) -> None:
        self.runtime = runtime
        self.max_batch = max_batch
        self.max_wait_s = max_wait_ms / 1000.0
        self._queue: asyncio.Queue[_Item] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._closed = True
        if self._task is not None:
            # Wake the worker so it sees `_closed`
            await self._queue.put(
                _Item(fut=asyncio.get_running_loop().create_future(), row=np.zeros(0))
            )
            await self._task

    async def submit(self, row: np.ndarray) -> float:
        if self._closed:
            raise RuntimeError("Batcher is closed")
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put(_Item(fut=fut, row=np.asarray(row, dtype=np.float32)))
        return await fut

    async def _run(self) -> None:
        while not self._closed:
            try:
                items = await self._collect_batch()
            except asyncio.CancelledError:
                return
            if not items:
                continue
            real = [it for it in items if it.row.size > 0]  # drop sentinel
            if not real:
                continue
            try:
                xs = np.stack([it.row for it in real])
                ys = self.runtime.predict(xs)
                ys = np.asarray(ys).reshape(-1)
                for it, y in zip(real, ys):
                    if not it.fut.done():
                        it.fut.set_result(float(y))
            except Exception as e:  # noqa: BLE001
                for it in real:
                    if not it.fut.done():
                        it.fut.set_exception(e)
                log.error("batcher.error", error=str(e))

    async def _collect_batch(self) -> list[_Item]:
        first = await self._queue.get()
        items = [first]
        deadline = time.monotonic() + self.max_wait_s
        while len(items) < self.max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                items.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return items
