"""Unit tests for TrainingService.

We don't actually train here — that's covered by tests/training/. We just
verify the task-table contract: start mints a 12-char id and registers a
task; status returns the record; unknown id raises."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import BackgroundTasks

from deepCab.api.services.train import TrainingService, UnknownTaskError
from deepCab.api.state import STATE, TaskRecord


@pytest.fixture(autouse=True)
def _clean_state():
    STATE.model = None
    STATE.tasks.clear()
    yield
    STATE.model = None
    STATE.tasks.clear()


def _make_req():
    from deepCab.schemas.config import DataRef, TrainConfig, XGBConfig

    return type(
        "_R",
        (),
        {
            "config": TrainConfig(
                backend=XGBConfig(),
                data=DataRef(size="1k", validation_size="1k"),
                seed=1,
            )
        },
    )()


def test_start_returns_12_char_task_id_and_registers_task() -> None:
    svc = TrainingService()
    bg = BackgroundTasks()  # we don't run the bg work in this unit test
    req = _make_req()

    resp = asyncio.run(svc.start(req, bg))
    assert isinstance(resp.task_id, str)
    assert len(resp.task_id) == 12
    assert resp.task_id in STATE.tasks


def test_status_unknown_id_raises_unknown_task_error() -> None:
    svc = TrainingService()
    with pytest.raises(UnknownTaskError):
        asyncio.run(svc.status("does-not-exist"))


def test_status_returns_record_state() -> None:
    svc = TrainingService()
    STATE.upsert_task(TaskRecord(task_id="abc123def456", status="succeeded", run_id="r1"))
    resp = asyncio.run(svc.status("abc123def456"))
    assert resp.task_id == "abc123def456"
    assert resp.status == "succeeded"
    assert resp.run_id == "r1"
