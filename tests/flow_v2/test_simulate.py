"""Continuous-training simulation flow. Stubs the executor + PromotionService
so the test doesn't fit a real model or hit MLflow."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from datetime import datetime, timedelta

os.environ.pop("PREFECT_API_URL", None)
os.environ.pop("PREFECT_API_KEY", None)
os.environ.setdefault("PREFECT_HOME", tempfile.mkdtemp(prefix="prefect-test-"))
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "WARNING")

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from deepCab.schemas.config import DataRef, TrainConfig, XGBConfig  # noqa: E402
from deepCab.schemas.enums import DataSize  # noqa: E402


def _avail(mod: str) -> bool:
    if importlib.util.find_spec(mod) is None:
        return False
    try:
        __import__(mod)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _avail("prefect"), reason="prefect not installed")
def test_chunks_splits_half_open() -> None:
    from deepCab.flow_v2.simulate import _chunks

    chunks = _chunks(datetime(2014, 1, 1), datetime(2014, 1, 29), timedelta(days=7))
    assert len(chunks) == 4
    # First chunk start == window start; last chunk end == window end.
    assert chunks[0][0] == datetime(2014, 1, 1)
    assert chunks[-1][1] == datetime(2014, 1, 29)
    # Half-open: chunk_i end == chunk_{i+1} start.
    for i in range(len(chunks) - 1):
        assert chunks[i][1] == chunks[i + 1][0]


@pytest.mark.skipif(not _avail("prefect"), reason="prefect not installed")
def test_chunks_truncates_final_chunk_at_end() -> None:
    from deepCab.flow_v2.simulate import _chunks

    # 10-day window, 7-day chunks → 2 chunks: [d0,d7) and [d7,d10) (truncated).
    chunks = _chunks(datetime(2014, 1, 1), datetime(2014, 1, 11), timedelta(days=7))
    assert len(chunks) == 2
    assert chunks[-1][1] == datetime(2014, 1, 11)


@pytest.mark.skipif(not _avail("prefect"), reason="prefect not installed")
def test_simulate_flow_aggregates_chunks_and_propagates_promotions() -> None:
    """Stub everything inside the per-chunk envelope; verify aggregation."""
    from deepCab.flow_v2 import simulate as sim
    from deepCab.registry.promotion import PromotionResult
    from deepCab.training.evaluate import EvalResult
    from deepCab.training.train import TrainResult

    cfg = TrainConfig(backend=XGBConfig(), data=DataRef())
    ref = DataRef(size=DataSize.S1K, validation_size=DataSize.S1K)

    # Stub executor: emits a deterministic run_id, no actual training.
    class _StubExecutor:
        estimated_cost_per_call_usd = 0.07

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def __call__(self, cfg, run_name, where):
            self.calls.append((cfg.backend.kind, run_name, where))
            # Return a TrainResult with a deterministic run_id mirroring run_name.
            return TrainResult(
                run_id=f"run-{len(self.calls)}",
                backend_kind=cfg.backend.kind,
                val_mae=2.0,
                model_path="",
            )

    executor = _StubExecutor()

    # Stub the rest of the envelope: evaluate + PromotionService.
    promote_results = iter(
        [
            PromotionResult(
                promoted=True, reason="no-existing-champion",
                challenger_version="run-1", challenger_metric=2.5,
                old_champion_version=None, champion_metric=None,
                new_champion_version="run-1",
            ),
            PromotionResult(
                promoted=False, reason="below-threshold",
                challenger_version="run-2", challenger_metric=2.7,
                old_champion_version="run-1", champion_metric=2.5,
                new_champion_version="run-1",
            ),
            PromotionResult(
                promoted=True, reason="beats-threshold",
                challenger_version="run-3", challenger_metric=2.0,
                old_champion_version="run-1", champion_metric=2.5,
                new_champion_version="run-3",
            ),
        ]
    )

    class _StubPromotionService:
        def maybe_promote(self, inputs):
            return next(promote_results)

    # evaluate must not need a real STATE.model
    with patch.object(
        sim, "_evaluate",
        side_effect=lambda cfg, tr, ref: EvalResult(mae=2.5 - tr.run_id[-1:].__hash__() % 10 * 0.0, rmse=3.0, n=100),
    ), patch.object(sim.notify, "notify_flow_event"), patch.object(sim.notify, "post"):
        result = sim._simulate_impl(
            cfg=cfg,
            reference_data=ref,
            time_window_start=datetime(2014, 1, 1),
            time_window_end=datetime(2014, 1, 22),
            chunk_period=timedelta(days=7),
            promotion_threshold=0.05,
            executor=executor,
            promotion_service=_StubPromotionService(),
        )

    assert len(result.chunks) == 3
    assert result.promotions == 2  # chunk 0 + chunk 2
    assert result.final_champion_version == "run-3"
    # Each chunk got its own deterministic run name + a half-open WHERE clause.
    run_names = [c[1] for c in executor.calls]
    assert run_names == ["sim-000-xgb", "sim-001-xgb", "sim-002-xgb"]
    wheres = [c[2] for c in executor.calls]
    assert all(">=" in w and "<" in w and "<=" not in w for w in wheres)
    assert result.estimated_total_cost_usd == pytest.approx(3 * 0.07)


@pytest.mark.skipif(not _avail("prefect"), reason="prefect not installed")
def test_flow_and_tasks_registered_with_expected_names() -> None:
    from deepCab.flow_v2.simulate import (
        evaluate_chunk_task,
        ingest_chunk_task,
        notify_chunk_task,
        promote_task,
        simulate_flow,
        train_chunk_task,
    )

    assert simulate_flow.name == "deepcab-simulate"
    assert ingest_chunk_task.name == "ingest_chunk"
    assert train_chunk_task.name == "train_chunk"
    assert evaluate_chunk_task.name == "evaluate_chunk"
    assert promote_task.name == "promote"
    assert notify_chunk_task.name == "notify_chunk"
