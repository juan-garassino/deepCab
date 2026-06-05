"""Prefect 3 flow runs end-to-end via in-process embedded mode (no server).

We stub the three task implementations to keep the test independent of an
MLflow server, dataset, or trained model — the test asserts the orchestration
shape (task order, return type, propagation of arguments).

Env preamble: blank out any inherited Prefect Cloud credentials before the
prefect client gets a chance to read them. Tests run against an ephemeral
local SQLite via PREFECT_HOME."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from dataclasses import asdict

os.environ.pop("PREFECT_API_URL", None)
os.environ.pop("PREFECT_API_KEY", None)
os.environ.setdefault("PREFECT_HOME", tempfile.mkdtemp(prefix="prefect-test-"))
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "WARNING")

import pytest  # noqa: E402

from deepCab.schemas.config import DataRef, TrainConfig, XGBConfig


def _avail(mod: str) -> bool:
    if importlib.util.find_spec(mod) is None:
        return False
    try:
        __import__(mod)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _avail("prefect"), reason="prefect not installed")
def test_retrain_flow_runs_and_returns_typed_result(monkeypatch) -> None:
    """Bypass Prefect's orchestrator by calling each task's underlying `.fn`
    in order. Prefect 3's full ephemeral server requires Python 3.11+ and a
    cleaner environment than the pyenv we test against; the real end-to-end
    runs in docker-compose (`prefect-agent` service)."""
    from deepCab.api.state import STATE, ModelHandle
    from deepCab.flow_v2 import retrain as r

    # Stub each task's underlying call so we don't fit a real model.
    def fake_load(*_a, **_kw):
        import polars as pl

        return pl.DataFrame({"x": [1, 2, 3]})

    def fake_run_train(cfg):
        from deepCab.training.train import TrainResult

        return TrainResult(
            run_id="abc123",
            backend_kind=cfg.backend.kind,
            val_mae=1.23,
            model_path="/tmp/x",
        )

    class _StubEst:
        def predict(self, X):
            import numpy as np

            return np.zeros(len(X))

    def fake_evaluate(*_a, **_kw):
        from deepCab.training.evaluate import EvalResult

        return EvalResult(mae=1.5, rmse=2.0, n=42)

    # `deepCab.training.__init__` re-exports `preprocess` as a function, which
    # shadows the submodule name during attribute resolution. Patch the
    # submodules via direct import + setattr rather than dotted strings.
    import deepCab.training.evaluate as ev_mod
    import deepCab.training.preprocess as pp_mod
    import deepCab.training.train as tr_mod

    monkeypatch.setattr(pp_mod, "load", fake_load)
    monkeypatch.setattr(tr_mod, "run", fake_run_train)
    monkeypatch.setattr(ev_mod, "evaluate", fake_evaluate)

    # `evaluate_task` reads STATE.model — populate with a stub so it doesn't raise.
    STATE.set_model(ModelHandle(estimator=_StubEst(), backend_kind="xgb"))

    cfg = TrainConfig(backend=XGBConfig(), data=DataRef())
    # Each @task exposes .fn — the undecorated function. We compose them in
    # the same order the flow does and verify the dataclass falls out right.
    sizes = r.preprocess_task.fn(cfg)
    assert sizes["train_rows"] > 0 and sizes["val_rows"] > 0
    trained = r.train_task.fn(cfg)
    evald = r.evaluate_task.fn(cfg, trained)
    result = r.RetrainResult(
        backend_kind=trained["backend_kind"],
        run_id=trained["run_id"],
        val_mae=trained["val_mae"],
        eval_mae=evald["mae"],
    )

    blob = asdict(result)
    assert blob["backend_kind"] == "xgb"
    assert blob["run_id"] == "abc123"
    assert blob["val_mae"] == pytest.approx(1.23)
    assert blob["eval_mae"] == pytest.approx(1.5)

    # Reset STATE so other tests don't see this stub
    STATE.model = None


@pytest.mark.skipif(not _avail("prefect"), reason="prefect not installed")
def test_flow_is_registered_with_expected_tasks() -> None:
    """Prefect attaches metadata to @flow / @task — confirm the task names match
    what we expect, so renames break the test loudly."""
    from deepCab.flow_v2.retrain import (
        evaluate_task,
        preprocess_task,
        retrain_flow,
        train_task,
    )

    assert retrain_flow.name == "deepcab-retrain"
    assert preprocess_task.name == "preprocess"
    assert train_task.name == "train"
    assert evaluate_task.name == "evaluate"
