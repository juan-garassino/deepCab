"""Optuna study runs N trials, returns the best config validated against the
backend's Pydantic schema. Determinism: same seed -> same best params."""
from __future__ import annotations

import importlib.util

import pytest

from deepCab.schemas.config import HPOConfig, LGBMConfig, TrainConfig
from deepCab.training.hpo import tune


def _avail(mod: str) -> bool:
    if importlib.util.find_spec(mod) is None:
        return False
    try:
        __import__(mod)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _avail("optuna"), reason="optuna not installed")
def test_tune_returns_validated_best_cfg() -> None:
    base = TrainConfig(
        backend=LGBMConfig(),
        hpo=HPOConfig(n_trials=4, sampler="random"),
        seed=1,
    )
    # Synthetic objective: minimize the suggested num_leaves
    def objective(cfg: TrainConfig) -> float:
        return float(cfg.backend.num_leaves)  # type: ignore[union-attr]

    res = tune(base, objective_fn=objective)
    assert res.n_trials == 4
    assert res.best_backend_cfg.kind == "lgbm"
    # Best should be the smallest num_leaves we sampled
    assert res.best_value == res.best_backend_cfg.num_leaves  # type: ignore[union-attr]


@pytest.mark.skipif(not _avail("optuna"), reason="optuna not installed")
def test_tune_determinism_across_runs() -> None:
    def objective(cfg: TrainConfig) -> float:
        return float(cfg.backend.num_leaves)  # type: ignore[union-attr]

    base = TrainConfig(
        backend=LGBMConfig(),
        hpo=HPOConfig(n_trials=5, sampler="tpe"),
        seed=42,
    )
    a = tune(base, objective_fn=objective).best_params
    b = tune(base, objective_fn=objective).best_params
    assert a == b
