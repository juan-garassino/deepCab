"""flatten() collapses nested Pydantic dumps to dotted keys + caps to 500 chars."""
from __future__ import annotations

from pydantic import BaseModel

from deepCab.schemas.config import TrainConfig, XGBConfig
from deepCab.training._mlflow import MLFLOW_PARAM_MAX, flatten


def test_flatten_collapses_nested() -> None:
    out = flatten({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
    assert out == {"a": "1", "b.c": "2", "b.d.e": "3"}


def test_flatten_pydantic_traincfg_round_trip() -> None:
    cfg = TrainConfig(backend=XGBConfig(n_estimators=42))
    flat = flatten(cfg.model_dump(mode="json"))
    assert flat["backend.kind"] == "xgb"  # strings pass through verbatim
    assert flat["backend.n_estimators"] == "42"
    assert flat["seed"] == "42"


def test_flatten_truncates_oversize_values() -> None:
    big = "x" * (MLFLOW_PARAM_MAX + 200)
    out = flatten({"k": big})
    assert len(out["k"]) <= MLFLOW_PARAM_MAX
    assert out["k"].endswith(")")  # truncation marker
