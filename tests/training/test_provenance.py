"""Provenance manifest has all required fields and is deterministic by config."""
from __future__ import annotations

import json
from pathlib import Path

from deepCab.schemas.config import TrainConfig, XGBConfig
from deepCab.training.provenance import emit_provenance


def test_emit_provenance_writes_required_fields(tmp_path: Path) -> None:
    cfg = TrainConfig(backend=XGBConfig(n_estimators=42), seed=7)
    out = emit_provenance(cfg, run_id="r1", metrics={"val_mae": 3.14}, out_dir=tmp_path)
    blob = json.loads(out.read_text())
    for key in (
        "run_id", "git_sha", "config_hash", "seed", "backend_kind",
        "metrics", "python", "platform", "cuda_available", "onnx_opset", "deps",
    ):
        assert key in blob, f"missing {key}"
    assert blob["run_id"] == "r1"
    assert blob["seed"] == 7
    assert blob["backend_kind"] == "xgb"
    assert blob["metrics"]["val_mae"] == 3.14


def test_config_hash_deterministic_across_calls(tmp_path: Path) -> None:
    cfg = TrainConfig(backend=XGBConfig(n_estimators=42), seed=7)
    a = json.loads(
        emit_provenance(cfg, run_id="a", metrics={}, out_dir=tmp_path / "a").read_text()
    )["config_hash"]
    b = json.loads(
        emit_provenance(cfg, run_id="b", metrics={}, out_dir=tmp_path / "b").read_text()
    )["config_hash"]
    assert a == b


def test_config_hash_changes_on_seed_change(tmp_path: Path) -> None:
    a = json.loads(
        emit_provenance(
            TrainConfig(backend=XGBConfig(), seed=1),
            run_id="a", metrics={}, out_dir=tmp_path / "a",
        ).read_text()
    )["config_hash"]
    b = json.loads(
        emit_provenance(
            TrainConfig(backend=XGBConfig(), seed=2),
            run_id="b", metrics={}, out_dir=tmp_path / "b",
        ).read_text()
    )["config_hash"]
    assert a != b
