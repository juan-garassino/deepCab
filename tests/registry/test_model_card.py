"""Model card renders required sections + backend YAML + metric table."""
from __future__ import annotations

from pathlib import Path

import pytest

from deepCab.registry.model_card import write_model_card
from deepCab.schemas.config import TrainConfig, XGBConfig


def test_write_model_card_emits_required_sections(tmp_path: Path) -> None:
    cfg = TrainConfig(backend=XGBConfig(n_estimators=200), seed=7)
    provenance = {
        "git_sha": "abc123",
        "config_hash": "cfg-hash-x",
        "python": "3.11.9",
        "platform": "Linux-6.0",
        "onnx_opset": 17,
        "cuda_available": False,
        "cuda_version": None,
    }
    out = write_model_card(
        out_path=tmp_path / "MODEL_CARD.md",
        model_name="deepcab",
        version=3,
        cfg=cfg,
        metrics={"val_mae": 3.14, "rmse": 4.2},
        provenance=provenance,
        shap_top={"pickup_datetime": 1.2, "distance": 0.8, "pickup_location": 0.5},
    )
    body = out.read_text()
    for needle in (
        "# Model Card — deepcab v3",
        "## Backend",
        "kind: xgb",                  # rendered inside the yaml code block
        "n_estimators: 200",
        "| val_mae  | 3.1400 |",
        "**pickup_datetime**: 1.2000",
        "Git SHA: `abc123`",
        "Config hash: `cfg-hash-x`",
        "Ethical / fairness notes",
    ):
        assert needle in body, f"missing {needle!r}"


def test_model_card_handles_missing_shap_summary(tmp_path: Path) -> None:
    cfg = TrainConfig(backend=XGBConfig())
    out = write_model_card(
        out_path=tmp_path / "MODEL_CARD.md",
        model_name="deepcab",
        version=1,
        cfg=cfg,
        metrics={"val_mae": 2.0},
        provenance={},
        shap_top=None,
    )
    body = out.read_text()
    assert "_(no SHAP summary captured)_" in body


def test_model_card_handles_empty_metrics(tmp_path: Path) -> None:
    cfg = TrainConfig(backend=XGBConfig())
    out = write_model_card(
        out_path=tmp_path / "MODEL_CARD.md",
        model_name="deepcab",
        version=1,
        cfg=cfg,
        metrics={},
        provenance={},
    )
    assert "_no metrics_" in out.read_text()
