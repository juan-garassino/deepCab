"""BackendConfig discriminated union must round-trip JSON cleanly. Catches the
Pydantic v2 'discriminator-as-Literal' gotcha that bites when MLflow re-loads
logged configs."""
from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from deepCab.schemas.config import (
    BackendConfig,
    CatBoostConfig,
    FTTransformerConfig,
    LGBMConfig,
    TFMLPConfig,
    TorchMLPConfig,
    TrainConfig,
    XGBConfig,
)

BACKEND_ADAPTER = TypeAdapter(BackendConfig)


@pytest.mark.parametrize(
    "cfg",
    [
        TFMLPConfig(),
        TorchMLPConfig(),
        XGBConfig(),
        LGBMConfig(),
        CatBoostConfig(),
        FTTransformerConfig(),
    ],
)
def test_backend_config_roundtrip(cfg) -> None:
    payload = cfg.model_dump()
    rebuilt = BACKEND_ADAPTER.validate_python(payload)
    assert rebuilt.kind == cfg.kind
    assert rebuilt.model_dump() == payload


def test_backend_discriminator_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        BACKEND_ADAPTER.validate_python({"kind": "nope"})


def test_train_config_compose() -> None:
    tc = TrainConfig(backend=XGBConfig(n_estimators=42), seed=7)
    assert tc.backend.kind == "xgb"
    assert tc.backend.n_estimators == 42
    assert tc.seed == 7
    assert tc.cv is None
