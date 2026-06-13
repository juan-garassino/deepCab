"""PromotionService threshold + alias-flip behaviour. Pure in-process tests —
no MLflow server. Injects a fake client + loader."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from deepCab.registry.promotion import PromotionInputs, PromotionService
from deepCab.schemas.config import DataRef
from deepCab.schemas.enums import DataSize
from deepCab.training.evaluate import EvalResult


class _StubEstimator:
    """Predicts a constant — `_score` resolves to a deterministic MAE."""

    def __init__(self, constant: float) -> None:
        self.constant = constant

    def predict(self, X) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=self.constant, dtype=np.float32)


@dataclass
class _FakeVersion:
    version: str


class _StubClient:
    """Minimal MlflowClient stand-in: holds the current @champion version."""

    def __init__(self, champion_version: str | None = None) -> None:
        self.champion_version = champion_version

    def get_model_version_by_alias(self, model: str, alias: str) -> _FakeVersion:
        if alias == "champion" and self.champion_version is not None:
            return _FakeVersion(self.champion_version)
        raise RuntimeError("no alias")


@pytest.fixture
def ref_data() -> DataRef:
    return DataRef(size=DataSize.S1K, validation_size=DataSize.S1K)


def _patch_set_alias():
    """Skip the actual MLflow alias write + notify side-effects."""
    return patch("deepCab.registry.promotion.set_alias", autospec=True)


def _eval_returning(mae: float):
    """Patch `evaluate` to skip the preprocess call and just emit a constant MAE."""
    return patch(
        "deepCab.registry.promotion.evaluate",
        return_value=EvalResult(mae=mae, rmse=mae, n=100),
    )


def test_bootstrap_no_existing_champion_promotes(ref_data: DataRef) -> None:
    svc = PromotionService(
        client=_StubClient(champion_version=None), loader=lambda *_: _StubEstimator(0.0)
    )
    with _patch_set_alias() as set_alias_mock, _eval_returning(5.0):
        with patch.dict("os.environ", {"MLFLOW_MODEL_NAME": "deepcab"}, clear=False):
            from deepCab.schemas.settings import get_settings

            get_settings.cache_clear()
            result = svc.maybe_promote(
                PromotionInputs(challenger_version="3", reference_data=ref_data)
            )

    assert result.promoted is True
    assert result.reason == "no-existing-champion"
    assert result.new_champion_version == "3"
    # Two set_alias calls expected: @challenger then @champion (no @legacy).
    aliases = [c.kwargs["alias"] for c in set_alias_mock.call_args_list]
    assert "challenger" in aliases and "champion" in aliases
    assert "legacy" not in aliases


def test_below_threshold_does_not_promote(ref_data: DataRef) -> None:
    svc = PromotionService(
        client=_StubClient(champion_version="1"), loader=lambda *_: _StubEstimator(0.0)
    )
    # champion: 5.0, challenger: 4.9 → 2% improvement; threshold 5% → no promote.
    with (
        _patch_set_alias() as set_alias_mock,
        patch(
            "deepCab.registry.promotion.evaluate",
            side_effect=[
                EvalResult(mae=4.9, rmse=4.9, n=100),  # challenger first
                EvalResult(mae=5.0, rmse=5.0, n=100),  # champion second
            ],
        ),
    ):
        from deepCab.schemas.settings import get_settings

        get_settings.cache_clear()
        result = svc.maybe_promote(
            PromotionInputs(
                challenger_version="2",
                reference_data=ref_data,
                improvement_threshold=0.05,
                model_name="deepcab",
            )
        )

    assert result.promoted is False
    assert result.reason == "below-threshold"
    assert result.new_champion_version == "1"  # champion unchanged
    aliases = [c.kwargs["alias"] for c in set_alias_mock.call_args_list]
    assert aliases == ["challenger"]  # only @challenger was set


def test_beats_threshold_promotes_and_legacies_old(ref_data: DataRef) -> None:
    svc = PromotionService(
        client=_StubClient(champion_version="1"), loader=lambda *_: _StubEstimator(0.0)
    )
    # champion: 5.0, challenger: 4.0 → 20% improvement; threshold 5% → promote.
    with (
        _patch_set_alias() as set_alias_mock,
        patch(
            "deepCab.registry.promotion.evaluate",
            side_effect=[
                EvalResult(mae=4.0, rmse=4.0, n=100),  # challenger
                EvalResult(mae=5.0, rmse=5.0, n=100),  # champion
            ],
        ),
    ):
        from deepCab.schemas.settings import get_settings

        get_settings.cache_clear()
        result = svc.maybe_promote(
            PromotionInputs(
                challenger_version="2",
                reference_data=ref_data,
                improvement_threshold=0.05,
                model_name="deepcab",
            )
        )

    assert result.promoted is True
    assert result.reason == "beats-threshold"
    assert result.old_champion_version == "1"
    assert result.new_champion_version == "2"

    aliases_in_order = [
        (c.kwargs["alias"], c.kwargs["version"]) for c in set_alias_mock.call_args_list
    ]
    # challenger set first (always), then @legacy on old champion, then @champion on new.
    assert aliases_in_order == [
        ("challenger", "2"),
        ("legacy", "1"),
        ("champion", "2"),
    ]


def test_exact_threshold_does_not_promote(ref_data: DataRef) -> None:
    """5% champion-relative improvement at exactly 5% threshold → strictly-less-than fails."""
    svc = PromotionService(
        client=_StubClient(champion_version="1"), loader=lambda *_: _StubEstimator(0.0)
    )
    with (
        _patch_set_alias(),
        patch(
            "deepCab.registry.promotion.evaluate",
            side_effect=[
                EvalResult(mae=4.75, rmse=4.75, n=100),  # challenger: 5.0 * 0.95
                EvalResult(mae=5.0, rmse=5.0, n=100),  # champion
            ],
        ),
    ):
        from deepCab.schemas.settings import get_settings

        get_settings.cache_clear()
        result = svc.maybe_promote(
            PromotionInputs(
                challenger_version="2",
                reference_data=ref_data,
                improvement_threshold=0.05,
                model_name="deepcab",
            )
        )

    assert result.promoted is False


def test_same_version_already_champion_is_noop(ref_data: DataRef) -> None:
    svc = PromotionService(
        client=_StubClient(champion_version="7"), loader=lambda *_: _StubEstimator(0.0)
    )
    with _patch_set_alias() as set_alias_mock, _eval_returning(3.0):
        from deepCab.schemas.settings import get_settings

        get_settings.cache_clear()
        result = svc.maybe_promote(
            PromotionInputs(challenger_version="7", reference_data=ref_data, model_name="deepcab")
        )

    assert result.promoted is False
    assert result.reason == "challenger-already-champion"
    aliases = [c.kwargs["alias"] for c in set_alias_mock.call_args_list]
    assert aliases == ["challenger"]  # idempotent — only re-stamps @challenger
