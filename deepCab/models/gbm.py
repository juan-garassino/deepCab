"""Gradient-boosted trees: XGBoost, LightGBM, CatBoost.

Each is a thin wrapper that maps its Pydantic config to the underlying
sklearn-API regressor. Tree models usually beat MLPs on tabular taxi-fare data
— they're the baseline every deep model has to beat, not the other way around.

CatBoost asymmetry: the model itself exports to ONNX but its categorical
preprocessing (CTR / target encoding) does NOT. For the current pipeline that's
fine — preprocessing happens upstream in features/pipeline.py — but document
this clearly so a future "ship CatBoost end-to-end" attempt doesn't surprise."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from deepCab.models.base import AbstractEstimator
from deepCab.schemas.config import CatBoostConfig, LGBMConfig, XGBConfig


class XGBEstimator(AbstractEstimator):
    cfg_cls = XGBConfig

    def _fit(self, X: np.ndarray, y: np.ndarray, **_: Any) -> None:
        from xgboost import XGBRegressor

        c = self.cfg
        self.model_ = XGBRegressor(
            n_estimators=c.n_estimators,
            max_depth=c.max_depth,
            learning_rate=c.learning_rate,
            subsample=c.subsample,
            tree_method="hist",
            n_jobs=-1,
        )
        self.model_.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> XGBEstimator:
        from xgboost import XGBRegressor

        est = cls(**XGBConfig().model_dump())
        est.model_ = XGBRegressor()
        est.model_.load_model(str(path))
        return est


class LGBMEstimator(AbstractEstimator):
    cfg_cls = LGBMConfig

    def _fit(self, X: np.ndarray, y: np.ndarray, **_: Any) -> None:
        from lightgbm import LGBMRegressor

        c = self.cfg
        self.model_ = LGBMRegressor(
            n_estimators=c.n_estimators,
            num_leaves=c.num_leaves,
            learning_rate=c.learning_rate,
            n_jobs=-1,
            verbose=-1,
        )
        self.model_.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model_.booster_.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> LGBMEstimator:
        import lightgbm as lgb

        est = cls(**LGBMConfig().model_dump())
        est.model_ = lgb.Booster(model_file=str(path))
        # Replace _predict with the booster's predict; booster doesn't have sklearn API
        est._predict = lambda X: est.model_.predict(X)  # type: ignore[method-assign]
        return est


class CatBoostEstimator(AbstractEstimator):
    cfg_cls = CatBoostConfig

    def _fit(self, X: np.ndarray, y: np.ndarray, **_: Any) -> None:
        from catboost import CatBoostRegressor

        c = self.cfg
        self.model_ = CatBoostRegressor(
            iterations=c.iterations,
            depth=c.depth,
            learning_rate=c.learning_rate,
            allow_writing_files=False,
            verbose=False,
        )
        self.model_.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> CatBoostEstimator:
        from catboost import CatBoostRegressor

        est = cls(**CatBoostConfig().model_dump())
        est.model_ = CatBoostRegressor()
        est.model_.load_model(str(path))
        return est
