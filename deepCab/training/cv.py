"""Cross-validation harness. Mirrors 017-sklearn-low-level/sklearn_agent/cv.py:14-48
spec-as-factory pattern: each fold rebuilds the estimator from cfg.backend rather
than calling sklearn.clone() (which silently breaks on TF/Torch/CatBoost mixes).

For time-series data, default split is sklearn.TimeSeriesSplit on a temporally-
sorted index. KFold is available for IID benchmarks only.

Per-fold MLflow nested runs let the agent's `compare_runs` tool see fold-level
breakdowns instead of an averaged metric."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepCab.data.splits import kfold_splits, time_series_splits
from deepCab.models.factory import build_estimator
from deepCab.obs.log import get_logger
from deepCab.schemas.config import TrainConfig

log = get_logger(__name__)


@dataclass
class CVResult:
    fold_mae: list[float]
    mean_mae: float
    std_mae: float


def run_cv(
    cfg: TrainConfig,
    X: np.ndarray,
    y: np.ndarray,
    pickup_datetime_sorted: np.ndarray | None = None,
) -> CVResult:
    """Run cv.n_splits folds. If cv.kind == 'timeseries', the caller must pass
    `pickup_datetime_sorted` (a sorted-ascending datetime ndarray, same length
    as X), or pass already-sorted X/y."""
    if cfg.cv is None:
        raise ValueError("TrainConfig.cv is None — call run_cv() only when cv is configured")
    n = len(X)
    if cfg.cv.kind == "timeseries":
        if pickup_datetime_sorted is not None and not _is_sorted(pickup_datetime_sorted):
            raise ValueError("pickup_datetime_sorted must be ascending; sort X/y first.")
        # Re-use the same guard as data.splits — but here we already have arrays,
        # so build folds via raw sklearn TimeSeriesSplit on np.arange.
        import polars as pl

        toy = pl.DataFrame(
            {
                "pickup_datetime": pickup_datetime_sorted
                if pickup_datetime_sorted is not None
                else np.arange(n)
            }
        )
        splits = list(time_series_splits(toy, n_splits=cfg.cv.n_splits))
    else:
        splits = list(kfold_splits(n, n_splits=cfg.cv.n_splits, seed=cfg.seed))

    fold_maes: list[float] = []
    for i, (train_idx, test_idx) in enumerate(splits):
        # Spec-as-factory: rebuild fresh per fold; no clone().
        est = build_estimator(cfg.backend)
        est.fit(X[train_idx], y[train_idx])
        pred = np.asarray(est.predict(X[test_idx])).ravel()
        mae = float(np.mean(np.abs(pred - y[test_idx])))
        fold_maes.append(mae)
        log.info("cv.fold", fold=i, mae=mae, n_train=len(train_idx), n_test=len(test_idx))

    mean = float(np.mean(fold_maes))
    std = float(np.std(fold_maes))
    log.info("cv.done", mean_mae=mean, std_mae=std, n_splits=len(fold_maes))
    return CVResult(fold_mae=fold_maes, mean_mae=mean, std_mae=std)


def _is_sorted(a: np.ndarray) -> bool:
    return bool(np.all(a[:-1] <= a[1:]))
