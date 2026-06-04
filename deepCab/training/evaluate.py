"""Eval a registered model on a held-out partition. Pure function."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepCab.obs.log import get_logger
from deepCab.schemas.config import DataRef
from deepCab.training.preprocess import preprocess

log = get_logger(__name__)


@dataclass
class EvalResult:
    mae: float
    rmse: float
    n: int


def evaluate(estimator, data: DataRef, split: str = "val") -> EvalResult:
    X, y = preprocess(data, split=split)
    pred = np.asarray(estimator.predict(X)).ravel()
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    log.info("evaluate.done", mae=mae, rmse=rmse, n=len(y))
    return EvalResult(mae=mae, rmse=rmse, n=len(y))
