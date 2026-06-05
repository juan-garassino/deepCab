"""Single-row + batch predict. The agent's `predict` tool calls these; the
FastAPI `/predict` router (Phase 6) calls these too."""

from __future__ import annotations

import numpy as np
import pandas as pd

from deepCab.features.pipeline import preprocess_features
from deepCab.schemas.data import FeatureRow


def predict_one(estimator, row: FeatureRow) -> float:
    X = preprocess_features(pd.DataFrame([row.model_dump()]))
    return float(np.asarray(estimator.predict(X)).ravel()[0])


def predict_many(estimator, rows: list[FeatureRow]) -> list[float]:
    if not rows:
        return []
    X = preprocess_features(pd.DataFrame([r.model_dump() for r in rows]))
    return [float(v) for v in np.asarray(estimator.predict(X)).ravel()]
