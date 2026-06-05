"""Sklearn ColumnTransformer assembly. Internally calls the Polars-backed
transformers in features/transformers.py (Phase 2 port).

Output is the legacy 65-d numpy array, so downstream estimators (esp. the
existing TF MLP from Phase 1) see exactly the input shape they were trained on.
Sklearn fit_transform is stateless here — all transformers either are stateless
themselves or have explicit `categories=` so no vocab learning happens — but
calling `fit_transform` per call is the safe contract. Phase 6+ will swap this
for a fully Polars assembly that caches a fitted state on disk."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from deepCab.features.transformers import (
    compute_geohash,
    transform_lonlat_features,
    transform_time_features,
)

# Top-20 NYC geohash-5 districts (~99% pickup/dropoff coverage), from the
# legacy notebook analysis. Hard-coded for now — Phase 6 can re-derive
# per-dataset if needed.
GEOHASH_DISTRICTS = [
    "dr5ru",
    "dr5rs",
    "dr5rv",
    "dr72h",
    "dr72j",
    "dr5re",
    "dr5rk",
    "dr5rz",
    "dr5ry",
    "dr5rt",
    "dr5rg",
    "dr5x1",
    "dr5x0",
    "dr72m",
    "dr5rm",
    "dr5rx",
    "dr5x2",
    "dr5rw",
    "dr5rh",
    "dr5x8",
]

YEAR_MIN, YEAR_MAX = 2009, 2019
PASSENGER_MIN, PASSENGER_MAX = 1, 8
DIST_MIN, DIST_MAX = 0, 100

LONLAT_COLS = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]


def _make_preprocessor() -> ColumnTransformer:
    passenger_pipe = FunctionTransformer(
        lambda p: (p - PASSENGER_MIN) / (PASSENGER_MAX - PASSENGER_MIN)
    )
    distance_pipe = make_pipeline(
        FunctionTransformer(transform_lonlat_features),
        FunctionTransformer(lambda d: (d - DIST_MIN) / (DIST_MAX - DIST_MIN)),
    )
    time_pipe = make_pipeline(
        FunctionTransformer(transform_time_features),
        make_column_transformer(
            (
                OneHotEncoder(
                    # sklearn 1.5+ expects a list of category arrays (one per
                    # feature being encoded), not a dict — the legacy dict form
                    # was an undocumented quirk in older sklearn.
                    categories=[np.arange(0, 7), np.arange(1, 13)],
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                [2, 3],
            ),
            (
                FunctionTransformer(lambda y: (y - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)),
                [4],
            ),
            remainder="passthrough",
        ),
    )
    geohash_pipe = make_pipeline(
        FunctionTransformer(compute_geohash),
        OneHotEncoder(
            categories=[GEOHASH_DISTRICTS, GEOHASH_DISTRICTS],
            handle_unknown="ignore",
            sparse_output=False,
        ),
    )
    return ColumnTransformer(
        [
            ("passenger", passenger_pipe, ["passenger_count"]),
            ("time", time_pipe, ["pickup_datetime"]),
            ("distance", distance_pipe, LONLAT_COLS),
            ("geohash", geohash_pipe, LONLAT_COLS),
        ],
        n_jobs=1,  # n_jobs=-1 forks tensorflow imports; n_jobs=1 is safer
    )


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    """Stateless: (N, 7-col) -> (N, 65) numpy. Categorical encoders use
    explicit `categories=` so no vocab is learned across calls."""
    return _make_preprocessor().fit_transform(X)
