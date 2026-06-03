"""65-d preprocessed SHAP -> user-meaningful feature groups.

The legacy preprocessor expands 6 raw inputs (passenger, pickup_datetime, and 4
lon/lat coords) into 65 derived features via one-hots, scalers, and geohash
districts. SHAP at the 65-d level is mathematically correct but unreadable:
"pickup_geohash_dr5rk = +0.12" doesn't help anyone. We collapse to functional
groups by summing component attributions:

| Group              | 65-d cols | What it captures                                  |
|--------------------|-----------|---------------------------------------------------|
| passenger          | 0         | passenger_count                                   |
| pickup_datetime    | 1..22     | OHE day-of-week + month + hour_sin/cos + year     |
| distance           | 23..24    | haversine + manhattan (functions of all 4 coords) |
| pickup_location    | 25..44    | OHE pickup_geohash5 over top-20 NYC districts     |
| dropoff_location   | 45..64    | OHE dropoff_geohash5 over top-20 NYC districts    |

Note the asymmetry: pickup_lat and pickup_lon don't separate cleanly — the
geohash encoding fuses them into one categorical, and the haversine/manhattan
fuses all four. Splitting further would be misleading."""
from __future__ import annotations

import numpy as np

# Column ranges in the 65-d output of features.pipeline.preprocess_features.
COLUMN_GROUPS: dict[str, tuple[int, int]] = {
    "passenger": (0, 1),
    "pickup_datetime": (1, 23),
    "distance": (23, 25),
    "pickup_location": (25, 45),
    "dropoff_location": (45, 65),
}

EXPECTED_DIM = 65
FEATURE_ORDER = list(COLUMN_GROUPS.keys())


def aggregate_shap(values: np.ndarray) -> dict[str, float]:
    """Sum SHAP values along the column axis per COLUMN_GROUPS. Accepts either
    a 1-D (per-row) or 2-D (n_rows, 65) array.

    Returns dict ordered as FEATURE_ORDER for stable JSON serialization."""
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Expected 65-d SHAP vector, got shape {arr.shape}. "
            f"Did the preprocessor change? Update COLUMN_GROUPS."
        )
    out: dict[str, float] = {}
    for name, (start, end) in COLUMN_GROUPS.items():
        out[name] = float(arr[:, start:end].sum())
    return out


def aggregate_global(values_2d: np.ndarray) -> dict[str, float]:
    """Compute mean(|SHAP|) per group — the global feature-importance summary.
    Cached at startup from a holdout background sample (see explain/cache.py)."""
    arr = np.asarray(values_2d)
    if arr.ndim != 2 or arr.shape[1] != EXPECTED_DIM:
        raise ValueError(f"Expected (n, 65) SHAP matrix, got {arr.shape}")
    out: dict[str, float] = {}
    for name, (start, end) in COLUMN_GROUPS.items():
        out[name] = float(np.mean(np.abs(arr[:, start:end]).sum(axis=1)))
    return out
