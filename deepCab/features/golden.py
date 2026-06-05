"""Golden-vector reference for the encoders. A second, independent
implementation of the legacy math (pure numpy + pandas, no env deps) plus a
fixed set of hand-picked rows that exercise the tricky cases:

- DST boundary (Jan = EST, Jul = EDT) — verifies tz conversion.
- Hour 0, 6, 12, 18 — sin/cos quadrants.
- Same-coord trip — distances should be 0.
- North-South vs East-West — anisotropic haversine.
- Sunday and Wednesday — weekday encoding.

The test in tests/features/test_golden.py asserts that the Polars port in
features/transformers.py produces identical output."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pygeohash as gh

EARTH_RADIUS_KM = 6371.0


# Hand-picked rows. Reproducible, NYC-bounded, mixed seasons.
GOLDEN_ROWS: list[dict] = [
    {  # Sat 2013-07-06 17:18 UTC = 13:18 EDT
        "pickup_datetime": "2013-07-06 17:18:00 UTC",
        "pickup_latitude": 40.783282,
        "pickup_longitude": -73.950655,
        "dropoff_latitude": 40.769802,
        "dropoff_longitude": -73.984365,
        "passenger_count": 2,
    },
    {  # Wed 2014-01-15 05:00 UTC = 00:00 EST (DST off)
        "pickup_datetime": "2014-01-15 05:00:00 UTC",
        "pickup_latitude": 40.7128,
        "pickup_longitude": -74.0060,
        "dropoff_latitude": 40.7128,
        "dropoff_longitude": -74.0060,
        "passenger_count": 1,
    },
    {  # Sun 2015-06-21 16:00 UTC = 12:00 EDT — purely north-south
        "pickup_datetime": "2015-06-21 16:00:00 UTC",
        "pickup_latitude": 40.75,
        "pickup_longitude": -73.99,
        "dropoff_latitude": 40.85,
        "dropoff_longitude": -73.99,
        "passenger_count": 3,
    },
    {  # Mon 2012-12-31 23:00 UTC = 18:00 EST — purely east-west
        "pickup_datetime": "2012-12-31 23:00:00 UTC",
        "pickup_latitude": 40.70,
        "pickup_longitude": -74.05,
        "dropoff_latitude": 40.70,
        "dropoff_longitude": -73.75,
        "passenger_count": 4,
    },
    {  # Fri 2016-03-11 10:00 UTC = 05:00 EST — pre-DST 2016 (DST starts Mar 13)
        "pickup_datetime": "2016-03-11 10:00:00 UTC",
        "pickup_latitude": 40.78,
        "pickup_longitude": -73.97,
        "dropoff_latitude": 40.74,
        "dropoff_longitude": -73.99,
        "passenger_count": 1,
    },
]


def as_dataframe() -> pd.DataFrame:
    return pd.DataFrame(GOLDEN_ROWS)


# ---------------------------------------------------------------------------
# Reference implementation — independent re-do of legacy ml_logic/encoders.py
# with the env dependency stripped out. Phase 3 deletes the legacy module; this
# stays as the canonical "what the math should produce".
# ---------------------------------------------------------------------------


def reference_time_features(X: pd.DataFrame) -> np.ndarray:
    dt = pd.to_datetime(X["pickup_datetime"], utc=True).dt.tz_convert("America/New_York").dt
    hour = dt.hour
    return np.stack(
        [
            np.sin(2 * math.pi / 24 * hour),
            np.cos(2 * math.pi / 24 * hour),
            dt.weekday,
            dt.month,
            dt.year,
        ],
        axis=1,
    )


def reference_lonlat_features(X: pd.DataFrame) -> pd.DataFrame:
    lat1 = np.radians(X["pickup_latitude"])
    lon1 = np.radians(X["pickup_longitude"])
    lat2 = np.radians(X["dropoff_latitude"])
    lon2 = np.radians(X["dropoff_longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    haversine_km = 2 * np.arcsin(np.sqrt(a)) * EARTH_RADIUS_KM
    manhattan_km = (np.abs(dlat) + np.abs(dlon)) * EARTH_RADIUS_KM
    return pd.DataFrame({"haversine_km": haversine_km, "manhattan_km": manhattan_km})


def reference_geohash(X: pd.DataFrame, precision: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geohash_pickup": [
                gh.encode(r["pickup_latitude"], r["pickup_longitude"], precision=precision)
                for _, r in X.iterrows()
            ],
            "geohash_dropoff": [
                gh.encode(r["dropoff_latitude"], r["dropoff_longitude"], precision=precision)
                for _, r in X.iterrows()
            ],
        }
    )
