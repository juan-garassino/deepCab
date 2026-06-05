"""Stateless feature transforms, ported to Polars expressions from legacy
ml_logic/encoders.py. Same math, vectorized via polars Expr instead of numpy +
pandas-apply (geohash still uses map_elements — pygeohash isn't vectorized).

Public API surface mirrors the legacy module so the sklearn FunctionTransformer
wrappers in features/pipeline.py keep working: each function accepts a pandas
DataFrame, returns a numpy.ndarray. Internally we convert to Polars, run the
expression, and emit numpy. Phase 3 will replace the wrappers with native Polars
pipelines and drop the pandas/numpy round-trip.

Gotcha (caught by features/golden.py): polars `dt.weekday()` is 1=Monday, pandas
`dt.dayofweek` is 0=Monday. We subtract 1 to preserve legacy semantics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pygeohash as gh

EARTH_RADIUS_KM = 6371.0
NYC_TZ = "America/New_York"


def transform_time_features(X: pd.DataFrame) -> np.ndarray:
    """Output columns (legacy order): hour_sin, hour_cos, dow (0=Mon), month, year."""
    df = pl.from_pandas(X[["pickup_datetime"]])
    # Parse strings or naive timestamps; accept the legacy "YYYY-MM-DD HH:MM:SS UTC"
    # format by stripping the trailing " UTC" before parsing. Already-tz-aware
    # datetimes pass through.
    dtype = df.schema["pickup_datetime"]
    if dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("pickup_datetime")
            .str.replace(r" UTC$", "")
            .str.to_datetime(time_unit="us")
            .dt.replace_time_zone("UTC")
        )
    elif not isinstance(dtype, pl.Datetime):
        raise TypeError(f"pickup_datetime must be Utf8 or Datetime, got {dtype}")
    elif dtype.time_zone is None:
        df = df.with_columns(pl.col("pickup_datetime").dt.replace_time_zone("UTC"))

    df = df.with_columns(pl.col("pickup_datetime").dt.convert_time_zone(NYC_TZ).alias("local"))
    df = df.with_columns(
        pl.col("local").dt.hour().alias("h"),
        (pl.col("local").dt.weekday() - 1).alias("dow"),  # polars 1=Mon -> 0=Mon
        pl.col("local").dt.month().alias("month"),
        pl.col("local").dt.year().alias("year"),
    )
    df = df.with_columns(
        (2 * math.pi / 24 * pl.col("h")).sin().alias("hour_sin"),
        (2 * math.pi / 24 * pl.col("h")).cos().alias("hour_cos"),
    )
    out = df.select("hour_sin", "hour_cos", "dow", "month", "year").to_numpy()
    return out


def transform_lonlat_features(X: pd.DataFrame) -> pd.DataFrame:
    """Haversine and Manhattan distances (km) between pickup and dropoff.
    Returned columns ordered (haversine, manhattan) to match legacy."""
    df = pl.from_pandas(
        X[
            [
                "pickup_latitude",
                "pickup_longitude",
                "dropoff_latitude",
                "dropoff_longitude",
            ]
        ]
    )
    df = df.with_columns(
        pl.col("pickup_latitude").radians().alias("lat1"),
        pl.col("pickup_longitude").radians().alias("lon1"),
        pl.col("dropoff_latitude").radians().alias("lat2"),
        pl.col("dropoff_longitude").radians().alias("lon2"),
    )
    df = df.with_columns(
        (pl.col("lat2") - pl.col("lat1")).alias("dlat"),
        (pl.col("lon2") - pl.col("lon1")).alias("dlon"),
    )
    df = df.with_columns(
        # haversine
        (
            (pl.col("dlat") / 2.0).sin().pow(2)
            + pl.col("lat1").cos() * pl.col("lat2").cos() * (pl.col("dlon") / 2.0).sin().pow(2)
        )
        .sqrt()
        .arcsin()
        .mul(2 * EARTH_RADIUS_KM)
        .alias("haversine_km"),
        # manhattan
        (pl.col("dlat").abs() + pl.col("dlon").abs()).mul(EARTH_RADIUS_KM).alias("manhattan_km"),
    )
    return df.select("haversine_km", "manhattan_km").to_pandas()


def compute_geohash(X: pd.DataFrame, precision: int = 5) -> pd.DataFrame:
    """Add 'geohash_pickup' / 'geohash_dropoff' columns of given precision. pygeohash
    isn't vectorized; we use map_elements. The output dtype is pl.Utf8."""
    df = pl.from_pandas(
        X[["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]]
    )

    def _enc(row: dict) -> str:
        return gh.encode(row["lat"], row["lon"], precision=precision)

    df = df.with_columns(
        pl.struct([pl.col("pickup_latitude").alias("lat"), pl.col("pickup_longitude").alias("lon")])
        .map_elements(_enc, return_dtype=pl.Utf8)
        .alias("geohash_pickup"),
        pl.struct(
            [pl.col("dropoff_latitude").alias("lat"), pl.col("dropoff_longitude").alias("lon")]
        )
        .map_elements(_enc, return_dtype=pl.Utf8)
        .alias("geohash_dropoff"),
    )
    out = df.select("geohash_pickup", "geohash_dropoff").to_pandas()

    # Legacy `compute_geohash` mutates the input in-place and returns the same two
    # columns; preserve that contract for downstream sklearn FunctionTransformers.
    X["geohash_pickup"] = out["geohash_pickup"].to_numpy()
    X["geohash_dropoff"] = out["geohash_dropoff"].to_numpy()
    return out
