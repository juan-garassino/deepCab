"""Pure preprocessing functions. No env reads, no MLflow side-effects, no
global state. Replaces the chunked-CSV loop in legacy interface/main.py.

Data shape:
    parquet (Hive: dataset_size/year/month) -> pl.DataFrame
        -> clean_data filters (NYC bounds + fare + passenger)
        -> features.pipeline.preprocess_features -> (N, 65) np.ndarray
"""
from __future__ import annotations

import numpy as np
import polars as pl

from deepCab.data.io import scan
from deepCab.features.pipeline import preprocess_features
from deepCab.schemas.config import DataRef

NYC_LAT = (40.5, 40.9)
NYC_LON = (-74.3, -73.7)


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the legacy clean_data filters. `key` (if present) is dropped."""
    out = df
    if "key" in out.columns:
        out = out.drop("key")
    return (
        out.unique()
        .drop_nulls()
        .filter(
            (pl.col("fare_amount") > 0)
            & (pl.col("fare_amount") <= 400)
            & (pl.col("passenger_count") >= 1)
            & (pl.col("passenger_count") <= 8)
            & pl.col("pickup_latitude").is_between(*NYC_LAT)
            & pl.col("pickup_longitude").is_between(*NYC_LON)
            & pl.col("dropoff_latitude").is_between(*NYC_LAT)
            & pl.col("dropoff_longitude").is_between(*NYC_LON)
        )
    )


def load(data: DataRef, split: str = "train") -> pl.DataFrame:
    """Read the relevant partition from the Parquet store and apply `clean()`."""
    lf = scan(size=data.size if split == "train" else data.validation_size)
    if "split" in lf.columns:
        lf = lf.filter(pl.col("split") == split)
    return clean(lf.collect())


def featurize(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Apply features.pipeline.preprocess_features. Returns (X, y) numpy."""
    pdf = df.to_pandas()
    y = pdf["fare_amount"].to_numpy(dtype=np.float32)
    X = preprocess_features(pdf.drop(columns=["fare_amount"]))
    return X.astype(np.float32), y


def preprocess(data: DataRef, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Convenience: load+clean+featurize one partition."""
    return featurize(load(data, split=split))
