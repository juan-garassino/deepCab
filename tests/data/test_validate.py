"""Pandera-Polars schemas reject out-of-range rows."""
from __future__ import annotations

import polars as pl
import pytest
from pandera.errors import SchemaError

from deepCab.data.validate import CleanSchema, RawSchema


def _valid_raw_row() -> dict:
    return {
        "key": "row-1",
        "fare_amount": 12.5,
        "pickup_datetime": "2014-01-15 05:00:00 UTC",
        "pickup_longitude": -73.97,
        "pickup_latitude": 40.78,
        "dropoff_longitude": -73.99,
        "dropoff_latitude": 40.74,
        "passenger_count": 2,
    }


def test_raw_schema_accepts_valid() -> None:
    df = pl.DataFrame([_valid_raw_row()])
    RawSchema.validate(df)


def test_clean_schema_rejects_passenger_zero() -> None:
    row = _valid_raw_row()
    row.pop("key")
    row["passenger_count"] = 0
    df = pl.DataFrame([row]).with_columns(
        pl.col("pickup_datetime").str.replace(r" UTC$", "").str.to_datetime()
    )
    with pytest.raises(SchemaError):
        CleanSchema.validate(df)


def test_clean_schema_rejects_outside_nyc() -> None:
    row = _valid_raw_row()
    row.pop("key")
    row["pickup_latitude"] = 1.0  # equator
    df = pl.DataFrame([row]).with_columns(
        pl.col("pickup_datetime").str.replace(r" UTC$", "").str.to_datetime()
    )
    with pytest.raises(SchemaError):
        CleanSchema.validate(df)
