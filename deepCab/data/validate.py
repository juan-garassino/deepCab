"""Pandera-Polars schemas. Validated at every IO boundary.

Pandera's Polars backend is feature-light compared to pandas (no Hypothesis
checks, etc.), but covers the dtype + range + nullability checks we need. If
we hit a gap, fall back to Patito — same DSL, native Polars."""
from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series

# NYC bounding box (matches legacy clean_data filters + schemas/data.py:NYC_*)
NYC_LAT = (40.5, 40.9)
NYC_LON = (-74.3, -73.7)


class RawSchema(pa.DataFrameModel):
    """The shape of CSV/BigQuery output before clean_data. Includes the
    vestigial `key` column. The clean step drops `key` and enforces tighter
    ranges (see CleanSchema)."""

    key: Series[str]
    fare_amount: Series[float] = pa.Field(ge=0)
    pickup_datetime: Series[str]
    pickup_longitude: Series[float]
    pickup_latitude: Series[float]
    dropoff_longitude: Series[float]
    dropoff_latitude: Series[float]
    passenger_count: Series[int] = pa.Field(ge=0)


class CleanSchema(pa.DataFrameModel):
    """Post-clean: NYC bounds enforced, fare bounded, passenger ∈ [1, 8]."""

    fare_amount: Series[float] = pa.Field(gt=0, le=400)
    pickup_datetime: Series[pl.Datetime]
    pickup_longitude: Series[float] = pa.Field(ge=NYC_LON[0], le=NYC_LON[1])
    pickup_latitude: Series[float] = pa.Field(ge=NYC_LAT[0], le=NYC_LAT[1])
    dropoff_longitude: Series[float] = pa.Field(ge=NYC_LON[0], le=NYC_LON[1])
    dropoff_latitude: Series[float] = pa.Field(ge=NYC_LAT[0], le=NYC_LAT[1])
    passenger_count: Series[int] = pa.Field(ge=1, le=8)

    class Config:
        strict = True
        coerce = True
