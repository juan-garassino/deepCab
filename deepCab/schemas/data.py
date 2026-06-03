"""Row-level data contracts. Validated at every IO boundary."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


# NYC bounding box (matches legacy clean_data filters)
NYC_LAT = (40.5, 40.9)
NYC_LON = (-74.3, -73.7)


class FeatureRow(BaseModel):
    """The seven user-facing fields the model consumes (post-clean, pre-preprocess)."""

    pickup_datetime: datetime
    pickup_longitude: float = Field(ge=NYC_LON[0], le=NYC_LON[1])
    pickup_latitude: float = Field(ge=NYC_LAT[0], le=NYC_LAT[1])
    dropoff_longitude: float = Field(ge=NYC_LON[0], le=NYC_LON[1])
    dropoff_latitude: float = Field(ge=NYC_LAT[0], le=NYC_LAT[1])
    passenger_count: int = Field(ge=1, le=8)


class RawRow(FeatureRow):
    """Raw input as it lands from BigQuery / CSV — adds the target and the vestigial key."""

    key: str
    fare_amount: float = Field(gt=0, le=400)

    @field_validator("fare_amount")
    @classmethod
    def _positive_fare(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("fare_amount must be > 0")
        return v
