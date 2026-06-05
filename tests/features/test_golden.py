"""Golden-vector regression: the Polars port in features/transformers.py must
match the legacy numpy reference in features/golden.py element-wise."""

from __future__ import annotations

import numpy as np
import pytest

from deepCab.features.golden import (
    as_dataframe,
    reference_geohash,
    reference_lonlat_features,
    reference_time_features,
)
from deepCab.features.transformers import (
    compute_geohash,
    transform_lonlat_features,
    transform_time_features,
)


def test_time_features_matches_reference() -> None:
    X = as_dataframe()
    out = transform_time_features(X)
    ref = reference_time_features(X)
    assert out.shape == ref.shape
    # Float columns (hour_sin, hour_cos) up to fp tolerance; integer columns exact
    np.testing.assert_allclose(out[:, :2], ref[:, :2], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(out[:, 2:].astype(int), ref[:, 2:].astype(int))


def test_lonlat_features_matches_reference() -> None:
    X = as_dataframe()
    out = transform_lonlat_features(X)
    ref = reference_lonlat_features(X)
    np.testing.assert_allclose(
        out[["haversine_km", "manhattan_km"]].to_numpy(),
        ref[["haversine_km", "manhattan_km"]].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_lonlat_zero_distance_row() -> None:
    """Row 1 has identical pickup/dropoff coords — both distances must be 0."""
    X = as_dataframe()
    out = transform_lonlat_features(X)
    assert out.loc[1, "haversine_km"] == pytest.approx(0.0, abs=1e-12)
    assert out.loc[1, "manhattan_km"] == pytest.approx(0.0, abs=1e-12)


def test_geohash_matches_reference() -> None:
    X = as_dataframe().copy()
    out = compute_geohash(X)
    ref = reference_geohash(as_dataframe())
    assert out["geohash_pickup"].tolist() == ref["geohash_pickup"].tolist()
    assert out["geohash_dropoff"].tolist() == ref["geohash_dropoff"].tolist()


def test_time_features_dst_boundary() -> None:
    """Row 0 (July, EDT = UTC-4): 17:18 UTC -> 13:18 NYC -> hour 13.
    Row 1 (Jan, EST = UTC-5): 05:00 UTC -> 00:00 NYC -> hour 0.
    Verifies tz conversion crosses the DST boundary correctly."""
    out = transform_time_features(as_dataframe())
    hour_july = np.arctan2(out[0, 0], out[0, 1]) * 24 / (2 * np.pi)
    hour_jan = np.arctan2(out[1, 0], out[1, 1]) * 24 / (2 * np.pi)
    # angles wrap; just check hour values via direct sin/cos round-trip
    assert np.isclose(out[0, 0], np.sin(2 * np.pi * 13 / 24), atol=1e-12)
    assert np.isclose(out[1, 0], np.sin(2 * np.pi * 0 / 24), atol=1e-12)
    assert np.isclose(out[1, 1], np.cos(2 * np.pi * 0 / 24), atol=1e-12)
    # quiet unused-var
    _ = hour_july, hour_jan
