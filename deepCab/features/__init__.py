"""Stateless feature transforms (Polars-backed) + pipeline assembly. Mirrors
the legacy ml_logic/encoders + preprocessor module surface so phase 0-2
swap-in is non-breaking."""
from deepCab.features.pipeline import preprocess_features  # noqa: F401
from deepCab.features.transformers import (  # noqa: F401
    compute_geohash,
    transform_lonlat_features,
    transform_time_features,
)
