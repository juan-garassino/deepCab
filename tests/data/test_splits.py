"""TimeSeriesSplit must refuse unsorted input and never let test indices precede
train indices."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from deepCab.data.splits import kfold_splits, time_series_splits


def _toy_frame(n: int = 100, shuffled: bool = False) -> pl.DataFrame:
    base = datetime(2014, 1, 1)
    times = [base + timedelta(days=i) for i in range(n)]
    if shuffled:
        times = list(reversed(times))
    return pl.DataFrame({"pickup_datetime": times, "y": list(range(n))})


def test_time_series_splits_no_temporal_leakage() -> None:
    df = _toy_frame(100)
    for train_idx, test_idx in time_series_splits(df, n_splits=5):
        assert train_idx.max() < test_idx.min()


def test_time_series_splits_refuses_unsorted() -> None:
    df = _toy_frame(50, shuffled=True)
    with pytest.raises(ValueError, match="not sorted"):
        list(time_series_splits(df, n_splits=3))


def test_kfold_splits_disjoint() -> None:
    for train, test in kfold_splits(60, n_splits=4, seed=1):
        assert set(train).isdisjoint(test)
        assert len(train) + len(test) == 60
