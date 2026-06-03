"""Train/val splits. TimeSeriesSplit on temporally-sorted indices — taxi fares
have a clear non-stationary structure (fares trend up, seasonal patterns, NYC
fleet composition shifts), so KFold leaks the future."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import polars as pl
from sklearn.model_selection import KFold, TimeSeriesSplit


def time_series_splits(
    df: pl.DataFrame, n_splits: int = 5, time_col: str = "pickup_datetime"
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yields (train_idx, test_idx) for a sliding window over the time-sorted rows.

    Guard: refuses to operate on unsorted data — silent reordering hides leakage."""
    times = df[time_col].to_numpy()
    if not _is_sorted(times):
        raise ValueError(
            f"DataFrame is not sorted ascending by '{time_col}'. "
            f"Call df.sort('{time_col}') before time_series_splits()."
        )
    tss = TimeSeriesSplit(n_splits=n_splits)
    yield from tss.split(np.arange(len(df)))


def kfold_splits(
    n: int, n_splits: int = 5, shuffle: bool = True, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Plain KFold — only use when there is *no* temporal leakage risk
    (e.g. synthetic data, IID benchmarks)."""
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    yield from kf.split(np.arange(n))


def _is_sorted(a: np.ndarray) -> bool:
    return bool(np.all(a[:-1] <= a[1:]))
