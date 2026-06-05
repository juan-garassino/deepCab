"""SHAP aggregation: column groups cover the full 65-d, sums are exact,
shape validation refuses anything but 65 columns."""

from __future__ import annotations

import numpy as np
import pytest

from deepCab.explain.aggregate import (
    COLUMN_GROUPS,
    EXPECTED_DIM,
    FEATURE_ORDER,
    aggregate_global,
    aggregate_shap,
)


def test_groups_cover_full_dim_no_overlap() -> None:
    seen = set()
    for start, end in COLUMN_GROUPS.values():
        for c in range(start, end):
            assert c not in seen, f"column {c} appears in multiple groups"
            seen.add(c)
    assert seen == set(range(EXPECTED_DIM))


def test_aggregate_row_sums_exactly() -> None:
    values = np.arange(EXPECTED_DIM, dtype=float)
    out = aggregate_shap(values)
    for name, (start, end) in COLUMN_GROUPS.items():
        assert out[name] == pytest.approx(float(values[start:end].sum()))


def test_aggregate_preserves_feature_order() -> None:
    values = np.zeros(EXPECTED_DIM)
    out = aggregate_shap(values)
    assert list(out) == FEATURE_ORDER


def test_aggregate_global_uses_abs_then_sum_then_mean() -> None:
    n = 4
    arr = np.zeros((n, EXPECTED_DIM))
    arr[:, 0] = [-1.0, 2.0, -3.0, 4.0]  # passenger group (col 0)
    out = aggregate_global(arr)
    # mean(|x|) = (1 + 2 + 3 + 4) / 4 = 2.5
    assert out["passenger"] == pytest.approx(2.5)


def test_aggregate_rejects_wrong_dim() -> None:
    with pytest.raises(ValueError, match="Expected 65-d"):
        aggregate_shap(np.zeros(60))
