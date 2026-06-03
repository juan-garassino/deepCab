"""Plateau detector: monotonic-trend check (P11 fix). The previous spread-only
check (`max - min < eps`) was fooled by oscillation."""
from __future__ import annotations


def _detect_plateau(window: list[float], eps: float) -> tuple[bool, float]:
    """Reimplementation of the detector logic for unit-testing without spinning
    up the full improve loop. Mirrors agent/improve.py:144-176."""
    half = len(window) // 2 or 1
    old_mean = sum(window[:half]) / half
    new_mean = sum(window[half:]) / max(len(window) - half, 1)
    improvement = old_mean - new_mean
    return improvement < eps, improvement


def test_plateau_fires_when_metric_flat() -> None:
    flat = [3.0, 3.0, 3.0, 3.0]
    fires, improvement = _detect_plateau(flat, eps=1e-3)
    assert fires
    assert improvement == 0.0


def test_plateau_fires_on_oscillation() -> None:
    """The pre-P11 spread check let oscillation pass through (spread=5 > eps).
    The trend check catches it because both halves have mean=7.5."""
    oscillating = [5.0, 10.0, 5.0, 10.0]
    fires, improvement = _detect_plateau(oscillating, eps=1e-3)
    assert fires, "monotonic-trend detector should treat oscillation as plateau"
    assert abs(improvement) < 1e-9


def test_plateau_does_not_fire_when_improving() -> None:
    improving = [10.0, 9.0, 8.0, 7.0]   # MAE going down -> improvement > 0
    fires, improvement = _detect_plateau(improving, eps=0.1)
    assert not fires
    assert improvement > 0


def test_plateau_fires_when_metric_worsening() -> None:
    """If the metric is getting WORSE, that's also a plateau-like signal —
    no point continuing. improvement < 0 < eps -> fires."""
    worsening = [5.0, 6.0, 7.0, 8.0]
    fires, improvement = _detect_plateau(worsening, eps=0.1)
    assert fires
    assert improvement < 0
