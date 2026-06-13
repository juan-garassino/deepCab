"""Nightly sliding-window cursor logic. Pure-Python + stub MlflowClient — no
mlflow / BQ / Prefect server required (mirrors the simulate test seams)."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from deepCab.flow_v2 import nightly

P = timedelta(days=14)


def test_next_window_starts_at_floor_on_first_run() -> None:
    start, end = nightly.next_window(None, P)
    assert start == nightly._FLOOR
    assert end == nightly._FLOOR + P


def test_next_window_advances_from_last_end() -> None:
    last = datetime(2014, 3, 1)
    start, end = nightly.next_window(last, P)
    assert start == last
    assert end == last + P


def test_next_window_truncates_final_chunk_at_horizon() -> None:
    last = datetime(2014, 12, 25)  # +14d would overshoot 2015-01-01
    start, end = nightly.next_window(last, P)
    assert start == last
    assert end == nightly._HORIZON


def test_next_window_wraps_to_floor_past_horizon() -> None:
    start, end = nightly.next_window(nightly._HORIZON, P)
    assert start == nightly._FLOOR
    assert end == nightly._FLOOR + P


def _stub_client(window_ends: list[str | None]):
    """MlflowClient stub whose search_runs returns one run per tag value."""
    runs = [SimpleNamespace(data=SimpleNamespace(tags={nightly._WINDOW_END_TAG: w} if w else {}))
            for w in window_ends]
    return SimpleNamespace(
        get_experiment_by_name=lambda name: SimpleNamespace(experiment_id="1"),
        search_runs=lambda **kw: runs,
    )


def test_read_last_window_end_returns_max() -> None:
    client = _stub_client(["2014-01-15T00:00:00", "2014-03-12T00:00:00", "2014-02-01T00:00:00"])
    assert nightly._read_last_window_end(client, "deepcab-nightly") == datetime(2014, 3, 12)


def test_read_last_window_end_none_when_no_tags() -> None:
    client = _stub_client([None, None])
    assert nightly._read_last_window_end(client, "deepcab-nightly") is None


def test_read_last_window_end_skips_unparseable_tags() -> None:
    client = _stub_client(["garbage", "2014-05-01T00:00:00"])
    assert nightly._read_last_window_end(client, "deepcab-nightly") == datetime(2014, 5, 1)


def test_read_last_window_end_none_when_experiment_missing() -> None:
    client = SimpleNamespace(get_experiment_by_name=lambda name: None)
    assert nightly._read_last_window_end(client, "deepcab-nightly") is None


def test_run_nightly_soft_skips_when_no_mlflow(monkeypatch) -> None:
    """No tracking URI → no client → exit 0, never touches training/BQ."""
    monkeypatch.setattr(nightly, "_mlflow_client", lambda: None)
    called = False

    def _boom(*a, **k):  # would raise if the body ran
        nonlocal called
        called = True
        raise AssertionError("should not train when MLflow is unreachable")

    monkeypatch.setattr(nightly, "_simulate_impl", _boom)
    assert nightly.run_nightly() == 0
    assert called is False
