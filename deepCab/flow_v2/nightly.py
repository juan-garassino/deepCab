"""Every-2-days sliding-window bench over the *real* public NYC-taxi data.

Resumes its window cursor from MLflow (Cloud SQL backend) so each scheduled
run advances one chunk forward in time — no local Parquet, no synthesised
rows. One tree-backend model is trained per run on a fresh time-slice queried
straight from ``bigquery-public-data.new_york`` (the deprecated dataset that
still carries pickup/dropoff lon-lat; the newer ``new_york_taxi_trips`` set
replaced them with zone IDs the feature pipeline can't use).

The walk is pinned to the 2014 table: a single ``DATA_BQ_TABLE`` env var feeds
both the training window and the held-out reference slice, so they must live in
the same per-year table. The window slides Jan→Dec and wraps back to Jan.

Billing vs data project: the public table lives in ``bigquery-public-data`` but
jobs bill to our own project — ``scan_bigquery(billing_project=...)`` carries
that split, fed from ``DATA_BQ_BILLING_PROJECT`` (set by the workflow).

Fails soft: when MLflow is unreachable (tracking URI unset, or Cloud SQL down
in show-and-destroy mode) the run logs a warning and exits 0 rather than
turning the schedule red — the old daily failure mode.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from deepCab.data.bigquery import chunk_where_clause
from deepCab.flow_v2.simulate import _env_overlay, _simulate_impl
from deepCab.obs.log import get_logger
from deepCab.schemas.config import DataRef, LGBMConfig, TrainConfig
from deepCab.schemas.enums import DataSource
from deepCab.schemas.settings import get_settings

log = get_logger(__name__)

# Public source carrying lon/lat. Pinned to one year so the reference slice and
# training windows share the single DATA_BQ_TABLE env var.
PUBLIC_DATA_PROJECT = "bigquery-public-data"
PUBLIC_DATASET = "new_york"
PUBLIC_TABLE = "tlc_yellow_trips_2014"

_FLOOR = datetime(2014, 1, 1)
_HORIZON = datetime(2015, 1, 1)

# Fixed held-out slice (last week of the year) both challenger and champion are
# scored on, so MAE is comparable across runs and promotion isn't chasing
# window noise. Sits in the same table as the training windows.
_REFERENCE_WHERE = chunk_where_clause("2014-12-24 00:00:00", "2015-01-01 00:00:00")

_WINDOW_START_TAG = "deepcab.window_start"
_WINDOW_END_TAG = "deepcab.window_end"


def next_window(
    last_end: datetime | None,
    period: timedelta,
    *,
    floor: datetime = _FLOOR,
    horizon: datetime = _HORIZON,
) -> tuple[datetime, datetime]:
    """Half-open ``[start, end)`` for the next chunk. Starts at ``floor`` on the
    first run; wraps back to ``floor`` once the walk reaches ``horizon``."""
    start = last_end or floor
    if start >= horizon:
        start = floor
    end = min(start + period, horizon)
    return start, end


def _read_last_window_end(client, experiment_name: str | None) -> datetime | None:
    """Max ``window_end`` tag across runs in the experiment, or None on first
    run / missing experiment. ``client`` is an MlflowClient (injected in tests)."""
    if not experiment_name:
        return None
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=100)
    ends: list[datetime] = []
    for r in runs:
        raw = r.data.tags.get(_WINDOW_END_TAG)
        if not raw:
            continue
        try:
            ends.append(datetime.fromisoformat(raw))
        except ValueError:
            continue
    return max(ends) if ends else None


def _mlflow_client():
    """MlflowClient bound to the configured tracking URI, or None when unset."""
    from deepCab.obs.mlflow import get_mlflow_client

    return get_mlflow_client()


def run_nightly(period_days: int = 14, *, client=None) -> int:
    """Train one sliding-window chunk on real public BQ data, persist the
    cursor as MLflow tags, advance. Returns a process exit code — 0 even when
    it deliberately no-ops, so the schedule stays green."""
    m = get_settings().mlflow
    client = client if client is not None else _mlflow_client()
    if client is None:
        log.warning(
            "nightly.skip_no_mlflow",
            reason="MLFLOW_TRACKING_URI unset — can't resume cursor or persist run",
        )
        return 0

    last_end = _read_last_window_end(client, m.experiment)
    start, end = next_window(last_end, timedelta(days=period_days))
    log.info("nightly.window", start=start.isoformat(), end=end.isoformat(), table=PUBLIC_TABLE)

    overlay = {
        "DATA_SOURCE": DataSource.QUERY.value,
        "DATA_BQ_PROJECT": PUBLIC_DATA_PROJECT,
        "DATA_BQ_DATASET": PUBLIC_DATASET,
        "DATA_BQ_TABLE": PUBLIC_TABLE,
        # Outer default WHERE = held-out reference slice (evaluate + promote read
        # it); train_chunk_task overrides it to the chunk window then restores.
        "DATA_BQ_WHERE": _REFERENCE_WHERE,
    }
    with _env_overlay(overlay, clear_settings_cache=True):
        cfg = TrainConfig(backend=LGBMConfig(), data=DataRef(size="10k"))
        result = _simulate_impl(
            cfg,
            reference_data=DataRef(size="10k"),
            time_window_start=start,
            time_window_end=end,
            chunk_period=end - start,  # exactly one chunk per run
        )

    run_id = result.chunks[-1].train_run_id if result.chunks else None
    if run_id:
        client.set_tag(run_id, _WINDOW_START_TAG, start.isoformat())
        client.set_tag(run_id, _WINDOW_END_TAG, end.isoformat())
        log.info("nightly.tagged", run_id=run_id, window_end=end.isoformat())
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="deepCab nightly sliding-window bench")
    parser.add_argument("--period-days", type=int, default=14, help="chunk width in days")
    args = parser.parse_args()
    raise SystemExit(run_nightly(period_days=args.period_days))


if __name__ == "__main__":
    main()
