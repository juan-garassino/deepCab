"""BigQuery → Polars ingest for the continuous-training simulation loop.

Companion to ``data/io.py`` (which scans the Hive-Parquet store). When
``settings.data.source == DataSource.QUERY``, ``training.preprocess.load()``
routes here instead of ``io.scan()``.

The returned DataFrame matches the 7-column "raw" shape that
``preprocess.clean()`` and ``features.pipeline.preprocess_features``
consume — exactly the columns Lewagon's raw CSV ships with, minus the
vestigial ``key``. Extra columns from the BQ table (vendor_id,
dropoff_datetime, trip_distance, total_amount) are dropped here so the
downstream pipeline doesn't need to know which path the rows came from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from google.cloud.bigquery import Client

# Columns the rest of the pipeline expects. Order doesn't matter — `clean()`
# selects by name.
_RAW_COLUMNS = (
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "fare_amount",
)


def scan_bigquery(
    project_id: str,
    dataset: str,
    table: str,
    *,
    limit: int | None = None,
    where: str | None = None,
    order_by: str = "pickup_datetime",
    client: "Client | None" = None,
) -> pl.DataFrame:
    """Read rows from ``<project_id>.<dataset>.<table>`` into a Polars DataFrame.

    Arguments are composed into a single SELECT — no string interpolation on
    user input; ``where`` and ``order_by`` are caller-controlled trusted
    fragments (the only callers are ``preprocess.load()`` and the simulation
    flow's ``ingest_chunk_task``).

    Returns a DataFrame with exactly the columns in ``_RAW_COLUMNS``.
    """
    cols = ", ".join(_RAW_COLUMNS)
    sql = f"SELECT {cols} FROM `{project_id}.{dataset}.{table}`"
    if where:
        sql += f" WHERE {where}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    if client is None:
        from google.cloud import bigquery

        client = bigquery.Client(project=project_id)
    arrow_table = client.query(sql).to_arrow()
    return pl.from_arrow(arrow_table)


def chunk_where_clause(
    chunk_start: str,
    chunk_end: str,
    *,
    column: str = "pickup_datetime",
) -> str:
    """Build a half-open ``WHERE`` clause for one simulation chunk.

    ``chunk_start`` is inclusive, ``chunk_end`` is exclusive — matches how
    ``simulate_flow`` slices the time window so no row is double-counted
    between adjacent chunks. Inputs must already be ISO-8601 timestamps; the
    caller (``simulate_flow``) formats them from ``datetime`` objects.
    """
    return (
        f"{column} >= TIMESTAMP('{chunk_start}') "
        f"AND {column} < TIMESTAMP('{chunk_end}')"
    )
