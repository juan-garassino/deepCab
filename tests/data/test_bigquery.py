"""BigQuery ingest path. Verifies the SQL the helper emits + that the
preprocess loader routes through it when DataSource.QUERY is set."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pyarrow as pa
import pytest

from deepCab.data.bigquery import chunk_where_clause, scan_bigquery


def _fake_arrow(n: int = 3) -> pa.Table:
    base = datetime(2014, 1, 1)
    return pa.table(
        {
            "pickup_datetime": [base for _ in range(n)],
            "pickup_longitude": [-73.99] * n,
            "pickup_latitude": [40.75] * n,
            "dropoff_longitude": [-73.96] * n,
            "dropoff_latitude": [40.78] * n,
            "passenger_count": [1] * n,
            "fare_amount": [9.5] * n,
        }
    )


def _client_returning(arrow_table: pa.Table) -> MagicMock:
    client = MagicMock()
    client.query.return_value.to_arrow.return_value = arrow_table
    return client


def test_scan_bigquery_returns_polars_with_expected_columns() -> None:
    client = _client_returning(_fake_arrow())
    df = scan_bigquery("proj", "ds", "tbl", client=client)
    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == {
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "fare_amount",
    }


def test_scan_bigquery_emits_limit_and_where() -> None:
    client = _client_returning(_fake_arrow())
    scan_bigquery("p", "d", "t", limit=42, where="pickup_datetime > '2014-01-01'", client=client)
    sent_sql = client.query.call_args.args[0]
    assert "`p.d.t`" in sent_sql
    assert "WHERE pickup_datetime > '2014-01-01'" in sent_sql
    assert "ORDER BY pickup_datetime" in sent_sql
    assert sent_sql.rstrip().endswith("LIMIT 42")


def test_scan_bigquery_no_where_no_limit() -> None:
    client = _client_returning(_fake_arrow())
    scan_bigquery("p", "d", "t", client=client)
    sent_sql = client.query.call_args.args[0]
    assert "WHERE" not in sent_sql
    assert "LIMIT" not in sent_sql


def test_scan_bigquery_reads_public_data_project_in_from() -> None:
    """The data project lands in the FROM clause (public dataset path)."""
    client = _client_returning(_fake_arrow())
    scan_bigquery("bigquery-public-data", "new_york", "tlc_yellow_trips_2014", client=client)
    sent_sql = client.query.call_args.args[0]
    assert "`bigquery-public-data.new_york.tlc_yellow_trips_2014`" in sent_sql


def test_scan_bigquery_bills_to_separate_project() -> None:
    """When no client is injected, the job bills to billing_project (our own),
    not the data project — required to query bigquery-public-data."""
    pytest.importorskip("google.cloud.bigquery")  # only present in CI/prod env
    with patch("google.cloud.bigquery.Client") as ClientCls:
        ClientCls.return_value.query.return_value.to_arrow.return_value = _fake_arrow()
        scan_bigquery(
            "bigquery-public-data",
            "new_york",
            "tlc_yellow_trips_2014",
            billing_project="garassino-ml",
        )
    assert ClientCls.call_args.kwargs["project"] == "garassino-ml"


def test_scan_bigquery_billing_defaults_to_data_project() -> None:
    pytest.importorskip("google.cloud.bigquery")
    with patch("google.cloud.bigquery.Client") as ClientCls:
        ClientCls.return_value.query.return_value.to_arrow.return_value = _fake_arrow()
        scan_bigquery("garassino-ml", "taxi", "yellow_trips_raw")
    assert ClientCls.call_args.kwargs["project"] == "garassino-ml"


def test_chunk_where_clause_is_half_open() -> None:
    clause = chunk_where_clause("2014-01-01 00:00:00", "2014-01-08 00:00:00")
    assert ">=" in clause and "<" in clause and "<=" not in clause


@pytest.mark.parametrize("source", ["local", "query"])
def test_preprocess_load_routes_on_data_source(source: str) -> None:
    """When DataSource.QUERY, preprocess.load() must call scan_bigquery, not
    the Hive-Parquet scan. Mock both paths and verify exactly one fired."""
    from deepCab.schemas.config import DataRef
    from deepCab.schemas.enums import DataSize
    from deepCab.training import preprocess

    fake_df = pl.from_arrow(_fake_arrow())

    with (
        patch.object(preprocess, "get_settings") as gs,
        patch("deepCab.data.bigquery.scan_bigquery", return_value=fake_df) as bq_scan,
        patch.object(preprocess, "scan") as parquet_scan,
    ):
        gs.return_value.data.source = source
        gs.return_value.data.bq_project = "p"
        gs.return_value.data.bq_dataset = "d"
        gs.return_value.data.bq_table = "t"
        gs.return_value.data.bq_where = None
        parquet_scan.return_value.collect.return_value = fake_df
        parquet_scan.return_value.columns = list(fake_df.columns)

        preprocess.load(DataRef(size=DataSize.S1K, validation_size=DataSize.S1K), split="train")

        if source == "query":
            assert bq_scan.called
            assert not parquet_scan.called
        else:
            assert parquet_scan.called
            assert not bq_scan.called


def test_preprocess_load_forwards_bq_where_to_scan() -> None:
    """The chunk WHERE clause that simulate sets via DATA_BQ_WHERE must reach
    the BQ read — without it every chunk would pull the full table slice."""
    from deepCab.schemas.config import DataRef
    from deepCab.schemas.enums import DataSize
    from deepCab.training import preprocess

    fake_df = pl.from_arrow(_fake_arrow())
    chunk_where = "pickup_datetime >= TIMESTAMP('2014-01-01 00:00:00') AND pickup_datetime < TIMESTAMP('2014-01-08 00:00:00')"

    with (
        patch.object(preprocess, "get_settings") as gs,
        patch("deepCab.data.bigquery.scan_bigquery", return_value=fake_df) as bq_scan,
    ):
        gs.return_value.data.source = "query"
        gs.return_value.data.bq_project = "p"
        gs.return_value.data.bq_dataset = "d"
        gs.return_value.data.bq_table = "t"
        gs.return_value.data.bq_where = chunk_where

        preprocess.load(DataRef(size=DataSize.S1K, validation_size=DataSize.S1K), split="train")

    assert bq_scan.call_args.kwargs["where"] == chunk_where
