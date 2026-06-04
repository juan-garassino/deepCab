"""Polars + DuckDB I/O. Hive-partitioned Parquet is the canonical on-disk format;
the chunked-CSV path is reserved for migration only (deepCab/data/migrate.py)."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from deepCab.schemas.settings import get_settings


def parquet_root() -> Path:
    return get_settings().data.parquet_path.expanduser()


def scan(size: str | None = None) -> pl.LazyFrame:
    """Lazy scan over the Parquet store. With hive_partitioning, year/month
    become columns. Pass size to read a specific dataset_size subtree."""
    root = parquet_root()
    glob = str(root / "**" / "*.parquet")
    lf = pl.scan_parquet(glob, hive_partitioning=True)
    if size is not None:
        lf = lf.filter(pl.col("dataset_size") == size) if "dataset_size" in lf.columns else lf
    return lf


def write_partitioned(df: pl.DataFrame, dataset_size: str) -> Path:
    """Write df as Hive-partitioned Parquet at parquet_root()/dataset_size=.../
    year=.../month=.../part-*.parquet. Requires a `pickup_datetime` column.

    Uses pyarrow.dataset.write_dataset directly — polars' own write_parquet
    path doesn't accept a partition_by + directory combination cleanly across
    pyarrow versions."""
    import pyarrow.dataset as pads

    root = parquet_root()
    root.mkdir(parents=True, exist_ok=True)
    df = df.with_columns(
        pl.col("pickup_datetime").dt.year().alias("year"),
        pl.col("pickup_datetime").dt.month().alias("month"),
        pl.lit(dataset_size).alias("dataset_size"),
    )
    pads.write_dataset(
        df.to_arrow(),
        base_dir=str(root),
        format="parquet",
        partitioning=["dataset_size", "year", "month"],
        partitioning_flavor="hive",
        existing_data_behavior="overwrite_or_ignore",
    )
    return root
