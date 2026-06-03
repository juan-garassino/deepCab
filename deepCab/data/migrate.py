"""One-shot CLI: legacy CSV (~/.lewagon/mlops/data/raw/{train,val}_{size}.csv)
-> Hive-partitioned Parquet (~/.lewagon/mlops/data/parquet/dataset_size=.../
year=.../month=.../). The path under settings.data.parquet_path is the canonical
root used by deepCab.data.io.

Usage:
    python -m deepCab.data.migrate --size 1k
    python -m deepCab.data.migrate --size 10k --split train
    python -m deepCab.data.migrate --size 1k --src ~/somewhere/else/data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from deepCab.data.io import write_partitioned
from deepCab.obs.log import get_logger
from deepCab.schemas.settings import get_settings

log = get_logger(__name__)


def migrate(size: str, split: str, src_root: Path | None = None) -> Path:
    settings = get_settings()
    src_root = (src_root or settings.data.local_path).expanduser() / "raw"
    csv = src_root / f"{split}_{size}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Expected legacy CSV at {csv}")

    log.info("migrate.read", csv=str(csv))
    df = pl.read_csv(csv, try_parse_dates=False)

    # Legacy datetime format: "YYYY-MM-DD HH:MM:SS UTC" — strip suffix, parse, tag UTC.
    df = df.with_columns(
        pl.col("pickup_datetime")
        .str.replace(r" UTC$", "")
        .str.to_datetime(time_unit="us")
        .dt.replace_time_zone("UTC")
    ).with_columns(pl.lit(split).alias("split"))

    out = write_partitioned(df, dataset_size=size)
    log.info("migrate.done", rows=df.height, partitioned_at=str(out))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", required=True, choices=["1k", "10k", "100k", "500k"])
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--src", type=Path, default=None, help="Override DATA_LOCAL_PATH")
    args = p.parse_args()
    migrate(size=args.size, split=args.split, src_root=args.src)


if __name__ == "__main__":
    main()
