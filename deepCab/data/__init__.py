"""Data layer: Polars + DuckDB I/O over Hive-partitioned Parquet, Pandera
schemas, leakage-safe splits, content-hash lineage."""
from deepCab.data.io import read, scan, write_partitioned  # noqa: F401
from deepCab.data.lineage import LineageEdge, hash_obj, hash_path  # noqa: F401
from deepCab.data.lineage_store import query_by_run, runs_sharing_input, write_edge  # noqa: F401
from deepCab.data.splits import kfold_splits, time_series_splits  # noqa: F401
from deepCab.data.validate import CleanSchema, RawSchema  # noqa: F401
