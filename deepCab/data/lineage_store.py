"""SQLite-backed lineage store. Every training run writes one LineageEdge row
tying (input parquet hash, preprocessor config hash, split definition hash) to
an MLflow run_id and a timestamp. Answers questions like:

    - "Which runs trained on the same input bytes?"   (group by input_hash)
    - "Did the preprocessor drift since last week?"   (preprocessor_hash group)
    - "Are these two runs comparable?"                (full-tuple equality)

Pure SQLite — no extra service. The DB file lives under
REGISTRY_LOCAL_PATH/lineage.db so it co-locates with model artifacts."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from deepCab.data.lineage import LineageEdge
from deepCab.schemas.settings import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lineage_edges (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  run_id TEXT,
  input_hash TEXT NOT NULL,
  preprocessor_hash TEXT NOT NULL,
  split_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_id ON lineage_edges(run_id);
CREATE INDEX IF NOT EXISTS idx_input_hash ON lineage_edges(input_hash);
"""


def _db_path() -> Path:
    return get_settings().registry.local_path.expanduser() / "lineage.db"


def _connect() -> sqlite3.Connection:
    p = _db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.executescript(_SCHEMA)
    return conn


def write_edge(edge: LineageEdge) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO lineage_edges (ts, run_id, input_hash, preprocessor_hash, split_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                time.time(),
                edge.run_id,
                edge.input_hash,
                edge.preprocessor_hash,
                edge.split_hash,
            ),
        )
        return int(cur.lastrowid)


def query_by_run(run_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, ts, run_id, input_hash, preprocessor_hash, split_hash "
            "FROM lineage_edges WHERE run_id = ? ORDER BY ts DESC",
            (run_id,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "ts": r[1],
            "run_id": r[2],
            "input_hash": r[3],
            "preprocessor_hash": r[4],
            "split_hash": r[5],
        }
        for r in rows
    ]


def runs_sharing_input(input_hash: str) -> list[str]:
    """All run_ids that trained on the same input parquet hash. Used by the
    agent's `compare_runs` tool to spot apples-to-apples comparisons."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT run_id FROM lineage_edges "
            "WHERE input_hash = ? AND run_id IS NOT NULL "
            "ORDER BY ts DESC",
            (input_hash,),
        ).fetchall()
    return [r[0] for r in rows]
