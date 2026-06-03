"""Lineage SQLite: edges persist, query_by_run round-trips, runs_sharing_input
groups correctly."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from deepCab.data.lineage import LineageEdge


@pytest.fixture
def tmp_lineage_db(tmp_path: Path, monkeypatch) -> Path:
    # Point REGISTRY_LOCAL_PATH at tmp so lineage.db lands there.
    monkeypatch.setenv("REGISTRY_LOCAL_PATH", str(tmp_path))
    # Settings is lru_cached; clear it.
    from deepCab.schemas.settings import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield tmp_path / "lineage.db"
    get_settings.cache_clear()  # type: ignore[attr-defined]
    os.environ.pop("REGISTRY_LOCAL_PATH", None)


def test_write_edge_and_query_by_run(tmp_lineage_db: Path) -> None:
    from deepCab.data.lineage_store import query_by_run, write_edge

    rid = write_edge(
        LineageEdge(input_hash="ih1", preprocessor_hash="ph1", split_hash="sh1", run_id="r1")
    )
    assert rid > 0
    rows = query_by_run("r1")
    assert len(rows) == 1
    assert rows[0]["input_hash"] == "ih1"
    assert rows[0]["preprocessor_hash"] == "ph1"
    assert rows[0]["split_hash"] == "sh1"


def test_runs_sharing_input_groups_correctly(tmp_lineage_db: Path) -> None:
    from deepCab.data.lineage_store import runs_sharing_input, write_edge

    write_edge(LineageEdge(input_hash="A", preprocessor_hash="p1", split_hash="s1", run_id="r1"))
    write_edge(LineageEdge(input_hash="A", preprocessor_hash="p2", split_hash="s2", run_id="r2"))
    write_edge(LineageEdge(input_hash="B", preprocessor_hash="p1", split_hash="s1", run_id="r3"))

    assert set(runs_sharing_input("A")) == {"r1", "r2"}
    assert set(runs_sharing_input("B")) == {"r3"}
    assert runs_sharing_input("nope") == []


def test_db_file_created_on_first_write(tmp_lineage_db: Path) -> None:
    from deepCab.data.lineage_store import write_edge

    assert not tmp_lineage_db.exists()
    write_edge(LineageEdge(input_hash="i", preprocessor_hash="p", split_hash="s", run_id="r"))
    assert tmp_lineage_db.exists()
