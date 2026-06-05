"""Lineage hashes must be stable and content-sensitive."""

from __future__ import annotations

from pathlib import Path

from deepCab.data.lineage import LineageEdge, hash_obj, hash_path


def test_hash_obj_order_invariant() -> None:
    a = {"x": 1, "y": [1, 2, 3], "nested": {"a": 1, "b": 2}}
    b = {"nested": {"b": 2, "a": 1}, "y": [1, 2, 3], "x": 1}
    assert hash_obj(a) == hash_obj(b)


def test_hash_obj_content_sensitive() -> None:
    assert hash_obj({"x": 1}) != hash_obj({"x": 2})


def test_hash_path_file(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text("hello\nworld\n")
    h1 = hash_path(p)
    p.write_text("hello\nWORLD\n")
    h2 = hash_path(p)
    assert h1 != h2


def test_hash_path_directory_is_content_hash(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    h1 = hash_path(tmp_path)
    (tmp_path / "b.txt").write_text("B")
    h2 = hash_path(tmp_path)
    assert h1 != h2


def test_manifest_shape() -> None:
    e = LineageEdge(input_hash="i", preprocessor_hash="p", split_hash="s", run_id="r1")
    m = e.manifest()
    assert set(m) == {"input_hash", "preprocessor_hash", "split_hash", "run_id"}
    assert m["run_id"] == "r1"
