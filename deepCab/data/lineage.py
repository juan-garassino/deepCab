"""Content-hash lineage. Every training run emits one LineageEdge that ties
(input parquet bytes, preprocessor config, split definition) -> MLflow run_id.
Used by the agent's `compare_runs` tool and the Phase 10 lineage SQLite store."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LineageEdge:
    input_hash: str
    preprocessor_hash: str
    split_hash: str
    run_id: str | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "input_hash": self.input_hash,
            "preprocessor_hash": self.preprocessor_hash,
            "split_hash": self.split_hash,
            "run_id": self.run_id,
        }


def hash_path(path: Path, chunk: int = 1 << 20) -> str:
    """blake2b over file content. For directories, hash the sorted list of
    (relpath, content_hash) — order-stable across runs."""
    if path.is_file():
        h = hashlib.blake2b(digest_size=16)
        with path.open("rb") as f:
            while data := f.read(chunk):
                h.update(data)
        return h.hexdigest()
    if path.is_dir():
        rolls: list[str] = []
        for sub in sorted(path.rglob("*")):
            if sub.is_file():
                rolls.append(f"{sub.relative_to(path)}:{hash_path(sub)}")
        return hashlib.blake2b("\n".join(rolls).encode(), digest_size=16).hexdigest()
    raise FileNotFoundError(path)


def hash_obj(obj: Any) -> str:
    """Canonical JSON dump -> blake2b. Stable for dicts, lists, Pydantic dumps."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()
