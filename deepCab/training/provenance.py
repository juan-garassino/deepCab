"""Per-run provenance manifest. One `provenance.json` per training run captures
enough context to reproduce: code SHA, OS, Python, key deps, CUDA, ONNX opset,
input data hash, config hash, seed, MLflow run id.

The Phase 10 lineage SQLite tags each MLflow run with the manifest hash so the
agent's `compare_runs` tool can answer "are these two runs comparable?"."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from deepCab.data.lineage import hash_obj
from deepCab.models.onnx_export import OPSET
from deepCab.obs.log import get_logger
from deepCab.schemas.config import TrainConfig

log = get_logger(__name__)


@dataclass
class Provenance:
    run_id: str | None
    git_sha: str | None
    config_hash: str
    seed: int
    backend_kind: str
    metrics: dict[str, float]
    python: str
    platform: str
    cuda_available: bool
    cuda_version: str | None
    onnx_opset: int
    deps: dict[str, str]


def emit_provenance(
    cfg: TrainConfig,
    run_id: str | None,
    metrics: dict[str, float],
    out_dir: Path | None = None,
) -> Path:
    out_dir = out_dir or Path("runs") / (run_id or "local")
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = Provenance(
        run_id=run_id,
        git_sha=_git_sha(),
        config_hash=hash_obj(cfg.model_dump(mode="json")),
        seed=cfg.seed,
        backend_kind=cfg.backend.kind,
        metrics=metrics,
        python=sys.version.split()[0],
        platform=platform.platform(),
        cuda_available=_cuda_available(),
        cuda_version=_cuda_version(),
        onnx_opset=OPSET,
        deps=_dep_versions(),
    )
    target = out_dir / "provenance.json"
    target.write_text(json.dumps(asdict(prov), indent=2, default=str))
    log.info("provenance.emitted", path=str(target), config_hash=prov.config_hash)
    return target


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_version() -> str | None:
    try:
        import torch  # type: ignore

        return getattr(torch.version, "cuda", None)
    except Exception:
        return None


def _dep_versions() -> dict[str, str]:
    """Pin the deps that affect numerical results. Failure to import is
    recorded as 'absent' (not an error)."""
    out: dict[str, str] = {}
    for pkg in (
        "numpy",
        "pandas",
        "polars",
        "scikit-learn",
        "tensorflow",
        "torch",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlflow",
        "onnx",
        "onnxruntime",
    ):
        try:
            mod = __import__(pkg.replace("-", "_"))
            out[pkg] = getattr(mod, "__version__", "?")
        except Exception:
            out[pkg] = "absent"
    return out
