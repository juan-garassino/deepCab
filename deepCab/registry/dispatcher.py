"""Backend-agnostic model save/load.

`save_full_state(handle, run_id)` writes the fitted estimator + background +
ACI calibration to a stable on-disk directory and updates a LATEST pointer.
`load_state_from_disk(run_id)` is the inverse — it's what the FastAPI
lifespan reads on startup to rehydrate `STATE.model` without a fresh train.

The on-disk layout:
    <REGISTRY_LOCAL_PATH>/runs/<run_id>/
        model/             — backend-native (TF SavedModel dir, .pt, .json, ...)
        model.cfg.json     — torch/ft-t sidecar (existing P11 contract)
        cfg.json           — backend cfg dict + backend_kind
        background.npy     — np.array, ~200 rows × 65 cols (for SHAP)
        aci.json           — residuals + alpha/alpha_t/gamma (absent when no ACI)
    <REGISTRY_LOCAL_PATH>/runs/LATEST  — text file with the latest run_id
"""
from __future__ import annotations

import base64
import json
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from deepCab.models.base import AbstractEstimator
from deepCab.models.conformal import ACIRegressor
from deepCab.models.factory import build_estimator
from deepCab.obs.log import get_logger
from deepCab.schemas.config import BackendConfig
from deepCab.schemas.settings import get_settings

if TYPE_CHECKING:
    from deepCab.api.state import ModelHandle

log = get_logger(__name__)


# ---- runs root ----------------------------------------------------------


def runs_root() -> Path:
    root = get_settings().registry.local_path.expanduser() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def latest_pointer() -> Path:
    return runs_root() / "LATEST"


# ---- legacy compat ------------------------------------------------------


def save_artifact(estimator: AbstractEstimator) -> Path:
    """Legacy: save estimator weights to a temp dir. Kept so training/train.py
    can still hand the path to MLflow.log_artifacts. New callers should use
    save_full_state for full ModelHandle persistence."""
    tmp = Path(tempfile.mkdtemp(prefix="deepcab-model-"))
    out = tmp / "model"
    estimator.save(out)
    log.info("registry.saved", path=str(out), backend=estimator.cfg.kind)
    return out


def load_artifact(backend_cfg: BackendConfig, path: Path) -> AbstractEstimator:
    cls = type(build_estimator(backend_cfg))
    est = cls.load(path)
    log.info("registry.loaded", path=str(path), backend=backend_cfg.kind)
    return est


# ---- persistent state (FR-1) -------------------------------------------


def save_full_state(handle: "ModelHandle", run_id: str | None = None) -> Path:
    """Persist a fitted ModelHandle to <runs_root>/<run_id>/. Updates the
    LATEST pointer atomically (write-then-rename). Returns the run directory."""
    rid = run_id or f"local-{uuid.uuid4().hex[:8]}"
    run_dir = runs_root() / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Backend weights (uses each AbstractEstimator's own save)
    handle.estimator.save(run_dir / "model")

    # 2. Manifest — backend_kind + cfg. Separate from the torch sidecar so
    #    every backend gets a uniform handle.
    (run_dir / "cfg.json").write_text(
        json.dumps(
            {
                "backend_kind": handle.backend_kind,
                "cfg": handle.estimator.cfg.model_dump(),
            }
        )
    )

    # 3. Background sample for SHAP
    if handle.background is not None:
        np.save(run_dir / "background.npy", handle.background)

    # 4. ACI calibration
    if handle.aci is not None:
        aci: ACIRegressor = handle.aci
        residuals = aci._residuals if aci._residuals is not None else np.array([])
        (run_dir / "aci.json").write_text(
            json.dumps(
                {
                    "residuals_b64": base64.b64encode(
                        residuals.astype("float32").tobytes()
                    ).decode(),
                    "residuals_shape": list(residuals.shape),
                    "alpha": aci.alpha,
                    "alpha_t": aci._alpha_t,
                    "gamma": aci.gamma,
                }
            )
        )

    # 5. Atomic LATEST pointer — write a sibling then rename
    pointer = latest_pointer()
    tmp_pointer = pointer.with_suffix(".tmp")
    tmp_pointer.write_text(rid)
    tmp_pointer.replace(pointer)

    log.info("registry.full_state_saved", run_id=rid, dir=str(run_dir))
    return run_dir


def load_state_from_disk(run_id: str) -> "ModelHandle":
    """Inverse of save_full_state. Rebuilds the full ModelHandle including
    estimator + background + ACI."""
    from deepCab.api.state import ModelHandle  # local import to avoid cycle
    from deepCab.models._kinds import BACKENDS

    run_dir = runs_root() / run_id
    cfg_path = run_dir / "cfg.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing cfg.json at {cfg_path}")

    manifest = json.loads(cfg_path.read_text())
    backend_kind = manifest["backend_kind"]
    if backend_kind not in BACKENDS:
        raise KeyError(f"unknown backend_kind={backend_kind!r} in {cfg_path}")

    est_cls = BACKENDS[backend_kind]
    est = est_cls.load(run_dir / "model")

    bg_path = run_dir / "background.npy"
    background = np.load(bg_path) if bg_path.exists() else None

    aci: ACIRegressor | None = None
    aci_path = run_dir / "aci.json"
    if aci_path.exists():
        blob = json.loads(aci_path.read_text())
        residuals = np.frombuffer(
            base64.b64decode(blob["residuals_b64"]), dtype="float32"
        ).reshape(blob["residuals_shape"])
        aci = ACIRegressor(base=est, alpha=blob["alpha"], gamma=blob["gamma"])
        aci._residuals = residuals
        aci._alpha_t = blob["alpha_t"]

    handle = ModelHandle(
        estimator=est, backend_kind=backend_kind, background=background, aci=aci
    )
    log.info(
        "registry.full_state_loaded",
        run_id=run_id,
        backend=backend_kind,
        has_aci=aci is not None,
    )
    return handle


def read_latest_run_id() -> str | None:
    p = latest_pointer()
    if not p.exists():
        return None
    rid = p.read_text().strip()
    return rid or None


# ---- MLflow alias (unchanged) ------------------------------------------


def set_alias(version: int, alias: str | None = None) -> None:
    """Set @<alias> on the latest registered model version. Defaults to
    settings.mlflow.challenger_alias."""
    import mlflow
    from mlflow.tracking import MlflowClient

    m = get_settings().mlflow
    alias = alias or m.challenger_alias
    if not m.tracking_uri or not m.model_name:
        log.warning("registry.alias.skipped", reason="mlflow tracking_uri/model_name unset")
        return
    mlflow.set_tracking_uri(m.tracking_uri)
    client = MlflowClient()
    client.set_registered_model_alias(name=m.model_name, alias=alias, version=version)
    log.info("registry.alias.set", alias=alias, version=version, model=m.model_name)
