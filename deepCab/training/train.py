"""Hydra entry + pure `run()` function for one training session.

Lifecycle:
    set_all(seed)
        -> preprocess(train) + preprocess(val)
        -> build_estimator(cfg.backend).fit(X, y, validation_data=(Xv, yv))
        -> ACI calibration on a holdout slice of val (no re-fit; uses
           ACIRegressor.from_fitted)
        -> mlflow.start_run() logs params + metrics + config-yaml + model artifact
        -> set_alias(@challenger)
        -> emit provenance.json
        -> publish ModelHandle into STATE so every entry point (Hydra CLI,
           Prefect flow, agent tool, lifespan autoloader, FastAPI predict
           router) sees the same fitted estimator

`run(cfg: TrainConfig) -> TrainResult` is the pure function the agent's `train`
tool calls. The Hydra `main()` is a thin OmegaConf->Pydantic bridge.

Sub-project F: when env `REGISTRY_GCS_BUCKET` is set, `run()` mirrors the
local run directory to `gs://<bucket>/runs/<run_id>/` after `_run_training`
returns. The push is a thin `gsutil` shell-out, so the unit tests stub the
helper out and never touch the cloud. The default cloud training trigger
(Cloud Scheduler -> Cloud Run Job) sets this env on the Job container."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from deepCab.models.conformal import ACIRegressor
from deepCab.models.factory import build_estimator
from deepCab.obs.log import get_logger
from deepCab.schemas.config import TrainConfig
from deepCab.schemas.settings import get_settings
from deepCab.training._mlflow import log_config_artifact, log_params_flat
from deepCab.training.preprocess import preprocess
from deepCab.training.provenance import emit_provenance
from deepCab.training.seed import set_all

log = get_logger(__name__)
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Fraction of the val split reserved for ACI residual calibration. The rest is
# used for the val_mae metric so reporting stays consistent with prior phases.
_ACI_CALIB_FRACTION = 0.3
_ACI_MIN_ROWS = 30


@dataclass
class TrainResult:
    run_id: str | None
    backend_kind: str
    val_mae: float
    model_path: str
    # Phase 11: surfaces of the fitted state so callers (agent tool, Prefect
    # flow, Hydra CLI) can populate STATE.model without re-fitting.
    estimator: Any = None
    background: np.ndarray | None = field(default=None, repr=False)
    aci: ACIRegressor | None = None
    # Sub-project F: persistent on-disk run directory (when save_full_state
    # succeeded). Used by the cloud Job to mirror artifacts into GCS.
    run_dir: Path | None = None


def _start_mlflow_run():
    """Returns (mlflow_module, run_context). If MLflow isn't configured, run
    context is a no-op so `run()` still works for tests."""
    settings = get_settings().mlflow
    try:
        import mlflow
    except ImportError:
        return None, _NoopCtx()
    if not settings.tracking_uri:
        return None, _NoopCtx()

    mlflow.set_tracking_uri(settings.tracking_uri)
    if settings.experiment:
        mlflow.set_experiment(settings.experiment)
    return mlflow, mlflow.start_run()


class _NoopCtx:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    info = type("I", (), {"run_id": None})()


def _calibrate_aci(
    estimator: Any, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[ACIRegressor | None, np.ndarray, np.ndarray]:
    """Carve a calibration slice off the val split (from the end, preserving
    temporal order), build an ACI wrapper from the already-fit base. Returns
    (aci, X_val_metric, y_val_metric) — the metric set is val minus the calib slice.

    Returns aci=None when the val split is too small for a usable calibration."""
    n = len(y_val)
    n_calib = int(n * _ACI_CALIB_FRACTION)
    if n_calib < _ACI_MIN_ROWS:
        return None, X_val, y_val
    # Calibration is the *tail* — closest to the future deployment distribution.
    X_calib, y_calib = X_val[-n_calib:], y_val[-n_calib:]
    X_metric, y_metric = X_val[:-n_calib], y_val[:-n_calib]
    aci = ACIRegressor.from_fitted(estimator, X_calib, y_calib, alpha=0.1)
    log.info("train.aci_calibrated", n_calib=n_calib, alpha=0.1)
    return aci, X_metric, y_metric


def _publish_to_state(
    estimator: Any, backend_kind: str, X_val: np.ndarray, aci: ACIRegressor | None
) -> None:
    """Set the active ModelHandle in api.state.STATE. The agent tool, Prefect
    evaluate_task, and FastAPI /predict all read from here. Single source of
    in-process truth."""
    from deepCab.api.state import STATE, ModelHandle

    background = X_val[: min(200, len(X_val))]
    STATE.set_model(
        ModelHandle(estimator=estimator, backend_kind=backend_kind, background=background, aci=aci)
    )


def _run_training(cfg: TrainConfig) -> TrainResult:
    """Core training body. Pure-ish: no env reads beyond what its dependencies
    already do (`get_settings()` reads `MLFLOW_*`, etc.). Returns the result
    with `run_dir` populated when the persistent state save succeeded."""
    set_all(cfg.seed, backend=cfg.backend.kind)
    log.info("train.start", backend=cfg.backend.kind, data=cfg.data.size, seed=cfg.seed)

    X_train, y_train = preprocess(cfg.data, split="train")
    X_val, y_val = preprocess(cfg.data, split="val")
    log.info("train.preprocessed", X_train=X_train.shape, X_val=X_val.shape)

    estimator = build_estimator(cfg.backend)
    estimator.fit(X_train, y_train, validation_data=(X_val, y_val))

    aci, X_val_metric, y_val_metric = _calibrate_aci(estimator, X_val, y_val)
    val_mae = float(_mae(estimator.predict(X_val_metric), y_val_metric))
    log.info("train.fit_done", val_mae=val_mae, aci=aci is not None)

    mlflow_mod, run_ctx = _start_mlflow_run()
    run_id: str | None = None
    run_dir: Path | None = None
    with run_ctx as r:
        if mlflow_mod is not None:
            log_params_flat(mlflow_mod, cfg)
            log_config_artifact(mlflow_mod, cfg)
            mlflow_mod.log_metric("val_mae", val_mae)
            run_id = r.info.run_id

        # P11: publish into STATE before any side-effects so the agent's
        # follow-up predict/explain tool calls see the fresh model + ACI.
        _publish_to_state(estimator, cfg.backend.kind, X_val, aci)

        from deepCab.api.state import STATE
        from deepCab.registry.dispatcher import save_full_state

        # FR-1: persistent on-disk state (single source of truth — used for the
        # MLflow upload, the ONNX export target, and the API lifespan loader's
        # cold-start rehydration). The LATEST pointer is updated atomically
        # inside save_full_state.
        if STATE.model is not None:
            run_dir = save_full_state(STATE.model, run_id=run_id)

        # model_path is the backend-native weights dir under the run_dir —
        # written by save_full_state above. Use it for MLflow + ONNX so we
        # don't double-write to a throwaway tempdir.
        model_path = (run_dir / "model") if run_dir is not None else None
        if mlflow_mod is not None and model_path is not None:
            mlflow_mod.log_artifacts(str(model_path.parent), artifact_path="model")

        provenance_path = emit_provenance(cfg, run_id=run_id, metrics={"val_mae": val_mae})

        # P12 wire-ups: model card + lineage + ONNX export, each independently
        # try/except so a failure in one doesn't kill the run.
        _emit_model_card(
            cfg,
            run_id=run_id,
            val_mae=val_mae,
            provenance_path=provenance_path,
            mlflow_mod=mlflow_mod,
        )
        _emit_lineage(cfg, run_id=run_id)
        if model_path is not None:
            _export_and_register_onnx(estimator, cfg, X_val[:1], model_path.parent)

    return TrainResult(
        run_id=run_id,
        backend_kind=cfg.backend.kind,
        val_mae=val_mae,
        model_path=str(model_path) if model_path is not None else "",
        estimator=estimator,
        background=X_val[: min(200, len(X_val))],
        aci=aci,
        run_dir=run_dir,
    )


def _push_to_gcs(local_dir: Path, gcs_uri: str) -> None:
    """Mirror a local run directory to GCS via `gsutil -m cp -r`.

    Kept as a thin shell-out (vs `google-cloud-storage`) so the runtime image
    needs only `google-cloud-sdk` already present in the deepcab-api image,
    and so unit tests can patch this single function without mocking a chain
    of boto-style clients."""
    import subprocess

    subprocess.run(
        ["gsutil", "-m", "cp", "-r", str(local_dir), gcs_uri],
        check=True,
    )


def run(cfg: TrainConfig | None) -> TrainResult:
    """Public entrypoint. Runs the training pipeline, then optionally mirrors
    the run directory to GCS when `REGISTRY_GCS_BUCKET` is set (Cloud Run Job
    path). Local / dev runs leave the env unset and skip the push."""
    result = _run_training(cfg)
    bucket = os.environ.get("REGISTRY_GCS_BUCKET")
    if bucket and result.run_id and result.run_dir is not None:
        # Tolerate either `deepcab-models` or `gs://deepcab-models` from
        # operators' env files. `removeprefix` is a no-op when the prefix
        # is absent.
        clean = bucket.removeprefix("gs://").rstrip("/")
        gcs_uri = f"gs://{clean}/runs/{result.run_id}/"
        log.info("train.gcs_push", local=str(result.run_dir), gcs=gcs_uri)
        _push_to_gcs(result.run_dir, gcs_uri)
    return result


def _emit_model_card(
    cfg: TrainConfig,
    run_id: str | None,
    val_mae: float,
    provenance_path: Path,
    mlflow_mod: Any,
) -> None:
    """Auto-emit MODEL_CARD.md alongside provenance.json. Logged to MLflow if
    available; persisted to runs/<run_id>/ regardless. Failure is non-fatal —
    a missing card on disk shouldn't kill the training run."""
    try:
        import json

        from deepCab.registry.model_card import write_model_card

        provenance = json.loads(provenance_path.read_text())
        card_path = provenance_path.parent / "MODEL_CARD.md"
        write_model_card(
            out_path=card_path,
            model_name=get_settings().mlflow.model_name or "deepcab",
            version=0,  # filled in when MLflow registers; 0 for local runs
            cfg=cfg,
            metrics={"val_mae": val_mae},
            provenance=provenance,
            shap_top=None,  # populated by a follow-up call once SHAP is computed
        )
        if mlflow_mod is not None:
            mlflow_mod.log_artifact(str(card_path), artifact_path="cards")
        log.info("train.model_card_emitted", path=str(card_path))
    except Exception as e:  # noqa: BLE001
        log.info("train.model_card_skipped", reason=f"{type(e).__name__}: {e}")


def _emit_lineage(cfg: TrainConfig, run_id: str | None) -> None:
    """Persist a LineageEdge row in the SQLite store. Hashes are computed over
    canonical JSON of (config + data ref + seed) so the same (cfg, data) pair
    deterministically produces the same edge — answering "are these runs
    comparable?" later via runs_sharing_input(input_hash)."""
    try:
        from deepCab.data.lineage import LineageEdge, hash_obj
        from deepCab.data.lineage_store import write_edge

        edge = LineageEdge(
            input_hash=hash_obj({"data": cfg.data.model_dump()}),
            preprocessor_hash=hash_obj(
                {"feature_pipeline": "features.pipeline.preprocess_features:v1"}
            ),
            split_hash=hash_obj({"strategy": "tail_30pct_for_aci", "seed": cfg.seed}),
            run_id=run_id,
        )
        write_edge(edge)
        log.info("train.lineage_written", run_id=run_id)
    except Exception as e:  # noqa: BLE001
        log.info("train.lineage_skipped", reason=f"{type(e).__name__}: {e}")


def _export_and_register_onnx(
    estimator: Any, cfg: TrainConfig, sample: np.ndarray, out_dir: Path
) -> None:
    """Export to ONNX and register the runtime so /predict can pick it up.
    Non-fatal: if the backend's converter isn't installed (e.g., onnxmltools
    missing for tree models) we just log and continue."""
    try:
        from deepCab.models.onnx_export import export_to_onnx
        from deepCab.serving.runtime import REGISTRY, ONNXRuntime

        onnx_path = out_dir / f"{cfg.backend.kind}.onnx"
        export_to_onnx(estimator, sample, onnx_path)
        rt = ONNXRuntime.from_path(onnx_path, backend_kind=cfg.backend.kind)
        REGISTRY.register(cfg.backend.kind, rt)
        REGISTRY.activate(cfg.backend.kind)
        log.info("train.onnx_registered", backend=cfg.backend.kind, path=str(onnx_path))
    except Exception as e:  # noqa: BLE001
        log.info("train.onnx_skipped", reason=f"{type(e).__name__}: {e}")


def _mae(pred, y) -> float:
    return float(np.mean(np.abs(np.asarray(pred).ravel() - np.asarray(y).ravel())))


# ---------- Hydra entry ----------


def _to_pydantic(cfg: DictConfig) -> TrainConfig:
    # OmegaConf -> dict -> Pydantic. Never pass DictConfig directly: the
    # discriminator comes through as ValueNode and discriminated unions fail.
    raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    return TrainConfig.model_validate(raw)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    run(_to_pydantic(cfg))


if __name__ == "__main__":
    main()
