"""FastAPI lifespan: startup loads telemetry + (optional) @champion model from
MLflow. Shutdown is a no-op.

The autoloader closes the Phase 10 gap: without it, every API restart left
`/predict` returning 503 until something explicitly trained a model. Now the
last-promoted @champion alias rehydrates STATE.model at startup; if MLflow
isn't configured or no @champion exists, we no-op cleanly and the route stays
503 until the agent's `train` tool or `make run_train` populates STATE."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from deepCab.api.state import STATE
from deepCab.obs.log import get_logger
from deepCab.schemas.settings import get_settings

log = get_logger(__name__)


def _try_init_otel() -> None:
    """Initialize the OTel tracer. In dev, missing SDK is logged + skipped;
    in prod, it raises — production deployments must have telemetry."""
    try:
        from deepCab.obs.otel import init_tracing  # lazy: optional install

        init_tracing()
    except Exception as e:  # noqa: BLE001
        if get_settings().app_env == "prod":
            raise RuntimeError(
                f"OTel tracing init failed in prod (APP_ENV=prod): {type(e).__name__}: {e}. "
                "Install opentelemetry-sdk + opentelemetry-exporter-otlp."
            ) from e
        log.info("api.otel.skipped", reason=str(e), env="non-prod")


def _try_load_local_latest() -> bool:
    """FR-1: rehydrate STATE.model from <REGISTRY_LOCAL_PATH>/runs/LATEST.

    This is the practical autoload path — after any `make run_train` or agent
    `train` tool call, the LATEST pointer holds a run_id that
    `registry.load_state_from_disk` can read back into a full ModelHandle
    (estimator + background + ACI).

    Returns True if STATE.model was populated."""
    try:
        from deepCab.registry.dispatcher import load_state_from_disk, read_latest_run_id

        rid = read_latest_run_id()
        if not rid:
            log.info("api.local_latest.skipped", reason="no LATEST pointer")
            return False
        handle = load_state_from_disk(rid)
        STATE.set_model(handle)
        log.info(
            "api.local_latest.loaded",
            run_id=rid,
            backend=handle.backend_kind,
            has_aci=handle.aci is not None,
        )
        return True
    except Exception as e:  # noqa: BLE001
        log.info("api.local_latest.skipped", reason=f"{type(e).__name__}: {e}")
        return False


def _try_load_champion() -> None:
    """Load order: MLflow `@champion` first (logs what it finds — full URI
    load lands when the dispatcher gains MLflow-URI awareness, post-MVP),
    then fall back to the local LATEST pointer."""
    settings = get_settings().mlflow
    mlflow_ok = False
    if settings.tracking_uri and settings.model_name:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(settings.tracking_uri)
            client = MlflowClient()
            mv = client.get_model_version_by_alias(settings.model_name, settings.champion_alias)
            log.info(
                "api.champion.found",
                version=mv.version,
                run_id=mv.run_id,
                note="MLflow URI rehydration deferred — falling through to local LATEST",
            )
            mlflow_ok = True
        except Exception as e:  # noqa: BLE001
            log.info("api.champion.skipped", reason=f"{type(e).__name__}: {e}")

    # Local-disk fallback always runs — it's the practical path.
    loaded = _try_load_local_latest()
    if not loaded and not mlflow_ok:
        log.info("api.autoload.skipped", note="no model loaded; /predict will 503 until /train")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api.startup")
    _try_init_otel()
    _try_load_champion()
    log.info("api.ready", model_loaded=STATE.model is not None)
    yield
    log.info("api.shutdown")
