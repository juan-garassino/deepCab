"""FastAPI factory. `create_app()` is the single entry point used by:
    - uvicorn deepCab.api.app:create_app    (Makefile run_api)
    - tests/api/* via TestClient
    - docker-compose api service

Order matters: routers register their routes BEFORE PromMiddleware computes
`request.scope['route']`."""
from __future__ import annotations

import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from deepCab.api.lifespan import lifespan
from deepCab.api.middleware import PromMiddleware, RequestIDMiddleware
from deepCab.api.routers import agent as agent_router
from deepCab.api.routers import explain as explain_router
from deepCab.api.routers import meta as meta_router
from deepCab.api.routers import monitor as monitor_router
from deepCab.api.routers import predict as predict_router
from deepCab.api.routers import train as train_router

try:
    from deepCab.api.routers import graphql as graphql_router  # P18, optional
except Exception:  # noqa: BLE001
    graphql_router = None  # type: ignore[assignment]
from deepCab.obs.log import get_logger
from deepCab.obs.prom import REGISTRY
from deepCab.schemas.settings import get_settings

log = get_logger(__name__)


def _resolve_cors_origins() -> list[str]:
    """Comma-split OBS_CORS_ALLOW_ORIGINS. Refuses "*" in prod — explicit
    allowlist required. Returns [] when unset in prod (which makes any
    cross-origin request fail closed)."""
    settings = get_settings()
    raw = (settings.obs.cors_allow_origins or "").strip()
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if settings.app_env == "prod" and "*" in origins:
        raise RuntimeError(
            "OBS_CORS_ALLOW_ORIGINS='*' is forbidden when APP_ENV=prod; "
            "set an explicit allowlist of origin URLs."
        )
    return origins


def create_app() -> FastAPI:
    app = FastAPI(
        title="deepCab API",
        description="NYC taxi-fare MLOps learning hub",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_resolve_cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(PromMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # /metrics is served by prometheus_client's ASGI app against our custom
    # registry, mounted as a sub-app so its scrape doesn't go through PromMiddleware
    # (which would double-count).
    app.mount("/metrics", make_asgi_app(registry=REGISTRY))

    app.include_router(meta_router.router)
    app.include_router(monitor_router.router)
    app.include_router(predict_router.router)
    app.include_router(explain_router.router)
    app.include_router(train_router.router)
    app.include_router(agent_router.router)
    if graphql_router is not None:
        app.include_router(graphql_router.router, prefix="/graphql")

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        log.error("api.unhandled", error=str(exc), trace=traceback.format_exc()[:2000])
        return JSONResponse(
            status_code=500,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    # OTel auto-instrumentation if available (optional)
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception as e:  # noqa: BLE001
        log.info("api.otel_instrument.skipped", reason=str(e))

    return app


# Convenience: `uvicorn deepCab.api.app:app --reload`
app = create_app()
