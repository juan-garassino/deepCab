"""Observability: pure-stdlib JSONL tracer + structlog logger.

Heavy SDKs (opentelemetry, prometheus_client) live in obs.otel and obs.prom
respectively — import them directly only where you need them (FastAPI lifespan
in Phase 6, agent loop in Phase 8). This keeps `from deepCab.obs import
get_logger` importable on any install."""
from deepCab.obs.jsonl import Tracer, attach_result  # noqa: F401
from deepCab.obs.log import get_logger  # noqa: F401
