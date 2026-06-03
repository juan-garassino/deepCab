"""OpenTelemetry tracer initialization. Real OTLP exporter pointing at the
otel-collector service in docker-compose.obs.yml. Span context propagates
automatically across FastAPI requests + the training pipeline + the agent loop."""
from __future__ import annotations

from functools import lru_cache

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from deepCab.schemas.settings import get_settings


@lru_cache(maxsize=1)
def init_tracing() -> trace.Tracer:
    obs = get_settings().obs
    resource = Resource.create({"service.name": obs.service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=obs.otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(obs.service_name)


def get_tracer() -> trace.Tracer:
    return init_tracing()
