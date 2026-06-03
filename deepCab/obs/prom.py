"""Prometheus metrics registry. Histograms for API latency + prediction values,
counters for agent tool calls + training epochs. Exposed at /metrics on the FastAPI app."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY = CollectorRegistry()

# API
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "API request latency",
    labelnames=("method", "route", "status"),
    registry=REGISTRY,
)

prediction_value = Histogram(
    "deepcab_prediction_value",
    "Distribution of predicted fares",
    labelnames=("backend_kind",),
    buckets=(2, 5, 10, 20, 40, 80, 160, 400),
    registry=REGISTRY,
)

# Training
training_epoch_duration_seconds = Histogram(
    "deepcab_training_epoch_duration_seconds",
    "Per-epoch training wall time",
    labelnames=("backend_kind",),
    registry=REGISTRY,
)

training_run_total = Counter(
    "deepcab_training_run_total",
    "Total training runs",
    labelnames=("backend_kind", "status"),
    registry=REGISTRY,
)

# Agent
agent_tool_call_total = Counter(
    "deepcab_agent_tool_call_total",
    "Total agent tool calls",
    labelnames=("tool", "status"),
    registry=REGISTRY,
)

agent_tokens_total = Counter(
    "deepcab_agent_tokens_total",
    "Total OpenAI tokens consumed",
    labelnames=("model", "kind"),  # kind = prompt|completion
    registry=REGISTRY,
)
