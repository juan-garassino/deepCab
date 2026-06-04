"""Observability: pure-stdlib JSONL tracer (obs.jsonl) + structlog logger
(obs.log). Heavy SDKs (opentelemetry, prometheus_client) live in obs.otel and
obs.prom — import each submodule directly so optional SDK installs stay
optional."""
