"""Structured logging via structlog. JSON-rendered in prod, console-pretty in dev."""

from __future__ import annotations

import logging
from functools import lru_cache

import structlog

from deepCab.schemas.settings import get_settings


@lru_cache(maxsize=1)
def configure_logging() -> structlog.stdlib.BoundLogger:
    settings = get_settings()
    is_dev = settings.app_env == "dev"

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    renderer = (
        structlog.dev.ConsoleRenderer(colors=True)
        if is_dev
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger("deepcab")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name or "deepcab")
