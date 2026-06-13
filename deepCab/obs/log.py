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
        # Must stay False: the default PrintLoggerFactory binds to sys.stdout at
        # logger-construction time. Caching would freeze that binding to whatever
        # stream was current on first emission — e.g. a CliRunner/pytest capture
        # buffer — and every later write would hit a closed file once that buffer
        # is torn down ("I/O operation on closed file" cascading across tests).
        cache_logger_on_first_use=False,
    )
    return structlog.get_logger("deepcab")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name or "deepcab")
