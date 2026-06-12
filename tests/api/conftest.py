"""Shared fixtures for the API test suite.

B.6 consolidation: `_clean_state` was duplicated between `test_routers.py`
(reset model + tasks) and `test_graphql.py` (reset model only). Hoisted here
in its strictest form so both files inherit the same isolation."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _scrub_auth_env(monkeypatch: pytest.MonkeyPatch):
    """Strip auth env vars before the api package reads them.

    A developer with OPENAI_API_KEY exported (e.g. for `secrets rotate`) would
    otherwise trip `api_key_guard` into demanding a matching header on every
    test request. Auth behaviour is exercised explicitly in `tests/security/`.
    """
    for var in ("DEEPCAB_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    from deepCab.schemas.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset in-process STATE (model handle + train-task table) around every
    API test so no residual fitted model or pending task bleeds between cases."""
    from deepCab.api.state import STATE

    STATE.model = None
    STATE.tasks.clear()
    yield
    STATE.model = None
    STATE.tasks.clear()
