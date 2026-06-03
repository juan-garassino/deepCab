"""Shared pytest fixtures.

P14 cleanup: the legacy env-var stub block (CHUNK_SIZE / DATASET_SIZE / etc.)
went away with the legacy ml_logic/params.py import. New tests use the Pydantic
Settings model directly; monkeypatch env vars and call get_settings.cache_clear()
in your fixture if you need to override (see tests/security/test_secrets_and_auth.py
for the pattern)."""
from __future__ import annotations

import pytest

from deepCab.schemas.settings import get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Force a fresh Settings() per test so monkeypatch env changes always
    take effect. The @lru_cache on get_settings is meant for prod (load once)
    not tests."""
    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield
    get_settings.cache_clear()  # type: ignore[attr-defined]
