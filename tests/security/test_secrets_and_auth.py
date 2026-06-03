"""P13 security: api-key precedence, file: secrets resolution, CORS allowlist."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def fresh_settings(monkeypatch):
    """Reset Settings cache before + after each test so env mutations take effect."""
    from deepCab.schemas.settings import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield
    get_settings.cache_clear()  # type: ignore[attr-defined]


# ---- api-key precedence -------------------------------------------------


def test_api_key_guard_prefers_deepcab_key_over_openai(monkeypatch, fresh_settings) -> None:
    monkeypatch.setenv("DEEPCAB_API_KEY", "deepcab-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    from deepCab.api.app import create_app

    client = TestClient(create_app())
    body = {
        "config": {
            "backend": {"kind": "xgb"},
            "data": {"size": "1k", "validation_size": "1k"},
            "seed": 1,
        }
    }
    # Wrong key (uses OpenAI value) -> 403
    r = client.post("/train", json=body, headers={"X-API-Key": "openai-secret"})
    assert r.status_code == 403
    # Correct deepCab key -> 200
    r = client.post("/train", json=body, headers={"X-API-Key": "deepcab-secret"})
    assert r.status_code == 200


def test_api_key_guard_falls_back_to_openai_when_deepcab_unset(monkeypatch, fresh_settings) -> None:
    monkeypatch.delenv("DEEPCAB_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-secret")

    from deepCab.api.app import create_app

    client = TestClient(create_app())
    body = {
        "config": {
            "backend": {"kind": "xgb"},
            "data": {"size": "1k", "validation_size": "1k"},
            "seed": 1,
        }
    }
    r = client.post("/train", json=body, headers={"X-API-Key": "legacy-secret"})
    assert r.status_code == 200


def test_api_key_guard_open_when_no_keys_set(monkeypatch, fresh_settings) -> None:
    monkeypatch.delenv("DEEPCAB_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from deepCab.api.app import create_app

    client = TestClient(create_app())
    body = {
        "config": {
            "backend": {"kind": "xgb"},
            "data": {"size": "1k", "validation_size": "1k"},
            "seed": 1,
        }
    }
    # No keys => open access (dev convenience)
    r = client.post("/train", json=body)
    assert r.status_code == 200


# ---- file: secret resolution -------------------------------------------


def test_file_prefix_resolves_to_file_content(tmp_path: Path, monkeypatch, fresh_settings) -> None:
    secret_path = tmp_path / "deepcab_api_key"
    secret_path.write_text("secret-from-file\n")  # trailing newline is stripped

    monkeypatch.setenv("DEEPCAB_API_KEY", f"file:{secret_path}")
    from deepCab.schemas.settings import get_settings

    assert get_settings().deepcab.api_key == "secret-from-file"


def test_file_prefix_raises_when_missing(tmp_path: Path, monkeypatch, fresh_settings) -> None:
    monkeypatch.setenv("DEEPCAB_API_KEY", f"file:{tmp_path / 'nope'}")
    from deepCab.schemas.settings import get_settings

    # B.2: the FileNotFoundError is raised inside FileUriEnvSettingsSource;
    # pydantic-settings wraps it in a SettingsError with the underlying
    # FileNotFoundError as __cause__.
    with pytest.raises(Exception) as exc_info:
        get_settings()
    chain = []
    cur = exc_info.value
    while cur is not None:
        chain.append(cur)
        cur = cur.__cause__
    assert any(isinstance(e, FileNotFoundError) for e in chain), (
        f"FileNotFoundError not in exception chain: {[type(e).__name__ for e in chain]}"
    )


def test_plain_env_value_unchanged(monkeypatch, fresh_settings) -> None:
    monkeypatch.setenv("DEEPCAB_API_KEY", "plain-string")
    from deepCab.schemas.settings import get_settings

    assert get_settings().deepcab.api_key == "plain-string"


# ---- CORS --------------------------------------------------------------


def test_cors_allowlist_from_env(monkeypatch, fresh_settings) -> None:
    monkeypatch.setenv("OBS_CORS_ALLOW_ORIGINS", "http://a.test,http://b.test")
    from deepCab.api.app import create_app

    app = create_app()
    # CORSMiddleware stores the list internally; we check the resolved config
    middleware = [m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"][0]
    assert set(middleware.kwargs["allow_origins"]) == {"http://a.test", "http://b.test"}


def test_cors_star_forbidden_in_prod(monkeypatch, fresh_settings) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("OBS_CORS_ALLOW_ORIGINS", "*")
    from deepCab.api.app import create_app

    with pytest.raises(RuntimeError, match="forbidden when APP_ENV=prod"):
        create_app()
