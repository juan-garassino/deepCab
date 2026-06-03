"""FastAPI dependency providers. Single source of truth for what gets injected
into route handlers.

Killing `app.state.model` global (legacy api/fast.py:26): the model now flows
through Depends(get_model_handle), which 503s cleanly when none is loaded
instead of crashing at import time."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from deepCab.api.state import STATE, ModelHandle
from deepCab.schemas.settings import Settings, get_settings


def settings_dep() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(settings_dep)]


def get_model_handle() -> ModelHandle:
    if STATE.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no model loaded — POST /train first or run `make run_train`",
        )
    return STATE.model


ModelDep = Annotated[ModelHandle, Depends(get_model_handle)]


def api_key_guard(
    settings: SettingsDep,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Gate for training + agent endpoints.

    Resolution order (P13):
      1. settings.deepcab.api_key — explicit deepCab key (preferred).
      2. settings.openai.api_key  — legacy fallback for transitions; emits a
         warning so the operator notices and sets DEEPCAB_API_KEY.
      3. None of the above set — open access (dev convenience).
    """
    expected = settings.deepcab.api_key
    if not expected and settings.openai.api_key:
        # Don't log the key itself; log only the fact we fell back.
        from deepCab.obs.log import get_logger

        get_logger(__name__).warning(
            "api.auth.deprecated_openai_key_fallback",
            note="set DEEPCAB_API_KEY to separate the deepCab API key from OPENAI_API_KEY",
        )
        expected = settings.openai.api_key
    if not expected:
        return  # dev open mode
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid or missing X-API-Key",
        )


ApiKeyDep = Annotated[None, Depends(api_key_guard)]
