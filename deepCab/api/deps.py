"""FastAPI dependency factory tray.

After the services/providers refactor, this module's job is to:
  1. Resolve singletons (`get_settings`, `get_model_handle`).
  2. Pick a provider strategy based on settings (`get_trace_provider`).
  3. Compose a service out of (handle, providers) for each router endpoint.

Tests override providers via `app.dependency_overrides[get_trace_provider]`,
which makes per-test isolation trivial.

Public surface for routers (unchanged names so existing imports keep working):
  - SettingsDep              — alias for Annotated[Settings, Depends(...)]
  - get_model_handle         — 503s when no model is loaded
  - api_key_guard            — X-API-Key gate for /train and /agent
  - get_prediction_service / get_explanation_service / get_training_service
    / get_agent_service       — per-router service factories.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from deepCab.api.providers import JsonlTraceProvider, TraceProvider
from deepCab.api.services.agent import AgentService
from deepCab.api.services.explain import ExplanationService
from deepCab.api.services.predict import PredictionService
from deepCab.api.services.train import TrainingService
from deepCab.api.state import STATE, ModelHandle
from deepCab.schemas.settings import Settings, get_settings

# ---------------------------------------------------------------------------
# Settings + model handle (unchanged contracts)
# ---------------------------------------------------------------------------


def settings_dep() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(settings_dep)]


def get_model_handle() -> ModelHandle:
    """503 when no model is loaded — POST /train first or run `make run_train`.

    Reads STATE directly (rather than going through a ModelHandleProvider) so
    the dependency override pattern in existing tests (`STATE.model = ...`)
    keeps working. Tests that want to stub the handle can either set STATE.model
    or override this dep with `app.dependency_overrides[get_model_handle]`."""
    if STATE.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no model loaded — POST /train first or run `make run_train`",
        )
    return STATE.model


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def api_key_guard(
    settings: SettingsDep,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Gate for training + agent endpoints.

    Resolution order (P13):
      1. settings.deepcab.api_key — explicit deepCab key (preferred).
      2. settings.openai.api_key  — legacy fallback; warns so the operator sees it.
      3. Neither set — open access (dev convenience).
    """
    expected = settings.deepcab.api_key
    if not expected and settings.openai.api_key:
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


# ---------------------------------------------------------------------------
# Provider factories (strategy selection happens here)
# ---------------------------------------------------------------------------


def get_trace_provider() -> TraceProvider:
    """JSONL on disk is the only production strategy today. Tests override to
    `NullTraceProvider` so they don't write to ~/.lewagon/.../traces."""
    return JsonlTraceProvider()


# ---------------------------------------------------------------------------
# Service factories
# ---------------------------------------------------------------------------


def get_prediction_service(
    model: ModelHandle = Depends(get_model_handle),
) -> PredictionService:
    return PredictionService(model=model)


def get_explanation_service(
    model: ModelHandle = Depends(get_model_handle),
) -> ExplanationService:
    return ExplanationService(model=model)


def get_training_service() -> TrainingService:
    return TrainingService()


def get_agent_service(
    trace: TraceProvider = Depends(get_trace_provider),
) -> AgentService:
    return AgentService(trace=trace)
