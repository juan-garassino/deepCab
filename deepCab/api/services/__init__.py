"""Service layer: business logic extracted out of routers.

Each service is a small dataclass with the provider strategies it needs
injected at construction time (via `deepCab.api.deps`). Routers become thin
adapters that parse the request and call `await svc.method(req)`.

The pattern keeps HTTP transport, validation, and orchestration cleanly
separated:
  - `routers/`  : FastAPI specifics (URL, methods, status codes, response_model)
  - `services/` : business logic, free of FastAPI imports
  - `providers.py` : side-effects (Slack, model registry, trace)
"""
from deepCab.api.services.agent import AgentService
from deepCab.api.services.explain import ExplanationService
from deepCab.api.services.predict import PredictionService
from deepCab.api.services.train import TrainingService

__all__ = [
    "AgentService",
    "ExplanationService",
    "PredictionService",
    "TrainingService",
]
