"""Single MLflow client adapter — contract tests.

`obs/mlflow.get_mlflow_client` is the one construction point used by registry/,
agent/, flow_v2/, and api/. These lock the None-handling contract every call
site relies on, without needing a live MLflow server.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

from deepCab.obs import mlflow as mlflow_adapter


def _settings(tracking_uri):
    return SimpleNamespace(mlflow=SimpleNamespace(tracking_uri=tracking_uri))


def test_returns_none_when_uri_unset_and_require_uri_default() -> None:
    with patch.object(mlflow_adapter, "get_settings", return_value=_settings(None)):
        assert mlflow_adapter.get_mlflow_client() is None


def test_returns_none_when_mlflow_not_installed() -> None:
    # Simulate the dependency being absent: block the import.
    with (
        patch.object(mlflow_adapter, "get_settings", return_value=_settings("sqlite:///x.db")),
        patch.dict(sys.modules, {"mlflow": None}),
    ):
        assert mlflow_adapter.get_mlflow_client() is None


def test_require_uri_false_builds_even_without_uri() -> None:
    # require_uri=False must not short-circuit on a missing URI; it should reach
    # the construction call. We stub the construction so the test needs no
    # mlflow install and asserts the URI guard didn't bail early.
    fake_client = object()
    fake_module = SimpleNamespace(set_tracking_uri=lambda _u: None)
    fake_tracking = SimpleNamespace(MlflowClient=lambda: fake_client)
    with (
        patch.object(mlflow_adapter, "get_settings", return_value=_settings(None)),
        patch.dict(
            sys.modules,
            {"mlflow": fake_module, "mlflow.tracking": fake_tracking},
        ),
    ):
        assert mlflow_adapter.get_mlflow_client(require_uri=False) is fake_client


def test_sets_tracking_uri_when_configured() -> None:
    calls = []
    fake_client = object()
    fake_module = SimpleNamespace(set_tracking_uri=lambda u: calls.append(u))
    fake_tracking = SimpleNamespace(MlflowClient=lambda: fake_client)
    with (
        patch.object(mlflow_adapter, "get_settings", return_value=_settings("sqlite:///y.db")),
        patch.dict(sys.modules, {"mlflow": fake_module, "mlflow.tracking": fake_tracking}),
    ):
        client = mlflow_adapter.get_mlflow_client()
    assert client is fake_client
    assert calls == ["sqlite:///y.db"]
