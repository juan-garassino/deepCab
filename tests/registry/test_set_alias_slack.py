"""set_alias must (a) delegate to the MLflow backend and (b) post a Slack
notification with the same model/alias/version arguments.

The backend call is patched so the test never reaches MLflow; what we care
about here is the contract between the wrapper and the helper, not whether
MLflow itself accepts the alias."""
from __future__ import annotations

from unittest.mock import patch

from deepCab.registry import dispatcher


def test_set_alias_posts_slack() -> None:
    with patch("deepCab.obs.slack.notify_alias_change") as mock_notify, patch(
        "deepCab.registry.dispatcher._set_alias_backend"
    ) as mock_backend:
        dispatcher.set_alias(model="deepcab", alias="champion", version="7")
    mock_backend.assert_called_once_with(model="deepcab", alias="champion", version="7")
    mock_notify.assert_called_once_with(model="deepcab", alias="champion", version="7")
