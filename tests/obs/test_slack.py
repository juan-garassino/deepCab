"""Unit tests for the in-process Slack helper (deepCab.obs.slack).

The helper is intentionally tiny: a single `post(text, *, tag, extra)` that
short-circuits when no webhook is configured and swallows network errors so a
flaky Slack never breaks training/inference. Two named wrappers — for the two
in-process call sites (registry alias change + flow lifecycle) — fix the tag
on each message so downstream filters/routing in Slack work."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from deepCab.obs import slack


def test_post_noop_when_webhook_unset(monkeypatch) -> None:
    monkeypatch.setattr(slack, "_webhook_url", lambda: None)
    with patch("requests.post") as mock_post:
        slack.post("hello", tag="ci")
    mock_post.assert_not_called()


def test_post_formats_with_tag(monkeypatch) -> None:
    monkeypatch.setattr(slack, "_webhook_url", lambda: "https://hooks.slack.com/abc")
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        slack.post("training started", tag="flow", extra={"run_id": "r-7"})

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "https://hooks.slack.com/abc"
    payload = kwargs["json"]
    assert payload["text"].startswith("[flow]")
    assert "training started" in payload["text"]
    assert "r-7" in payload["text"]


def test_post_swallows_network_errors(monkeypatch) -> None:
    monkeypatch.setattr(slack, "_webhook_url", lambda: "https://hooks.slack.com/abc")
    with patch("requests.post", side_effect=Exception("boom")):
        slack.post("x", tag="ci")  # must not raise


def test_notify_alias_change_uses_mlflow_tag(monkeypatch) -> None:
    monkeypatch.setattr(slack, "_webhook_url", lambda: "https://hooks.slack.com/abc")
    with patch("deepCab.obs.slack.post") as mock_post:
        slack.notify_alias_change(model="deepcab", alias="champion", version="3")
    mock_post.assert_called_once()
    kwargs = mock_post.call_args.kwargs
    assert kwargs["tag"] == "mlflow"


def test_notify_flow_event_uses_flow_tag(monkeypatch) -> None:
    monkeypatch.setattr(slack, "_webhook_url", lambda: "https://hooks.slack.com/abc")
    with patch("deepCab.obs.slack.post") as mock_post:
        slack.notify_flow_event(flow="retrain", state="success", run_id="r-1")
    kwargs = mock_post.call_args.kwargs
    assert kwargs["tag"] == "flow"
