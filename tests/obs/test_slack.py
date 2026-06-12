"""Unit tests for the in-process Slack backend (deepCab.obs.slack).

The backend is intentionally tiny: a single `post(text, *, tag, extra)` that
short-circuits when no webhook is configured and swallows network errors so a
flaky Slack never breaks training/inference. Named wrappers
(`notify_alias_change`, `notify_flow_event`) live in `deepCab.obs.notify`
and are tested separately in `tests/obs/test_notify.py`."""

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
