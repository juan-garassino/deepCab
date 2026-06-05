"""B.2 regression: `file:` URI in env values is resolved at the settings-source
layer (`FileUriEnvSettingsSource`), not via per-field validators."""

from __future__ import annotations

from pathlib import Path

from deepCab.schemas.settings import ObsSettings


def test_file_uri_resolved_for_string_field(tmp_path: Path, monkeypatch):
    f = tmp_path / "webhook"
    f.write_text("https://hooks.slack.com/abc\n")
    monkeypatch.setenv("OBS_SLACK_WEBHOOK_URL", f"file:{f}")
    s = ObsSettings()
    assert s.slack_webhook_url == "https://hooks.slack.com/abc"


def test_plain_string_left_untouched(monkeypatch):
    monkeypatch.setenv("OBS_SLACK_WEBHOOK_URL", "https://hooks.slack.com/literal")
    s = ObsSettings()
    assert s.slack_webhook_url == "https://hooks.slack.com/literal"
