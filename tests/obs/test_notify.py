"""Cross-backend notify fan-out.

`obs/notify.post` must call every registered backend exactly once with the
same (text, tag, extra) tuple. Adding a backend is a one-line edit in
`_BACKENDS`; the wrappers (`notify_alias_change`, `notify_flow_event`) fix
the tag + payload schema once for everyone.
"""

from __future__ import annotations

from unittest.mock import patch

from deepCab.obs import notify


def test_post_fans_out_to_every_backend() -> None:
    with patch("deepCab.obs.notify._BACKENDS", new=[]) as backends:
        b1, b2 = _RecordingBackend(), _RecordingBackend()
        backends.extend([b1, b2])
        notify.post("hello", tag="ci", extra={"k": "v"})

    for b in (b1, b2):
        assert b.calls == [("hello", "ci", {"k": "v"})]


def test_notify_alias_change_uses_mlflow_tag() -> None:
    with patch("deepCab.obs.notify.post") as mock_post:
        notify.notify_alias_change(model="deepcab", alias="champion", version="3")
    mock_post.assert_called_once()
    kwargs = mock_post.call_args.kwargs
    assert kwargs["tag"] == "mlflow"
    assert kwargs["extra"] == {"model": "deepcab", "alias": "champion", "version": "3"}


def test_notify_flow_event_uses_flow_tag() -> None:
    with patch("deepCab.obs.notify.post") as mock_post:
        notify.notify_flow_event(flow="retrain", state="success", run_id="r-1")
    kwargs = mock_post.call_args.kwargs
    assert kwargs["tag"] == "flow"
    assert kwargs["extra"] == {"run_id": "r-1"}


class _RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, text, *, tag, extra=None):
        self.calls.append((text, tag, dict(extra) if extra else extra))
