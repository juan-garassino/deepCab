"""The retrain flow must announce three lifecycle transitions to Slack:
running, success, failed. This is the only out-of-band notification path for
the flow (Prefect UI shows in-band state); without it, operators don't see
retrain progress in the same Slack channel as MLflow alias changes.

We patch the per-step helpers + slack.notify_flow_event so the test runs
without a dataset or MLflow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_retrain_flow_emits_slack_on_success() -> None:
    from deepCab.flow_v2 import retrain

    with (
        patch("deepCab.obs.slack.notify_flow_event") as mock_notify,
        patch.object(retrain, "_preprocess", return_value=MagicMock()),
        patch.object(retrain, "_train", return_value=MagicMock(run_id="r-9")),
        patch.object(retrain, "_evaluate", return_value={"val_mae": 3.4}),
    ):
        result = retrain.retrain_flow.fn()
    assert result is not None
    states = [c.kwargs.get("state") for c in mock_notify.call_args_list]
    assert "running" in states
    assert "success" in states


def test_retrain_flow_emits_slack_on_failure() -> None:
    from deepCab.flow_v2 import retrain

    with (
        patch("deepCab.obs.slack.notify_flow_event") as mock_notify,
        patch.object(retrain, "_preprocess", side_effect=RuntimeError("boom")),
    ):
        try:
            retrain.retrain_flow.fn()
        except RuntimeError:
            pass
    states = [c.kwargs.get("state") for c in mock_notify.call_args_list]
    assert "failed" in states
