"""Regression tests for the env-driven GCS artifact push after training.

When `REGISTRY_GCS_BUCKET` is set, `train.run()` must mirror the local run
directory to `gs://<bucket>/runs/<run_id>/` via `_push_to_gcs`. When the env
is unset, no GCS calls happen. We stub out the helpers so the test never
shells out to `gsutil` and never reaches the real cloud."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from deepCab.training import train as tmod


def test_gcs_push_called_when_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv("REGISTRY_GCS_BUCKET", "deepcab-models")
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "model.onnx").write_bytes(b"dummy")
    with (
        patch.object(
            tmod,
            "_run_training",
            return_value=MagicMock(run_id="run-1", run_dir=run_dir),
        ),
        patch.object(tmod, "_push_to_gcs") as mock_push,
    ):
        tmod.run(cfg=None)
    mock_push.assert_called_once()
    args = mock_push.call_args.args
    assert args[0] == run_dir
    assert args[1] == "gs://deepcab-models/runs/run-1/"


def test_gcs_push_skipped_when_env_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("REGISTRY_GCS_BUCKET", raising=False)
    run_dir = tmp_path / "run-2"
    run_dir.mkdir()
    with (
        patch.object(
            tmod,
            "_run_training",
            return_value=MagicMock(run_id="run-2", run_dir=run_dir),
        ),
        patch.object(tmod, "_push_to_gcs") as mock_push,
    ):
        tmod.run(cfg=None)
    mock_push.assert_not_called()


def test_gcs_push_handles_gs_prefix_in_bucket(monkeypatch, tmp_path):
    """A bucket value like `gs://deepcab-models` must NOT double-prefix the URI."""
    monkeypatch.setenv("REGISTRY_GCS_BUCKET", "gs://deepcab-models")
    run_dir = tmp_path / "run-3"
    run_dir.mkdir()
    with (
        patch.object(
            tmod,
            "_run_training",
            return_value=MagicMock(run_id="run-3", run_dir=run_dir),
        ),
        patch.object(tmod, "_push_to_gcs") as mock_push,
    ):
        tmod.run(cfg=None)
    mock_push.assert_called_once()
    assert mock_push.call_args.args[1] == "gs://deepcab-models/runs/run-3/"
