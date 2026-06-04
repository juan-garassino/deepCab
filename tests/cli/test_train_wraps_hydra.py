"""`train` must accept Hydra-style overrides and forward them to
`deepCab.training.train.run`. We mock `_run` so the test doesn't actually
fit a model — we only verify the override list reaches the underlying
entry as a TrainConfig and that the CLI completes successfully."""
from __future__ import annotations

from typer.testing import CliRunner

from deepCab.cli import app

runner = CliRunner()


def test_train_forwards_hydra_overrides_to_run(monkeypatch):
    captured: dict = {}

    class _FakeResult:
        run_id = "fake-run"
        backend_kind = "tf_mlp"
        val_mae = 1.234

    def _fake_run(cfg):
        # cfg is a TrainConfig instance.
        captured["cfg"] = cfg
        return _FakeResult()

    # Patch the symbol the CLI imports lazily; we patch the source module so
    # the late `from deepCab.training.train import run as _run` picks it up.
    monkeypatch.setattr("deepCab.training.train.run", _fake_run)

    result = runner.invoke(app, ["train", "backend=tf_mlp", "data=1k", "seed=7"])
    assert result.exit_code == 0, result.output
    assert "trained:" in result.output
    assert "tf_mlp" in result.output

    cfg = captured.get("cfg")
    assert cfg is not None, "run() was not invoked"
    # Hydra/OmegaConf composed the overrides into the TrainConfig — the
    # discriminator/backend, dataset size and seed should all line up.
    assert cfg.backend.kind == "tf_mlp"
    assert cfg.data.size == "1k"
    assert cfg.seed == 7


def test_train_runs_with_no_overrides(monkeypatch):
    captured: dict = {}

    class _FakeResult:
        run_id = None
        backend_kind = "tf_mlp"
        val_mae = 0.0

    def _fake_run(cfg):
        captured["cfg"] = cfg
        return _FakeResult()

    monkeypatch.setattr("deepCab.training.train.run", _fake_run)

    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0, result.output
    assert "cfg" in captured
