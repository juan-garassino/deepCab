"""`deepcab train [overrides...]` — wraps `deepCab.training.train.run`.

Positional arguments are forwarded to Hydra's `compose()` as overrides, so
the same `backend=tf_mlp data=1k seed=7` syntax works under the unified CLI
as under the legacy Hydra entry (`python -m deepCab.training.train ...`).
The Hydra entry is preserved — this is additive surface.
"""
from __future__ import annotations

from typing import Optional

import typer


def train(
    overrides: Optional[list[str]] = typer.Argument(
        None,
        help="Hydra-style overrides, e.g. backend=tf_mlp data=1k seed=7",
        show_default=False,
    ),
) -> None:
    """Train a model. Accepts Hydra-style key=value overrides as positional args."""
    # Lazy imports keep `deepcab --help` snappy and avoid pulling Hydra +
    # the full training stack into the import graph for `status` / `serve`.
    from hydra import compose, initialize
    from omegaconf import DictConfig, OmegaConf

    from deepCab.schemas.config import TrainConfig
    from deepCab.training.train import run as _run

    overrides_list = list(overrides) if overrides else []

    # config_path is relative to *this file* (cli/) — climb one level up to
    # `deepCab/` then descend into `config/`. The Hydra entry in
    # training/train.py uses an absolute path; for `compose()` we use the
    # standard relative form so it works wherever the user invokes us.
    with initialize(config_path="../config", version_base=None):
        cfg: DictConfig = compose(config_name="config", overrides=overrides_list)

    # OmegaConf → dict → Pydantic mirrors the bridge in training/train.py::main.
    raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    train_cfg = TrainConfig.model_validate(raw)
    result = _run(train_cfg)
    typer.echo(
        f"trained: run_id={result.run_id} backend={result.backend_kind} val_mae={result.val_mae:.4f}"
    )
