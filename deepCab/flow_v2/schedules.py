"""Prefect deployment definitions. Run `python -m deepCab.flow_v2.schedules`
to register them against the configured PREFECT_API_URL (compose: http://prefect:4200).

A deployment binds (flow, parameters, schedule, work-pool) and is what the
agent's future `schedule_retrain` tool would create programmatically."""

from __future__ import annotations

from deepCab.flow_v2.retrain import retrain_flow
from deepCab.schemas.config import DataRef, TrainConfig, XGBConfig

DEFAULT_RETRAIN_CFG = TrainConfig(
    backend=XGBConfig(),
    data=DataRef(size="1k", validation_size="1k"),
)


def deploy_nightly() -> None:
    """Cron 02:00 UTC daily retrain on the 1k slice as a smoke. Production
    schedules would point at 100k+ and a beefier backend.

    Prefect 3 prefers `flow.serve()` for development and `flow.deploy(...)` +
    a worker for prod. We use `serve` here so `python -m deepCab.flow_v2.schedules`
    blocks and runs the schedule in-process — convenient for the compose
    `prefect-agent` service to invoke."""
    retrain_flow.serve(
        name="deepcab-retrain-nightly",
        cron="0 2 * * *",
        parameters={"cfg": DEFAULT_RETRAIN_CFG.model_dump()},
        description="Nightly retrain on the 1k slice; smoke + canary.",
        tags=["deepcab", "retrain", "nightly"],
    )


if __name__ == "__main__":
    deploy_nightly()
