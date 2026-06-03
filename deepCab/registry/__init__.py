"""Backend-agnostic registry: save/load artifacts + MLflow alias mgmt.
Replaces legacy deepCab/model_target/."""
from deepCab.registry.dispatcher import (  # noqa: F401
    load_artifact,
    load_state_from_disk,
    read_latest_run_id,
    save_artifact,
    save_full_state,
    set_alias,
)
from deepCab.registry.model_card import write_model_card  # noqa: F401
