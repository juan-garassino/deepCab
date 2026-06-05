"""JSON-parity tests for `schemas/enums.py`.

The contract: every enum migrated from a prior ``Literal[...]`` field MUST
serialize through Pydantic v2 to the same string. This regression test catches
two classes of subtle drift:

  1. ``StrEnum`` vs ``str + Enum`` divergence on 3.10/3.11
  2. ``model_dump_json`` emitting ``"AppEnv.DEV"`` instead of ``"dev"`` when the
     enum forgets the ``str`` mixin
  3. The ``BackendConfig`` discriminated union breaking when its structural
     ``Literal[...]`` discriminators get accidentally swapped to ``BackendKind``
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from deepCab.schemas.agent import AgentMessage
from deepCab.schemas.api import (
    ExplainRequest,
    TrainStatusResponse,
)
from deepCab.schemas.config import (
    CVConfig,
    DataRef,
    HPOConfig,
    TFMLPConfig,
    TorchMLPConfig,
    TrainConfig,
)
from deepCab.schemas.enums import (
    AppEnv,
    BackendKind,
    CVKind,
    DataSize,
    DataSource,
    ExplainMode,
    MessageRole,
    ModelTarget,
    OptunaDirection,
    OptunaPruner,
    OptunaSampler,
    RunStatus,
    SlackTag,
    Split,
)
from deepCab.schemas.settings import (
    DataSettings,
    RegistrySettings,
)

# ---------------------------------------------------------------------------
# Bare-enum invariants
# ---------------------------------------------------------------------------


ALL_ENUMS = [
    AppEnv,
    ModelTarget,
    DataSource,
    DataSize,
    Split,
    BackendKind,
    RunStatus,
    MessageRole,
    ExplainMode,
    CVKind,
    OptunaSampler,
    OptunaPruner,
    OptunaDirection,
    SlackTag,
]


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_enum_inherits_str(enum_cls) -> None:
    """`str + Enum` mixin — members must be `str` so JSON serialization is the
    raw value, not `"EnumName.MEMBER"`."""
    for member in enum_cls:
        assert isinstance(member, str)
        assert member == member.value


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_enum_member_equals_value_string(enum_cls) -> None:
    """`EnumMember == "value"` must hold so existing equality checks survive."""
    for member in enum_cls:
        assert member == member.value
        assert hash(member) == hash(member.value)


# ---------------------------------------------------------------------------
# Pydantic-model parity: JSON dump matches the prior Literal string
# ---------------------------------------------------------------------------


def test_explain_request_mode_json_parity() -> None:
    row = _row_payload()

    # String input still accepted (caller-facing contract).
    er = ExplainRequest.model_validate({"row": row, "mode": "summary"})
    assert er.mode == ExplainMode.SUMMARY
    assert er.mode == "summary"
    assert er.model_dump()["mode"] == "summary"
    assert json.loads(er.model_dump_json())["mode"] == "summary"

    # Enum-member input also accepted.
    er2 = ExplainRequest(row=er.row, mode=ExplainMode.PER_ROW)
    assert er2.model_dump()["mode"] == "per_row"

    # Default value unchanged.
    er3 = ExplainRequest(row=er.row)
    assert er3.mode == ExplainMode.PER_ROW
    assert er3.model_dump()["mode"] == "per_row"


def test_train_status_response_json_parity() -> None:
    ts = TrainStatusResponse.model_validate({"task_id": "abc", "status": "running"})
    assert ts.status == RunStatus.RUNNING
    assert ts.model_dump()["status"] == "running"
    assert json.loads(ts.model_dump_json())["status"] == "running"

    ts2 = TrainStatusResponse(task_id="x", status=RunStatus.SUCCEEDED)
    assert ts2.model_dump()["status"] == "succeeded"


def test_agent_message_role_json_parity() -> None:
    am = AgentMessage.model_validate({"role": "tool", "content": "result"})
    assert am.role == MessageRole.TOOL
    assert am.model_dump()["role"] == "tool"
    assert json.loads(am.model_dump_json())["role"] == "tool"

    am2 = AgentMessage(role=MessageRole.ASSISTANT, content="hi")
    assert am2.model_dump()["role"] == "assistant"


def test_data_ref_size_json_parity() -> None:
    dr = DataRef.model_validate({"size": "10k", "validation_size": "100k"})
    assert dr.size == DataSize.S10K
    assert dr.model_dump()["size"] == "10k"
    assert dr.model_dump()["validation_size"] == "100k"

    # Default ("1k") preserved.
    dr_default = DataRef()
    assert dr_default.model_dump()["size"] == "1k"

    # Newly added "full" value is accepted (additive).
    dr_full = DataRef(size=DataSize.FULL)
    assert dr_full.model_dump()["size"] == "full"


def test_cv_config_json_parity() -> None:
    cv = CVConfig.model_validate({"kind": "kfold", "n_splits": 3})
    assert cv.kind == CVKind.KFOLD
    assert cv.model_dump()["kind"] == "kfold"


def test_hpo_config_json_parity() -> None:
    hpo = HPOConfig.model_validate(
        {"n_trials": 5, "sampler": "cmaes", "pruner": "none", "direction": "maximize"}
    )
    assert hpo.sampler == OptunaSampler.CMAES
    assert hpo.pruner == OptunaPruner.NONE
    assert hpo.direction == OptunaDirection.MAXIMIZE
    payload = hpo.model_dump()
    assert payload["sampler"] == "cmaes"
    assert payload["pruner"] == "none"
    assert payload["direction"] == "maximize"


def test_data_settings_source_json_parity() -> None:
    ds = DataSettings()
    assert ds.source == DataSource.LOCAL
    assert ds.model_dump()["source"] == "local"

    ds2 = DataSettings(source="cloud")
    assert ds2.source == DataSource.CLOUD
    assert ds2.model_dump()["source"] == "cloud"


def test_registry_settings_target_json_parity() -> None:
    rs = RegistrySettings()
    assert rs.target == ModelTarget.LOCAL
    assert rs.model_dump()["target"] == "local"

    rs2 = RegistrySettings(target="gcs", bucket_name="x")
    assert rs2.target == ModelTarget.GCS
    assert rs2.model_dump()["target"] == "gcs"


# ---------------------------------------------------------------------------
# Negative cases — unknown strings still rejected
# ---------------------------------------------------------------------------


def test_enum_field_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        TrainStatusResponse.model_validate({"task_id": "x", "status": "weird"})


def test_data_ref_rejects_unknown_size() -> None:
    with pytest.raises(ValidationError):
        DataRef.model_validate({"size": "999k"})


# ---------------------------------------------------------------------------
# Discriminated-union regression — Lane A constraint: do NOT migrate the
# per-backend structural `Literal["tf_mlp"]` discriminators to BackendKind.
# This test wedges that contract in place.
# ---------------------------------------------------------------------------


def test_backend_discriminated_union_still_routes_by_kind_string() -> None:
    """`BackendConfig` is `Annotated[Union[...], Field(discriminator="kind")]`.
    The string `"tf_mlp"` must still select `TFMLPConfig`."""
    tc = TrainConfig.model_validate({"backend": {"kind": "tf_mlp"}})
    assert isinstance(tc.backend, TFMLPConfig)
    assert tc.backend.kind == "tf_mlp"

    tc2 = TrainConfig.model_validate({"backend": {"kind": "torch_mlp"}})
    assert isinstance(tc2.backend, TorchMLPConfig)
    assert tc2.backend.kind == "torch_mlp"

    # Dump round-trip — kind stays a plain string in the JSON payload.
    payload = tc.model_dump()
    assert payload["backend"]["kind"] == "tf_mlp"
    assert isinstance(payload["backend"]["kind"], str)

    # BackendKind enum value matches the discriminator string (proves the enum
    # is the right caller-facing alias even though the discriminator stays
    # Literal).
    assert BackendKind.TF_MLP == "tf_mlp"
    assert tc2.backend.kind == BackendKind.TORCH_MLP


# ---------------------------------------------------------------------------
# String-vs-enum input parity — both shapes accepted by every migrated field.
# ---------------------------------------------------------------------------


class _AppEnvHolder(BaseModel):
    """Minimal probe for `AppEnv` JSON round-trip on the composite root."""

    env: AppEnv = AppEnv.DEV


def test_app_env_both_input_shapes() -> None:
    a = _AppEnvHolder(env="prod")
    b = _AppEnvHolder(env=AppEnv.PROD)
    assert a.env == b.env == AppEnv.PROD
    assert a.model_dump()["env"] == "prod"
    assert json.loads(a.model_dump_json())["env"] == "prod"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_payload() -> dict:
    return {
        "pickup_datetime": "2014-01-01T08:00:00Z",
        "pickup_longitude": -73.98,
        "pickup_latitude": 40.75,
        "dropoff_longitude": -73.98,
        "dropoff_latitude": 40.75,
        "passenger_count": 1,
    }
