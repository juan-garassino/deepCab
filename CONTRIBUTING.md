# Contributing — deepCab

## Setup

```bash
make actions_reinstall            # uv sync --extra dev
make actions_reinstall_silicon    # mac arm64
cp .env.dev.sample .env.dev       # edit secrets values
```

## Test loop

```bash
uv run pytest tests/ -q
uv run ruff format --check .
uv run ruff check .
uv run mypy deepCab --ignore-missing-imports --check-untyped-defs   # informational
```

CI runs the same three commands plus `compose up` smoke. Pre-commit hooks
(`uv run pre-commit install`) lint+format every commit.

## Runbook — common snags

### Prefect 401 "Invalid authentication credentials"

Cause: a stale `PREFECT_API_KEY` from `prefect cloud login` is in your shell.
Fix:

```bash
unset PREFECT_API_KEY
unset PREFECT_API_URL
```

In compose this never happens (the agent service points at `prefect:4200`).

### macOS: `xgboost.core.XGBoostError: libxgboost.dylib could not be loaded`

`libomp` ABI mismatch with the xgboost wheel. Fix:

```bash
brew install libomp
brew reinstall libomp@11   # if the system libomp is too new
```

The Docker image (`python:3.11-slim` + `tensorflow/tensorflow` base) doesn't hit this.

### Encoder math drift — golden vectors fail

If you change `features/transformers.py` and `test_golden.py` starts failing
intentionally, regenerate the reference vectors:

```bash
uv run python -c "
from deepCab.features.golden import as_dataframe, reference_time_features, reference_lonlat_features
import numpy as np
df = as_dataframe()
np.save('tests/features/_golden_time.npy', reference_time_features(df))
np.save('tests/features/_golden_lonlat.npy', reference_lonlat_features(df).to_numpy())
"
```

Then update the asserted values in `tests/features/test_golden.py`. Commit the
intent in the message — golden vectors are the **canonical** record of what
the math produces.

### Parquet location / `IsADirectoryError` on first `make run_train`

The Hydra entry expects Parquet at `~/.lewagon/mlops/data/parquet/` (override
via `DATA_PARQUET_PATH`). To seed from the legacy CSVs:

```bash
python -m deepCab.data.migrate --size 1k --split train
python -m deepCab.data.migrate --size 1k --split val
```

### "docker compose can't find docker-compose.yml at root"

The compose files moved under `infra/compose/` in Sub-project A (2026-06-03). Run `make docker_up` instead of bare `docker compose up`, or pass `-f infra/compose/docker-compose.yml` explicitly.

### `/predict` returns 503 after API restart

**Shouldn't happen after FR-1.** The lifespan autoloader reads
`<REGISTRY_LOCAL_PATH>/runs/LATEST` and rehydrates `STATE.model` from the most
recent `make run_train` or agent `train` call.

If it does, check in order:

1. Have you trained at all? `cat $(python -c 'from deepCab.schemas.settings import get_settings; print(get_settings().registry.local_path / "runs/LATEST")')` should print a run_id.
2. Is the run dir intact? Expect `model/`, `cfg.json`, `background.npy`, and (when ACI calibrated) `aci.json`.
3. Backend mismatch? `cfg.json`'s `backend_kind` must appear in `deepCab/models/_kinds.py::BACKENDS`.

MLflow `@champion` → STATE rehydration is still post-MVP: lifespan logs
`api.champion.found` when an alias exists but doesn't yet load by MLflow URI.
The local-LATEST path covers the "I just trained, restart the API" case.

## Adding a new backend

1. **Pydantic config**: append a `<Backend>Config(BackendBase)` with `kind: Literal["..."]` in `deepCab/schemas/config.py`. Add it to the `BackendConfig` union.
2. **Estimator class**: subclass `AbstractEstimator` in `deepCab/models/<backend>.py`. Implement `_fit`, `_predict`, `save`, `load`. Lazy-import the framework so the module is importable without the dep.
3. **Register**: one line in `deepCab/models/_kinds.py::BACKENDS`.
4. **ONNX export** (optional): add a branch in `deepCab/models/onnx_export.py::export_to_onnx`.
5. **Search space**: one entry in `deepCab/training/hpo.py::SPACES`.
6. **Test**: `tests/models/test_backend_zoo.py` parametrizes on `BackendConfig`; the new backend gets exercised automatically (gated by importability).

## Adding a new agent tool

1. Define `MyToolIn(_ToolIn)` and `MyToolOut(_ToolOut)` Pydantic models in `deepCab/agent/tools.py`.
2. Write `_my_tool(args: MyToolIn) -> MyToolOut` — thin shim around an existing pure function.
3. Append `(MyToolIn, _my_tool, "description")` to `_TOOLS`.

OpenAI tool schema + FastAPI body materialize automatically — no extra wiring.

## Adding a new FastAPI endpoint

1. Drop a router under `deepCab/api/routers/<name>.py`.
2. `include_router` in `deepCab/api/app.py::create_app`.
3. Request/response Pydantic models go in `deepCab/schemas/api.py` (so they're discoverable from agent tools).

## Hot-loops to avoid

- **Don't pass `DictConfig` to `Pydantic.model_validate`** — use `OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)` first. Discriminator fields come through as `ValueNode` and unions fail.
- **Don't re-export `preprocess` or `evaluate` from `deepCab/training/__init__.py`** — they shadow the submodule names and break `monkeypatch.setattr("deepCab.training.preprocess.load", ...)`.
- **Don't `torch.load(..., weights_only=False)`** — closes a CVE-class issue. Use the structured state-dict + JSON sidecar pattern from `models/torch_mlp.py::save`.
- **Don't put `OBS_CORS_ALLOW_ORIGINS="*"` with `APP_ENV=prod`** — `create_app` will raise. Use an explicit allowlist in production.
