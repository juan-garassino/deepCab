# deepCab v1 polish — services, providers, commands, Pydantic v2, enums

**Date:** 2026-06-04
**Status:** In-progress (3 lanes dispatched in parallel)
**Owner:** Juan Garassino
**Predecessor:** v1 shipped earlier today (commits `82b292d` → `d7d575b`); 116 tests passing.

## 1. Goal

Tighten the deepCab codebase along five axes the user called out: **services, providers, commands, Pydantic, enums**. Move scattered business logic out of routers into a real services layer. Make provider strategies (Slack, model registry, trace) explicit and injectable. Give the package a real unified CLI. Audit Pydantic v2 idioms. Replace `Literal[...]` magic strings with proper enums.

Net effect: same behavior, sharper structure. **No public-surface breakage** (HTTP contracts, env var names, module entrypoints all preserved).

## 2. Lane partition (parallel-safe by file ownership)

### Lane A — Schemas (Enums + Pydantic v2 idioms)

**Owns**: `deepCab/schemas/*.py` exclusively. May read consumers to verify imports, but the only consumer edits allowed are renaming the inline `Literal[...]` annotations in a file the lane owns to use the new enum import — no business logic changes.

**New file**:
- `deepCab/schemas/enums.py` — `class FooEnum(str, Enum): VALUE = "value"` style. All enums inherit from `str` so JSON serialization stays identical to the prior Literal-string. Python's `str + Enum` is the right idiom (avoids the 3.11-only `StrEnum` import).

**Enums in scope** (replace where used; do NOT replace the structural discriminator `kind: Literal["tf_mlp"]` etc. in the backend config subclasses — those are type-level, not values):

| Enum | Values | Source `Literal[...]` |
|---|---|---|
| `AppEnv` | `DEV`, `STAGING`, `PROD` | `settings.py:161` |
| `ModelTarget` | `LOCAL`, `GCS`, `MLFLOW` | `settings.py:87` |
| `DataSource` | `LOCAL`, `QUERY`, `CLOUD` | `settings.py:76` |
| `DataSize` | `S1K`, `S10K`, `S100K`, `S500K`, `FULL` | `config.py:89,90` (values: `"1k"`, `"10k"`, `"100k"`, `"500k"`; add `"full"` per F's spec) |
| `Split` | `TRAIN`, `VAL` | (used in data/io.py + features/pipeline) |
| `BackendKind` | `TF_MLP`, `TORCH_MLP`, `XGB`, `LGBM`, `CATBOOST`, `FT_TRANSFORMER` | union of `config.py` discriminator values (additive — keeps the per-class `Literal[...]` discriminators intact) |
| `RunStatus` | `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED` | `api.py:71` |
| `MessageRole` | `SYSTEM`, `USER`, `ASSISTANT`, `TOOL` | `agent.py:34` |
| `ExplainMode` | `PER_ROW`, `SUMMARY` | `api.py:44` |
| `CVKind` | `TIMESERIES`, `KFOLD` | `config.py:96` |
| `OptunaSampler` | `TPE`, `CMAES`, `RANDOM` | `config.py:104` |
| `OptunaPruner` | `MEDIAN`, `HYPERBAND`, `NONE` | `config.py:105` |
| `OptunaDirection` | `MINIMIZE`, `MAXIMIZE` | `config.py:106` |
| `SlackTag` | `CI`, `FLOW`, `ALERT`, `MLFLOW` | (used in `obs/slack.py` post() argument) |

**Pydantic v2 idioms to apply** (consistently):
- Replace any `class Config:` blocks with `model_config = ConfigDict(...)`.
- Replace `.dict()` → `.model_dump()`, `.parse_obj(x)` → `.model_validate(x)`, `.json()` → `.model_dump_json()`.
- Where a field has `Field(...)` + a type, prefer `Annotated[<type>, Field(...)]` (Pydantic-v2 recommended).
- Verify discriminated unions on `BackendConfig` already use `Field(discriminator="kind")` (it does, per spec); add the same pattern to any other union we have.

**Out of scope for Lane A**:
- Touching `pyproject.toml`, `Makefile`, `tests/conftest.py`, any router/service/CLI file.
- Adding new dependencies.

### Lane B — Services + Providers

**Owns**: NEW `deepCab/api/services/*.py`, NEW `deepCab/api/providers.py`. Allowed to edit `deepCab/api/routers/*.py`, `deepCab/api/deps.py`, `deepCab/api/state.py`. No edits to schemas or CLI.

**New files**:
- `api/services/__init__.py`
- `api/services/predict.py` — `class PredictionService` with `predict_one(req, ctx) -> resp`, `predict_many(req, ctx) -> resp`, `predict_stream(req, ctx) -> AsyncIterator[bytes]`. Pulls all the inference + Doppler logic out of `routers/predict.py`.
- `api/services/explain.py` — `class ExplanationService` with `explain_row()`, `summary()`.
- `api/services/train.py` — `class TrainingService` with `start(cfg) -> TrainStartResponse`, `status(id) -> TrainStatusResponse`. Owns the background-task plumbing currently in routers/train.py.
- `api/services/agent.py` — `class AgentService` with `run_turn(messages, ctx) -> AsyncIterator[Event]`. Owns the OpenAI tool-call loop for the SSE endpoint.

- `api/providers.py` — Strategy abstractions:
  - `class ModelHandleProvider(Protocol)` + `LocalHandleProvider`, `GCSHandleProvider`, `MLflowHandleProvider`. Already loosely modeled in `registry/dispatcher.py`; this is the clean injectable form.
  - `class SlackProvider(Protocol)` + `WebhookSlackProvider`, `NoopSlackProvider`. The latter is the default in tests (avoids real network calls).
  - `class TraceProvider(Protocol)` + `JsonlTraceProvider`, `NullTraceProvider`.

**Refactor pattern** (apply to each router):

```python
# routers/predict.py — BEFORE: 107 LOC, business logic inline
@router.post("/predict")
async def predict(req: PredictRequest, request: Request) -> PredictResponse:
    # 80 lines of feature building, ONNX call, doppler, etc.
    ...

# routers/predict.py — AFTER: ~25 LOC, thin adapter
@router.post("/predict", response_model=PredictResponse)
async def predict(
    req: PredictRequest,
    svc: PredictionService = Depends(get_prediction_service),
) -> PredictResponse:
    return await svc.predict_one(req)
```

**`api/deps.py` becomes a factory tray**:
- `get_settings()` → `Settings` (cached)
- `get_model_handle()` → `ModelHandle` (existing)
- `get_slack_provider()` → `SlackProvider`
- `get_prediction_service()` → composes model handle + slack into a `PredictionService`
- `get_explanation_service()` → ...
- `get_training_service()` → ...
- `get_agent_service()` → ...
- `api_key_guard()` → existing

Tests override providers via `app.dependency_overrides[get_slack_provider] = lambda: NoopSlackProvider()`.

**Out of scope**:
- Changing HTTP shapes, URLs, or methods.
- Touching `pyproject.toml`, `Makefile`, `tests/conftest.py`, `schemas/`, `cli/`.
- Adding new runtime deps.

### Lane C — CLI / Commands

**Owns**: NEW `deepCab/cli/*.py`, NEW `deepCab/__main__.py`. Allowed to edit `pyproject.toml` (add `[project.scripts]` + `typer` dep), `Makefile` (new `cli_*` shortcuts). Allowed to add tests under `tests/cli/`.

**New files**:
- `cli/__init__.py` — `app = typer.Typer(name="deepcab", help="deepCab MLOps CLI")`
- `cli/train.py` — `@app.command()` wrapping `training/train.py::run`. Accepts Hydra-style overrides via `typer.Argument(list[str])` and composes the cfg programmatically.
- `cli/predict.py` — `@app.command()` reading a JSON row from stdin or `--input file.json`, calling `training/predict.py::predict_one`.
- `cli/migrate.py` — `@app.command()` wrapping `data/migrate.py` CLI args (`--size`, `--split`).
- `cli/agent.py` — `@app.command()` starting the existing agent REPL (`agent/cli.py`'s logic).
- `cli/serve.py` — `@app.command()` invoking `uvicorn deepCab.api.app:create_app --factory` with sane defaults.
- `cli/status.py` — `@app.command()` printing settings + recent runs + health.

**Entry**:
- `deepCab/__main__.py` — `from deepCab.cli import app; app()` (so `python -m deepCab ...` works).
- `pyproject.toml` adds:
  ```toml
  [project.scripts]
  deepcab = "deepCab.cli:app"
  ```

**Dep**:
- Add `typer>=0.12.0` to the main `[project.dependencies]` list (it's small, pure-Python, no transitive surprises).

**Old entrypoints remain functional**:
- `python -m deepCab.training.train backend=tf_mlp data=1k` — still works (Hydra)
- `python -m deepCab.agent.cli` — still works (existing REPL)
- `python -m deepCab.data.migrate --size 1k --split train` — still works
- Makefile `run_train`, `run_api`, etc. — unchanged

**Tests**: `tests/cli/test_*.py` using `from typer.testing import CliRunner`. Each command gets a smoke test that asserts exit code 0 with `--help`.

**Out of scope**:
- Touching schemas, routers, services, providers.
- Removing existing entrypoints.

## 3. Cross-lane contract

- Each lane is implementable as a single commit on top of `d7d575b`.
- Public surfaces preserved: HTTP response shapes, env var names, `python -m deepCab.X` entrypoints, Makefile `run_*` targets.
- Each lane's enum/service/CLI addition is consumed via existing names — Lane B importing `RunStatus` from `schemas/api.py` keeps working because Lane A re-exports it from the same module.
- Tests stay green at 116+. Each lane adds its own tests:
  - Lane A: 4–6 enum round-trip tests
  - Lane B: ~12 service-level unit tests (mocked providers)
  - Lane C: ~6 CLI smoke tests via CliRunner
  - Target: 138+ tests passing after integration.

## 4. Integration

After all three commits land:
1. `uv sync --extra dev` (Lane C may have updated deps)
2. `uv run pytest tests/ -q --ignore=tests/all` — target 138+ passing
3. `uv run deepcab --help` should list 7 commands
4. `uv run deepcab status` should print env + recent runs without crashing
5. `curl -k https://api.deepcab.localhost/predict` (if compose stack up) should still return the same response shape
6. Update CLAUDE.md sections (Architecture, Common commands) to reference services/providers/cli

## 5. Risks

| Risk | Mitigation |
|---|---|
| Enum serialization changes JSON | All enums inherit from `str`; JSON output stays identical |
| Lane A imports cycle with Lane B routers | Enums live in `schemas/enums.py` — lowest-level module, no circular risk |
| Typer adds a heavy transitive tree | Typer is ~50KB, pure-Python, depends only on `click` + `rich` (rich is already a transitive dep of `structlog`/`pretty exceptions`) |
| Background train task semantics changed by extracting service | Lane B keeps the existing `BackgroundTasks` pattern; only relocates it |
| Hydra-style overrides break in Typer | `deepcab train` accepts a `list[str]` of overrides and composes via `OmegaConf.from_dotlist`; the existing Hydra entry stays for power users |

## 6. Done criteria

- `infra/AUDIT.md` gets a short addendum section "v1 polish — 2026-06-04" listing the 3 lanes + LOC delta
- 138+ tests passing
- `uv run deepcab --help` works
- No new runtime dependencies beyond `typer`
- CLAUDE.md (parent) and README.md updated
- Three commits, one per lane
