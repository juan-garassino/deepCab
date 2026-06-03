# Sub-project E — Full audit + simplify

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After sub-projects A–D land, run a full audit-and-simplify pass over the whole codebase. Capture baseline metrics, run automated analyzers, do a targeted manual review of six high-payoff areas, apply accepted simplifications, and write `infra/AUDIT.md` documenting every finding.

**Architecture:** Two phases. **Phase A** — automated baseline (radon, vulture, pylint duplicate-code, pydeps cycles). **Phase B** — manual review of `registry/dispatcher.py`, `schemas/settings.py`, `agent/{executor,improve,trace}.py`, `training/preprocess.py` vs `features/pipeline.py`, `api/routers/*.py`, test fixtures. All 103 + new tests must still pass. No new dependencies.

**Tech Stack:** `radon`, `vulture`, `pylint`, `pydeps`, `tokei` (optional), Python AST.

**Reference:** [Design spec §9](../specs/2026-06-03-deepcab-gcp-infra-and-audit-design.md#9-audit--simplify-sub-project-e).

**Prerequisite:** Sub-projects A + B + C + D landed.

---

## File map

| Action | Path | Purpose |
|---|---|---|
| Create | `infra/AUDIT.md` | Findings + actions + LOC/complexity delta |
| Create (temp) | `infra/audit/AUDIT.before.txt` | radon baseline |
| Create (temp) | `infra/audit/AUDIT.after.txt` | radon after simplifications |
| Create (temp) | `infra/audit/dead.txt` | vulture output |
| Create (temp) | `infra/audit/dup.txt` | pylint duplicate-code output |
| Create (temp) | `infra/audit/cycles.png` | pydeps cycles diagram (committed for the AUDIT.md to reference) |
| Modify | varied — only the files where accepted simplifications apply | per Phase B |
| Modify | `CLAUDE.md`, `README.md` | reference the AUDIT.md |

`infra/audit/` is committed so the audit is reproducible later; the markdown report at `infra/AUDIT.md` is the canonical artifact.

---

## Task E1: Install audit tools as dev deps

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add to `[project.optional-dependencies] dev`**

Inspect existing `dev` extra:

```bash
grep -A 20 "optional-dependencies" pyproject.toml | head -30
```

Append (idempotently — skip if already there):

```toml
dev = [
  # ... existing entries ...
  "radon>=6.0",
  "vulture>=2.13",
  "pylint>=3.3",
  "pydeps>=2.0",
]
```

- [ ] **Step 2: Re-sync**

```bash
uv sync --extra dev
```

- [ ] **Step 3: Sanity check**

```bash
uv run radon --version
uv run vulture --version
uv run pylint --version
uv run pydeps --version
```

Expected: all four print versions.

---

## Task E2: Phase A — capture baseline metrics

**Files:**
- Create: `infra/audit/AUDIT.before.txt`
- Create: `infra/audit/dead.txt`
- Create: `infra/audit/dup.txt`
- Create: `infra/audit/cycles.png` (may fail on systems without graphviz — record skip if so)

- [ ] **Step 1: Make audit dir**

```bash
mkdir -p infra/audit
```

- [ ] **Step 2: radon — cyclomatic complexity + maintainability index**

```bash
uv run radon cc -s -a deepCab/ > infra/audit/AUDIT.before.txt
echo "" >> infra/audit/AUDIT.before.txt
echo "=== Maintainability Index ===" >> infra/audit/AUDIT.before.txt
uv run radon mi -s deepCab/ >> infra/audit/AUDIT.before.txt
echo "" >> infra/audit/AUDIT.before.txt
echo "=== Raw metrics (LOC, LLOC, comments) ===" >> infra/audit/AUDIT.before.txt
uv run radon raw -s deepCab/ >> infra/audit/AUDIT.before.txt
```

- [ ] **Step 3: vulture — dead code**

```bash
uv run vulture deepCab/ tests/ --min-confidence 70 > infra/audit/dead.txt || true
```

(vulture exits non-zero when it finds things — `|| true` lets the pipeline continue.)

- [ ] **Step 4: pylint duplicate-code**

```bash
uv run pylint deepCab/ --disable=all --enable=duplicate-code \
  --min-similarity-lines=8 --output-format=text > infra/audit/dup.txt || true
```

- [ ] **Step 5: pydeps cycles (best-effort)**

```bash
uv run pydeps deepCab --max-bacon=2 --show-cycles -T png -o infra/audit/cycles.png 2>/dev/null \
  || echo "pydeps requires graphviz; record skip" > infra/audit/cycles.skip.txt
```

- [ ] **Step 6: Stage baseline artifacts**

```bash
git add infra/audit/
```

---

## Task E3: Phase A — triage automated findings

**Files:** none — produce a triage table in your head / scratchpad to drive Phase B; record final decisions in `infra/AUDIT.md` later.

- [ ] **Step 1: Read `AUDIT.before.txt`**

Scan for functions with complexity `C` (10–20) or `D` (20–30) or worse. List them.

Expected complexity hotspots (from prior knowledge of the repo): `agent/executor.py:_run_one_turn`, `training/train.py:run`, possibly `api/routers/predict.py:predict_stream`.

- [ ] **Step 2: Read `dead.txt`**

Filter out false positives (Pydantic field default factories, FastAPI dependency injection parameters that look unused, test fixtures auto-discovered by pytest). For each remaining candidate, note: **accept** (remove) or **reject** (with reason) or **defer**.

- [ ] **Step 3: Read `dup.txt`**

For each duplicate-code finding, decide: **collapse** (extract helper), **leave** (incidental similarity), **defer**.

- [ ] **Step 4: Note cycles**

If `cycles.png` was produced and shows any import cycles, document them. If `cycles.skip.txt` exists, note that pydeps could not run.

---

## Task E4: Phase B.1 — collapse legacy registry path

**Files:**
- Modify: `deepCab/registry/dispatcher.py`
- Modify: tests under `tests/registry/`

- [ ] **Step 1: Identify legacy path**

```bash
grep -n "save_artifact\|load_artifact" deepCab/registry/dispatcher.py
```

Expected: a temp-dir-based `save_artifact(estimator, ...) -> str` and `load_artifact(uri) -> estimator`, plus the new persistent `save_full_state` / `load_state_from_disk`.

- [ ] **Step 2: Find callers of the legacy functions**

```bash
grep -rn "save_artifact\|load_artifact" deepCab/ tests/ --include='*.py'
```

For each caller, decide: migrate to `save_full_state`/`load_state_from_disk` (the new API) or leave (if the caller is itself legacy and will be deleted).

- [ ] **Step 3: Migrate one caller at a time**

For each caller `<file>`, edit it to use the new persistent API. Run targeted tests after each migration:

```bash
uv run pytest tests/registry tests/training tests/api -q
```

Commit after each migration if the diff is big enough to be reviewable on its own (one batch is fine if small).

- [ ] **Step 4: Delete the legacy functions**

Once no callers remain, delete `save_artifact` and `load_artifact` from `dispatcher.py`.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -q --ignore=tests/all
```

Expected: still passing (count = whatever B+ added; should be ≥111).

---

## Task E5: Phase B.2 — kill `_maybe_read_file` if pydantic-settings can do it

**Files:**
- Modify: `deepCab/schemas/settings.py`

- [ ] **Step 1: Identify `_maybe_read_file`**

```bash
grep -n "_maybe_read_file\|def _read_file" deepCab/schemas/settings.py
```

- [ ] **Step 2: Check pydantic-settings native support**

The `file:` URI pattern is a pydantic-settings idiom but is NOT supported out of the box. Two options:

(a) Switch to **`SecretStr` + a custom source** that reads file URIs. Adds ~10 lines.
(b) Switch to **Docker secrets via the standard `DockerSecretsSettingsSource`** that pydantic-settings ships. Requires changing the env var names from `OBS_SLACK_WEBHOOK_URL=file:/run/secrets/X` to mounting the secret at `/run/secrets/obs_slack_webhook_url` and letting pydantic find it automatically.

(b) is more idiomatic; (a) is fewer compose-file changes.

**Default: pick (a)** — replace `_maybe_read_file` with a small `EnvSettingsSource` subclass that strips `file:` prefixes and reads the file. Cleaner than the current ad-hoc helper because the resolution happens at source level, not field level.

- [ ] **Step 3: Implement (a)**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict, EnvSettingsSource, PydanticBaseSettingsSource
from pydantic.fields import FieldInfo
from typing import Any
from pathlib import Path


class FileUriEnvSettingsSource(EnvSettingsSource):
    """Treat env values like `file:/path/to/secret` as references — read the file's contents."""
    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        value, key, complex = super().get_field_value(field, field_name)
        if isinstance(value, str) and value.startswith("file:"):
            path = Path(value.removeprefix("file:")).expanduser()
            if path.is_file():
                value = path.read_text().strip()
        return value, key, complex


class Settings(BaseSettings):
    # ... existing fields ...

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        return (
            init_settings,
            FileUriEnvSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )
```

Remove the old `_maybe_read_file` function and its `@field_validator(..., mode='before')` callers.

- [ ] **Step 4: Add a regression test**

`tests/schemas/test_file_uri.py`:

```python
import os
from pathlib import Path
from deepCab.schemas.settings import Settings


def test_file_uri_resolved_for_string_field(tmp_path: Path, monkeypatch):
    f = tmp_path / "webhook"
    f.write_text("https://hooks.slack.com/abc\n")
    monkeypatch.setenv("OBS_SLACK_WEBHOOK_URL", f"file:{f}")
    s = Settings()
    assert s.obs.slack_webhook_url == "https://hooks.slack.com/abc"


def test_plain_string_left_untouched(monkeypatch):
    monkeypatch.setenv("OBS_SLACK_WEBHOOK_URL", "https://hooks.slack.com/literal")
    s = Settings()
    assert s.obs.slack_webhook_url == "https://hooks.slack.com/literal"
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/schemas tests/api tests/obs -q
```

Expected: 2 new passing in `tests/schemas/test_file_uri.py`. Total goes up by 2.

---

## Task E6: Phase B.3 — agent trace audit

**Files:** read-only audit of `deepCab/agent/{executor,improve,trace}.py`. Modify only if findings warrant.

- [ ] **Step 1: Map trace write sites**

```bash
grep -n "trace\|append\|write" deepCab/agent/executor.py deepCab/agent/improve.py deepCab/agent/trace.py
```

For each `trace.append(...)` or `_append_event(...)` call, confirm it's the only writer for that event class. If you find two writers (e.g. executor logs `tool_call` and improve.py also logs `tool_call`), keep one.

- [ ] **Step 2: Confirm Budget.restore from trace**

Inspect `agent/budget.py:Budget.restore`. Add an assertion / test if missing:

```python
def test_budget_restore_from_trace(tmp_path):
    from deepCab.agent.budget import Budget
    from deepCab.agent.trace import write_event
    trace = tmp_path / "agent-x.jsonl"
    for usd, calls, iters in [(0.01, 1, 1), (0.02, 1, 1), (0.005, 1, 0)]:
        write_event(trace, {"kind": "tool_call_done", "usd": usd, "calls": calls, "iters": iters})
    b = Budget.restore(trace, cap_usd=1.0, max_tool_calls=10, max_iters=10)
    assert abs(b.spent_usd - 0.035) < 1e-9
    assert b.calls_made == 3
    assert b.iters_made == 2
```

Place in `tests/agent/test_budget_restore.py`.

- [ ] **Step 3: If duplicate writers exist, delete one**

If §B.3.1 found a duplicate writer, remove it. The canonical writer should be the one closest to the event source — e.g. inside the dispatch function rather than at the caller. Update the test for that path so it still asserts the event is written.

- [ ] **Step 4: Run agent tests**

```bash
uv run pytest tests/agent -q
```

---

## Task E7: Phase B.4 — clarify preprocess vs features/pipeline boundary

**Files:**
- Modify (potentially): `deepCab/training/preprocess.py`, `deepCab/features/pipeline.py`

- [ ] **Step 1: Diagram the current relationship**

```bash
grep -n "preprocess_features\|preprocess(\|featurize\|clean(" deepCab/training/preprocess.py deepCab/features/pipeline.py
```

Goal: confirm that `training/preprocess.py::preprocess` is the public entry that:
1. Loads data via `data/io.py`
2. Cleans via `data/validate.py`
3. Featurizes via `features/pipeline.py::preprocess_features`
4. Returns `(X, y)` numpy arrays

`features/pipeline.py::preprocess_features` should be a pure `pd.DataFrame -> (N, 65) numpy` transform with no I/O.

- [ ] **Step 2: If `training/preprocess.py` duplicates the encoder list or transforms**

Move them into `features/pipeline.py` or `features/transformers.py`. The training module imports them; it never redefines them.

- [ ] **Step 3: Add boundary doc**

Add a one-line docstring to each module clarifying its role:

```python
"""features/pipeline.py — pure feature engineering: pd.DataFrame -> (N, 65) numpy. No I/O."""
"""training/preprocess.py — orchestrates load + clean + featurize for training; calls features.pipeline."""
```

(Allowed — module-level docstrings are not the "obvious docstrings" we suppress.)

- [ ] **Step 4: Run training + features tests**

```bash
uv run pytest tests/training tests/features -q
```

---

## Task E8: Phase B.5 — keep routers thin

**Files:** read-only audit; modify each router file that has business logic creeping in.

- [ ] **Step 1: For each router file**

```bash
for f in deepCab/api/routers/*.py; do
  printf "=== %s ===\n" "$f"
  wc -l "$f"
done
```

Threshold: routers should be < 100 LOC each. If any is larger, that's a signal business logic snuck in.

- [ ] **Step 2: For routers > 100 LOC**

Extract the meaty function bodies into `deepCab/api/services/<name>.py`. The router function becomes:

```python
@router.post("/foo")
async def foo(req: FooReq, svc: FooService = Depends(get_foo_service)) -> FooResp:
    return await svc.handle(req)
```

Move all the logic, including any pandas/numpy work, into `FooService.handle`.

- [ ] **Step 3: Run API tests**

```bash
uv run pytest tests/api -q
```

---

## Task E9: Phase B.6 — test fixtures consolidation

**Files:**
- Modify: `tests/conftest.py` (root)
- Modify: per-module `conftest.py` files

- [ ] **Step 1: Find duplicated fixtures**

```bash
grep -rn "@pytest.fixture" tests/ --include='*.py' | awk -F'@pytest.fixture' '{print $2}' | sort | uniq -c | sort -rn | head -20
```

Look for fixtures defined in multiple places (e.g. `sample_predict_request` in both `tests/api/conftest.py` and `tests/agent/conftest.py`).

- [ ] **Step 2: Hoist common fixtures to root `tests/conftest.py`**

For each duplicate, write the fixture once at the root level. Delete the per-module copies.

- [ ] **Step 3: Parametrize collections**

If you see the same `@pytest.mark.parametrize("backend", ["xgb", "lgbm", "catboost", "tf_mlp", "torch_mlp", "ft_transformer"])` line repeated in several test files, hoist the backend list into `tests/conftest.py`:

```python
BACKENDS = ["xgb", "lgbm", "catboost", "tf_mlp", "torch_mlp", "ft_transformer"]
```

Import + use in each test file.

- [ ] **Step 4: Run full suite**

```bash
uv run pytest tests/ -q --ignore=tests/all
```

Expected: same number of passing tests.

---

## Task E10: Capture after-metrics

**Files:**
- Create: `infra/audit/AUDIT.after.txt`

- [ ] **Step 1: Re-run radon**

```bash
uv run radon cc -s -a deepCab/ > infra/audit/AUDIT.after.txt
echo "" >> infra/audit/AUDIT.after.txt
echo "=== Maintainability Index ===" >> infra/audit/AUDIT.after.txt
uv run radon mi -s deepCab/ >> infra/audit/AUDIT.after.txt
echo "" >> infra/audit/AUDIT.after.txt
echo "=== Raw metrics ===" >> infra/audit/AUDIT.after.txt
uv run radon raw -s deepCab/ >> infra/audit/AUDIT.after.txt
```

- [ ] **Step 2: Compute LOC delta**

```bash
B=$(grep -E "^.* +- " infra/audit/AUDIT.before.txt | grep "LOC:" | tail -1)
A=$(grep -E "^.* +- " infra/audit/AUDIT.after.txt  | grep "LOC:" | tail -1)
echo "before: $B"
echo "after:  $A"
```

(Record both in the AUDIT.md.)

---

## Task E11: Write `infra/AUDIT.md`

**Files:**
- Create: `infra/AUDIT.md`

- [ ] **Step 1: Compose the report**

```markdown
# AUDIT — 2026-06-04

Sub-project E of the 2026-06-03 GCP infra design.

## Phase A — automated baseline

| Tool | Output | Findings (raw) |
|---|---|---|
| radon (CC + MI + raw) | `infra/audit/AUDIT.before.txt` | <N> functions ≥ C complexity |
| vulture (dead code) | `infra/audit/dead.txt` | <N> candidates, <M> after filtering |
| pylint (duplicate-code) | `infra/audit/dup.txt` | <N> duplicate blocks |
| pydeps (cycles) | `infra/audit/cycles.png` or `cycles.skip.txt` | <0 or N> cycles |

## Phase B — manual triage + actions

### B.1 — Registry dispatcher legacy collapse

| Finding | Action | Result |
|---|---|---|
| `save_artifact` + `load_artifact` (legacy temp-dir paths) coexisted with `save_full_state` / `load_state_from_disk` | Migrated <N> callers, deleted legacy functions | -<X> LOC, no test regression |

### B.2 — schemas/settings _maybe_read_file

| Finding | Action | Result |
|---|---|---|
| Field-level `file:` URI resolver duplicated for every secret field | Replaced with `FileUriEnvSettingsSource` (settings source-level resolution) | -<X> LOC, +2 regression tests |

### B.3 — Agent trace writers

| Finding | Action |
|---|---|
| <findings from §E6> | <accept/reject/defer with reason> |

### B.4 — Preprocess vs features/pipeline

| Finding | Action |
|---|---|
| <findings from §E7> | <accept/reject/defer with reason> |

### B.5 — Router thinness

| Router | LOC before | LOC after | Extracted service |
|---|---|---|---|
| `predict.py` | <N> | <M> | <yes/no> |
| `explain.py` | <N> | <M> | <yes/no> |
| `train.py`   | <N> | <M> | <yes/no> |
| `agent.py`   | <N> | <M> | <yes/no> |
| `meta.py`    | <N> | <M> | n/a |
| `monitor.py` | <N> | <M> | n/a |

### B.6 — Fixture consolidation

| Fixture | Was in | Now in |
|---|---|---|
| `sample_predict_request` | <files> | `tests/conftest.py` |
| `BACKENDS` constant | scattered | `tests/conftest.py` |

## Rejected findings (with reasons)

- <finding 1> — kept because <reason>.
- <finding 2> — kept because <reason>.

## Deferred findings

- <finding> — left as a follow-up issue: <link>.

## Delta

| Metric | Before | After | Δ |
|---|---|---|---|
| LOC (deepCab/) | <N> | <M> | <signed> |
| Avg cyclomatic complexity | <X> | <Y> | <signed> |
| Tests passing | <N> | <M> | <signed; expected ≥ 0> |
| New dependencies | — | 0 | 0 |

## Acceptance check

- [x] All tests pass (count = <N>; was <M> before sub-project A).
- [x] No new runtime dependencies.
- [x] Cyclomatic complexity did not regress on average.
- [x] `infra/AUDIT.md` lists every finding with an action.
```

Fill in the `<...>` placeholders with the real numbers from your audit run.

- [ ] **Step 2: Sanity check**

```bash
grep -n "<" infra/AUDIT.md
```

Expected: no remaining `<...>` placeholders.

---

## Task E12: Update top-level docs

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: CLAUDE.md — add Audit subsection**

After the "CI/CD" subsection from Sub-project C:

```markdown
## Audit

Cumulative audit findings live in `infra/AUDIT.md`. Raw analyzer output is in `infra/audit/` (radon before/after, vulture dead-code, pylint duplicate-code, pydeps cycles). Re-run with: `uv sync --extra dev && uv run radon cc -s -a deepCab/`.
```

- [ ] **Step 2: README.md — final sub-section**

Append:

```markdown
## Audit

See `infra/AUDIT.md` for the latest simplification pass — LOC delta, complexity delta, dead code removed, fixtures consolidated.
```

---

## Task E13: Commit Sub-project E

- [ ] **Step 1: Commit**

```bash
git add -A
git status --short
git commit -m "$(cat <<'EOF'
chore(audit): full code audit + targeted simplifications; AUDIT.md

Phase A — automated baseline:
- radon CC/MI/raw → infra/audit/AUDIT.{before,after}.txt
- vulture --min-confidence 70 → infra/audit/dead.txt
- pylint --enable=duplicate-code → infra/audit/dup.txt
- pydeps --show-cycles → infra/audit/cycles.png (best-effort)

Phase B — targeted manual simplifications:
- registry/dispatcher.py: collapsed legacy save_artifact/load_artifact
  into the persistent save_full_state path; deleted legacy functions
- schemas/settings.py: replaced _maybe_read_file with
  FileUriEnvSettingsSource (settings-source level resolution)
- agent/{executor,improve,trace}.py: confirmed single trace writer
- training/preprocess.py vs features/pipeline.py: clarified boundary
- api/routers/*.py: extracted services where routers exceeded 100 LOC
- tests: hoisted duplicated fixtures to tests/conftest.py

infra/AUDIT.md documents every finding with action + LOC + complexity delta.

All tests pass. No new runtime dependencies.

Sub-project E of the GCP infra design.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Done criteria

- [x] `radon` baseline + after captured under `infra/audit/`
- [x] `vulture`, `pylint --enable=duplicate-code`, `pydeps` run with results captured
- [x] B.1: legacy registry path collapsed
- [x] B.2: `_maybe_read_file` replaced with settings source
- [x] B.3: agent trace writers confirmed single-source
- [x] B.4: preprocess ↔ features/pipeline boundary documented
- [x] B.5: every router < 100 LOC; logic in services
- [x] B.6: duplicate fixtures hoisted to root conftest
- [x] `infra/AUDIT.md` exists with no `<...>` placeholders
- [x] All tests pass (count ≥ baseline)
- [x] No new runtime dependencies
- [x] One commit
