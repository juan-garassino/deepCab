# deepCab simplification audit — 5-lane real LOC reduction pass

**Date:** 2026-06-04 (post-polish)
**Status:** In-progress (5 parallel lanes dispatched)
**Predecessor:** Polish pass landed at `761d2e6` (Lanes A/B/C). 178 tests passing.

## 1. Goal

A REAL LOC-reduction simplification pass over the entire `deepCab/` source tree. The polish pass added ~+930 LOC (services, providers, CLI, enums) in exchange for better structure. This pass goes the other way: **find code that doesn't earn its place and delete it**. Target: net negative LOC. No public-surface breakage.

Plan E's audit was conservative (-2 LOC). This is a more aggressive, file-by-file scrutiny.

## 2. What counts as "simplification"

Concrete signals each lane should hunt for:

1. **Dead code** — unused imports (`from x import y` where `y` is never used), unused functions, unused classes, unreachable branches
2. **Single-use abstractions** — a wrapper / Protocol / ABC / dataclass with ONE caller or ONE implementation. Inline it.
3. **Trivial wrappers** — `def foo(x): return bar(x)` with no value-add. Inline.
4. **Speculative abstractions** — a Protocol with one concrete impl, an ABC with one subclass, a strategy pattern that has only one strategy. Collapse.
5. **Backwards-compat shims** — `_old_name = new_name` aliases, re-exports of removed things, "removed" comments
6. **Commented-out code** — delete
7. **Redundant comments / docstrings** — comments that just paraphrase the next line; one-line docstrings on obvious functions (`def add(a, b): return a + b` with `"""Add two numbers."""`)
8. **`print()` debug statements** in non-CLI source files
9. **Stale TODO / FIXME / XXX** — assess and either fix, delete, or convert to an issue link
10. **Try/except that catches everything and just logs** — assess whether the suppression is correct; if not, narrow
11. **`if __name__ == "__main__"` blocks** that duplicate what the CLI/Hydra entry already does
12. **Duplicated logic** — two slightly different impls of the same thing; pick one
13. **Pydantic models with no extra validation that just wrap a single field** — use the field directly
14. **Cyclomatic complexity hotspots** — flagged in `infra/audit/AUDIT.before.txt`; look for obvious early returns or split-by-case opportunities

## 3. What NOT to delete (preserve list)

Things the user has expressed as intentional (do NOT collapse):

- **6 backends + factory + plain-dict registry (`models/_kinds.py`)** — explicit learning surface; even if some look duplicated
- **Hand-rolled OpenAI agent loop in `agent/`** — CLAUDE.md mandate: "no high-level orchestrators, no LangChain"
- **`017-sklearn-low-level` patterns** — BaseEstimator wrapping, plain-dict BACKENDS registry, spec-as-factory CV — these are intentional
- **Hydra training entry** (`training/train.py`'s `@hydra.main`) — intentional learning surface
- **Provider Protocols added by Lane B** — `SlackProvider`, `ModelHandleProvider`, `TraceProvider` — these have multiple impls (Webhook/Noop, State/Stub, Jsonl/Null); they ARE legitimate strategies
- **Enums from Lane A** — `schemas/enums.py` — keep all 14 even if some only have 1 consumer today (they're meant to grow)
- **Tests** — never delete a test (you may rewrite/consolidate fixtures)
- **TF + PyTorch dual implementation** — duplicates are intentional for the learning project
- **Pydantic schemas at boundaries** — `schemas/{api,config,data,registry,settings,agent}.py` are the contract

You may, however:
- Inline trivial wrappers WITHIN preserved subsystems
- Delete unused imports + dead branches WITHIN preserved subsystems
- Consolidate one-line docstrings, comments
- Replace `dict[str, Any]` returns with proper response models where unambiguous

## 4. Hard constraints

- **All 178 existing tests must still pass.** Each lane runs the full suite before committing.
- **No public-surface breakage**: HTTP endpoints, env var names, CLI commands (`deepcab *`), legacy `python -m deepCab.*` entrypoints, Makefile targets — all unchanged.
- **No new runtime dependencies.** No new tools added.
- **Commits include `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`**
- **No `--no-verify`**
- **Cross-package imports**: BEFORE deleting any function/class from your lane, `grep -rn "from deepCab.<your-pkg> import <name>" deepCab/ tests/` — if it's imported elsewhere, leave it OR migrate the importer too (only if the importer is also in your lane).
- **Don't touch `tests/conftest.py`, `pyproject.toml`, `Makefile`, `.github/`, `infra/`**. These are out of scope.

## 5. Lane partition

| Lane | Owned subtrees | LOC | Tests |
|---|---|---|---|
| **S1 — Foundations** | `deepCab/schemas/` + `deepCab/obs/` | ~1050 | `tests/schemas/`, `tests/obs/` |
| **S2 — Data pipeline + registry + serving + flow + explain** | `deepCab/{data,features,registry,serving,flow_v2,explain}/` | ~1485 | `tests/data/`, `tests/features/`, `tests/registry/`, `tests/serving/`, `tests/flow_v2/`, `tests/explain/` |
| **S3 — Models & Training** | `deepCab/{models,training}/` | ~1845 | `tests/models/`, `tests/training/` |
| **S4 — API + gRPC** | `deepCab/{api,grpc}/` + `deepCab/__init__.py` + `deepCab/__main__.py` | ~1680 | `tests/api/` |
| **S5 — Agent + CLI** | `deepCab/{agent,cli}/` | ~1535 | `tests/agent/`, `tests/cli/`, `tests/security/` |

Lanes are file-disjoint. Cross-package callers preserved; agents grep before deleting.

## 6. Done criteria

- Each lane reports **LOC delta** (negative is the goal).
- All 178 tests still pass (+ any tests the agent ADDED to verify a simplification doesn't break behavior).
- One commit per lane.
- An addendum to `infra/AUDIT.md` (handled by the integrator after all lanes land) summarizing what each lane deleted, with the running total.

## 7. Acceptance — what "good" looks like

A great lane report:
- Concrete deletions with file:line references
- Justification per deletion ("unused import" / "single-call wrapper inlined" / "stale TODO from 2025")
- Things considered + rejected with reasons
- Net LOC delta in the lane's owned files

A bad lane report:
- "Cleaned up some things"
- "Improved readability"
- No LOC numbers
- Mass-deletes things without verifying callers

If a lane finds NOTHING worth deleting (lane is already lean), the right answer is "no changes; lane is clean" + a short justification, NOT "I deleted a docstring to claim a win."
