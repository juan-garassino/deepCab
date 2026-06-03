# 2026-06-03 deepCab GCP infra + audit — plan index

Five sub-projects, executed in order. Each = one PR / one commit. Spec: `docs/superpowers/specs/2026-06-03-deepcab-gcp-infra-and-audit-design.md`.

| # | Plan | Goal | Done when |
|---|---|---|---|
| A | `2026-06-03-A-infra-reorganization.md` | Move Dockerfile + 3 compose files + compose/conf + secrets.example/ under `infra/`. Update Makefile + docs. `.github/workflows/` stays at repo root. | Compose syntax valid, 103 tests pass, one commit |
| B | `2026-06-03-B-real-service-additions.md` | Add Traefik (reverse proxy + mkcert TLS), Loki + Promtail (logs), Alertmanager (Slack), pgAdmin (dev), ngrok (dev). Add `obs/slack.py` helper used by `registry.set_alias` and `flow_v2.retrain`. | 18-container stack healthy via `make docker_obs_up`, Slack test alert delivered, 111 tests pass |
| C | `2026-06-03-C-gcp-cicd.md` | GCP CI/CD with Workload Identity Federation: build-and-push.yml (reusable), deploy-cloud-run.yml (default, tag-triggered), deploy-gke.yml (dispatch-only). Slack via `rtcamp/action-slack-notify@v2.3.3`. | Workflows lint OK, WIF README documents one-time bootstrap, GitHub vars/secret list documented |
| D | `2026-06-03-D-react-and-colab.md` | Replace Streamlit at `003-deepCab-website/` with Vite+React+TS+Tailwind SPA (Predict/Explain/Runs). Add Colab notebook bridging VS Code via ngrok kernel + GCS push + trigger deploy. | `npm run build` succeeds, `app.deepcab.localhost` renders, 7-cell ipynb valid JSON |
| E | `2026-06-03-E-audit-and-simplify.md` | Phase A automated metrics (radon + vulture + pylint + pydeps), Phase B manual simplifications across 6 areas, `infra/AUDIT.md`. No new deps. | AUDIT.md has no `<...>` placeholders, tests pass, LOC/complexity delta recorded |
| F | `2026-06-03-F-cloud-training-trigger.md` | Cloud Scheduler → Cloud Run Job (default training trigger) + Prefect Cloud + Cloud Run worker documented as alternative. Adds `REGISTRY_GCS_BUCKET` env-driven artifact push to `training/train.py`. | `gh workflow run deploy-retrain-job.yml -f tag=v0.1.0` builds + pushes + replaces the Job; `make scheduler_bootstrap` wires Scheduler |

## Execution dependencies

```
A → B → C → D → E
        ↓
        F (after C lands; can run in parallel with D)
```

For simplicity execute sequentially: A → B → C → F → D → E. F slots after C because it reuses C's WIF + GAR setup; before D because D's notebook (cell 6) can choose to fire either `deploy-cloud-run.yml` (API) or `deploy-retrain-job.yml` (Job). Each plan ends with its own commit.

## Test count expectations

| After | Tests passing |
|---|---|
| Plan A | 103 (unchanged) |
| Plan B | 111 (+8 from obs/slack helpers) |
| Plan C | 111 (no Python changes) |
| Plan D | 111 (frontend + notebook, no Python tests) |
| Plan E | 113+ (+2 from B.2 file-URI tests) |

## Execution mode

Two options (see writing-plans skill handoff):

- **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks. Tighter context per task, slower but safer.
- **Inline Execution** — execute tasks in-session via `superpowers:executing-plans`, batched checkpoints.

For a 5-plan sequence this size, subagent-driven is the safer default; the harness can review each subagent's diff before moving on.
