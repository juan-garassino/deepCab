# deepCab — GCP infra, CI/CD, React frontend, Colab notebook bridge, and code audit

**Date:** 2026-06-03
**Status:** Draft pending user review
**Owner:** Juan Garassino
**Predecessors:** [STANDING.md 2026-05-15](~/.claude/projects/-Users-juan-garassino-Code-005-products-006-deep-projects-001-deepCab/STANDING.md) (Phase -1 → P10 + MVP closeout + FR-1→FR-5 round-up)

## 1. Goal

Take the post-MVP deepCab repo (103 tests passing, all 6 backends, FastAPI + agent + Prefect, but never actually deployed) and make it production-deployable on GCP with the same look-and-feel as the user's reference CI/CD screenshot (build → push → render task → deploy → Slack notification), keyless auth via Workload Identity Federation, real production-feel services in the compose stack, a React frontend replacing the Streamlit demo, a Colab-backed training notebook reachable from VS Code, and a final audit/simplification pass over the whole codebase.

This is a single design covering five intentional sub-projects executed in order:

| # | Sub-project | Goal | Status |
|---|---|---|---|
| A | `infra/` reorganization | Pull all container / compose / GCP / k8s artifacts under one folder; preserve `.github/workflows/` at repo root | Sequenced first |
| B | "Real production-feel" container stack | Add Traefik (reverse proxy + TLS), Loki/Promtail (logs), Alertmanager (Slack alerts), pgAdmin (dev), ngrok (dev), MLflow webhook helper if needed | Sequenced second |
| C | GCP CI/CD with OIDC + Slack | Mirror the screenshot exactly but on GCP (GAR + Cloud Run default + GKE secondary). Workload Identity Federation, no long-lived JSON keys in CI | Sequenced third |
| D | React frontend + Colab notebook bridge | Replace Streamlit (003-deepCab-website) with Vite+React+TS; add Colab notebook reachable via VS Code over ngrok-tunneled jupyter | Sequenced fourth |
| E | Full audit + simplify of the new whole | `rx-simplify-agent` + manual targeted review; output an AUDIT.md | Sequenced last |

## 2. Out of scope

- Terraform / Pulumi for GCP IAM. We use one-time `gcloud` commands documented in `infra/gcp/workload-identity/README.md`. (Can fold in TF later without touching anything in this design.)
- Helm chart unifying Cloud Run + GKE manifests. We accept the small duplication of two YAML targets in exchange for a flatter learning surface (this is a low-level learning project).
- `002-deepCab-interface/` — left as a static GitHub-Pages landing page; the React app goes into `003-deepCab-website/`.
- Vertex AI managed training. Colab is the training surface; Vertex AI integration is a later sub-project.
- Migrating the broader `005-products/` workspace. Only `001-deepCab-api/` and `003-deepCab-website/` change.

## 3. Decisions locked in

| Decision | Choice | Reasoning |
|---|---|---|
| Deploy target | GCP Cloud Run (default) + GKE (secondary) | User specified both, Cloud Run default |
| Cred path | Workload Identity Federation for CI + cloud; JSON Docker secret for local dev | Mirrors AWS OIDC flow in the screenshot; keyless wherever possible |
| Slack scope (first cut) | CI/CD pipeline events + Prefect flow events | Highest signal, free wins via existing actions / hooks |
| Slack scope (second cut) | API/runtime alerts via Alertmanager + MLflow promotion via tiny runtime helper | Adds 1 container (Alertmanager); MLflow promotion handled in-process by `obs/slack.py` helper (no extra container) |
| Layout | `infra/` folder; `.github/workflows/` stays at repo root | `.github/` is GitHub-discovered and cannot move |
| Compose strategy | Layered: core + obs + dev + gpu (four files; profile-like) | Matches existing pattern; preserves user familiarity |
| Cloud target strategy | Cloud Run service.yaml + kustomize for GKE base/overlays | Approach A from brainstorming; least new-concept count |
| Frontend | Replace Streamlit (003-deepCab-website) with Vite+React+TS | User picked this option |
| Notebook bridge | Colab as remote kernel via ngrok TCP tunnel + VS Code "Specify Jupyter Server" | User picked this option |
| ngrok purpose | Expose local API for a React frontend during dev | User-specified |
| Sequence | Infra first, audit the new whole last | User picked this option |

## 4. Architecture

### 4.1 Folder layout (target end state)

```
001-deepCab-api/
├── infra/
│   ├── docker/
│   │   ├── Dockerfile                  # api image (moved from root, unchanged content)
│   │   └── .dockerignore
│   ├── compose/
│   │   ├── docker-compose.yml          # core: traefik + postgres + minio + redis + mlflow + prefect + prefect-agent + api
│   │   ├── docker-compose.obs.yml      # otel + jaeger + prom + alertmanager + grafana + loki + promtail
│   │   ├── docker-compose.dev.yml      # ngrok + pgadmin + react-dev
│   │   ├── docker-compose.gpu.yml      # GPU override (nvidia runtime, GPU=1)
│   │   └── conf/
│   │       ├── traefik/{traefik.yml,dynamic.yml}
│   │       ├── otel-collector.yaml
│   │       ├── prometheus.yml
│   │       ├── prometheus-rules.yml    # alert rules (high error rate, latency SLO, /metrics scrape failure)
│   │       ├── alertmanager.yml        # Slack receiver
│   │       ├── loki-config.yaml
│   │       ├── promtail-config.yaml
│   │       └── grafana/provisioning/   # datasources: prom, loki, jaeger
│   ├── gcp/
│   │   ├── cloud-run/service.yaml      # Knative service KRM; __IMAGE__ placeholder rendered in CI
│   │   ├── gke/
│   │   │   ├── base/
│   │   │   │   ├── deployment.yaml
│   │   │   │   ├── service.yaml
│   │   │   │   ├── configmap.yaml
│   │   │   │   └── kustomization.yaml
│   │   │   └── overlays/prod/kustomization.yaml
│   │   └── workload-identity/
│   │       └── README.md               # one-time gcloud commands
│   ├── secrets/                        # gitignored; populated from secrets.example/
│   ├── secrets.example/                # moved from root, expanded (slack_webhook_url, gcp_sa_key, ngrok_authtoken, traefik_acme_email, pgadmin_password added)
│   └── README.md                       # entry doc: layer order, bootstrap, mkcert -install
├── .github/workflows/                  # stays at repo root (GitHub-discovered)
│   ├── ci.yml                          # existing; paths updated to infra/compose/
│   ├── train-smoke.yml                 # unchanged
│   ├── nightly-bench.yml               # unchanged
│   ├── build-and-push.yml              # NEW: reusable workflow — OIDC → GAR → build → push
│   ├── deploy-cloud-run.yml            # NEW: calls build-and-push, then gcloud run services replace
│   ├── deploy-gke.yml                  # NEW: calls build-and-push, then kustomize | kubectl apply
│   └── deploy-frontend-gh-pages.yml    # NEW: builds React app, publishes 003-deepCab-website/dist to gh-pages
├── notebooks/
│   ├── colab-train-and-push.ipynb      # NEW: 6-cell notebook (setup, auth, ngrok kernel, train, GCS push, deploy trigger)
│   ├── 00-overview.ipynb … 09-ci-and-cards.ipynb  # unchanged
│   └── datascientist_deliverable.ipynb, recap_train_at_scale.ipynb  # unchanged
├── deepCab/                            # touched only by Sub-project E (audit/simplify) — no functional change in A-D
│   └── obs/slack.py                    # NEW (tiny): post_to_slack(event, payload) used by runtime (e.g. registry.set_alias)
├── Makefile                            # path updates: every `docker compose -f` now uses infra/compose/<file>
├── pyproject.toml                      # add: pyngrok (dev), optional google-cloud-aiplatform (for trigger-deploy cell)
├── CLAUDE.md                           # surgical edits (see §11)
├── README.md                           # surgical edits (see §11)
└── CONTRIBUTING.md                     # surgical edits (see §11)

003-deepCab-website/                    # full rewrite
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.ts
├── postcss.config.js
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── api/
│   │   ├── client.ts                   # typed fetch wrapper; reads VITE_API_BASE_URL
│   │   └── schemas.ts                  # GENERATED via openapi-typescript from /openapi.json
│   ├── pages/
│   │   ├── Predict.tsx                 # form → POST /predict, renders prediction + 95% PI
│   │   ├── Explain.tsx                 # GET /explain/summary → bar chart of 5 SHAP groups
│   │   └── Runs.tsx                    # GET /train list + status badges
│   ├── components/                     # Form, Card, Chart wrappers
│   └── lib/                            # utils, types
├── public/                             # static assets
├── Dockerfile                          # multi-stage: node:20-alpine builder → nginx:alpine runtime
├── .env.example                        # VITE_API_BASE_URL=https://api.deepcab.localhost
├── README.md                           # rewritten Streamlit → React
└── Makefile                            # dev, build, gen:types, docker_build

# DELETIONS in 003-deepCab-website/
# app.py, requirements.txt, Streamlit-specific Dockerfile
```

### 4.2 Container topology — full stack at a glance

```
                       ┌─────────────────────────┐
  browser ────TLS:443──►        traefik          │
                       │  (route by Host header) │
                       └────┬────────┬────────┬──┘
                            │        │        │
       app.deepcab.local────┘   api──┘   mlflow / prefect / grafana / jaeger / pgadmin / minio-console
            │                    │                            │
     ┌──────▼──────┐      ┌──────▼──────┐              ┌──────▼──────┐
     │  react-dev  │      │     api     │              │   mlflow    │
     │ (vite :5173)│      │ (fastapi    │              │  (server)   │
     └──────┬──────┘      │  +grpc      │              └──────┬──────┘
            │             │  +graphql)  │                     │
            ▼ HMR         │             │                     ▼
        (browser)         └──┬───┬───┬──┘               ┌────────────┐
                             │   │   │                  │  postgres  │
                       ┌─────┘   │   └───┐              │ (mlflow +  │
                       ▼         ▼       ▼              │  prefect)  │
                  ┌────────┐ ┌─────┐ ┌────────┐         └──────┬─────┘
                  │ redis  │ │mlflow│ │otel-col│                │
                  └────────┘ └──┬──┘ └────┬────┘         ┌──────▼──────┐
                                │         │              │   prefect   │
                                ▼         ├──► jaeger    │ + agent     │
                            ┌─────┐       │              └─────────────┘
                            │minio│       ├──► prometheus ──► alertmanager ──► Slack
                            └─────┘       │       │
                                          │       └──► grafana ◄── loki ◄── promtail
                                          └──► (all logs/traces/metrics)

                       dev profile only:    ngrok  ──── public URL ──► React (when hosted) / Colab kernel
```

18 services total when running `core + obs + dev`. Only Traefik publishes ports to host (`:80`, `:443`, `:8080` traefik dashboard).

### 4.3 Routing (Traefik)

| Hostname | → Service | Used by |
|---|---|---|
| `api.deepcab.localhost` | `api:8000` | React app, agent CLI, gRPC clients on `:50051` |
| `mlflow.deepcab.localhost` | `mlflow:5000` | dev inspection |
| `prefect.deepcab.localhost` | `prefect:4200` | dev inspection |
| `grafana.deepcab.localhost` | `grafana:3000` | dev inspection |
| `jaeger.deepcab.localhost` | `jaeger:16686` | dev inspection |
| `minio.deepcab.localhost` | `minio:9001` | dev inspection (S3 console) |
| `traefik.deepcab.localhost` | `traefik:8080` | dev (routing dashboard) |
| `pgadmin.deepcab.localhost` | `pgadmin:80` | dev only |
| `app.deepcab.localhost` | `react-dev:5173` | dev only (HMR) |

`*.deepcab.localhost` resolves to `127.0.0.1` natively in Chrome/Firefox/Safari — no `/etc/hosts` edit needed. mkcert installs a local CA so TLS is real. Production hostnames (`api.deepcab.com`, etc.) use Traefik's Let's Encrypt resolver.

### 4.4 Secrets

File-based Docker secrets under `infra/secrets/` (gitignored, seeded from `infra/secrets.example/`):

| Secret | Used by | Source in CI | Source locally |
|---|---|---|---|
| `postgres_password` | postgres, mlflow, prefect | random in setup script | `infra/secrets.example/postgres_password.example` |
| `minio_root_password` | minio, mlflow | random | example file |
| `deepcab_api_key` | api (auth) | GitHub secret | example file |
| `openai_api_key` | api (agent) | GitHub secret `OPENAI_API_KEY` | user-provided |
| `slack_webhook_url` | alertmanager, prefect-agent, runtime helper | GitHub secret `SLACK_WEBHOOK_URL` | user-provided |
| `gcp_sa_key` | api (GCS/Vertex), prefect-agent | **never in CI** — OIDC instead | user-provided JSON |
| `ngrok_authtoken` | ngrok | n/a | user-provided |
| `traefik_acme_email` | traefik | n/a (only used in staging) | user-provided or skipped locally |
| `pgadmin_password` | pgadmin | n/a | user-provided |

### 4.5 Data flow on a single user request

```
React (app.deepcab.localhost)
   │ POST /predict {pickup_*, dropoff_*, passenger_count, pickup_datetime}
   ▼
Traefik → api
   │ → app.state.model.predict_one(FeatureRow)
   │   → ONNX runtime (preferred) OR sklearn estimator fallback
   │ → emit OTel span via otel-collector
   │ → emit prom metric (request_count, request_duration)
   │ → emit log line → promtail → loki
   ▼ JSON {prediction, p_lo, p_hi, request_id}
React
   │
   ├─ Explain.tsx: GET /explain/summary?run_id=LATEST → group bars
   └─ Runs.tsx:    GET /train  → list of run IDs + status
```

## 5. CI/CD design (Sub-project C)

### 5.1 Workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | PR, push to main | lint + mypy + unit/integration tests + compose smoke (path-gated) — existing, paths updated to `infra/compose/` |
| `train-smoke.yml` | PR (paths under deepCab/models, features, training, data, schemas) | 6-backend × 1k synth training smoke (existing, unchanged) |
| `nightly-bench.yml` | cron `0 6 * * *` | Optuna 10k bench (existing, unchanged) |
| `build-and-push.yml` | `workflow_call` (reusable) | OIDC → GAR auth → docker build → tag → push image. Inputs: `image`, `tag`, `target` (cloud-run/gke). Outputs: pushed image URI |
| `deploy-cloud-run.yml` | push tag `v*`; `workflow_dispatch` | Calls build-and-push, renders `infra/gcp/cloud-run/service.yaml`, `gcloud run services replace`, Slack. **Default deploy target.** |
| `deploy-gke.yml` | `workflow_dispatch` only (until user opts in) | Calls build-and-push, `kustomize build infra/gcp/gke/overlays/prod \| kubectl apply -f -`, Slack. **Dispatch-only by default** to avoid standing GKE cluster cost; flip to tag trigger when ready |
| `deploy-frontend-gh-pages.yml` | push to main with paths under `003-deepCab-website/**` | builds React app with `VITE_API_BASE_URL` from GitHub var, publishes `dist/` to `gh-pages` branch |

### 5.2 Deploy workflow shape (mirrors the screenshot)

```yaml
name: deploy-cloud-run
on:
  push:
    tags: ['v*']
  workflow_dispatch:
    inputs:
      model_uri:
        description: 'GCS URI of model artifact (e.g. gs://deepcab-models/runs/<id>/)'
        required: false

permissions:
  id-token: write
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Slack — deploy starting
        uses: rtcamp/action-slack-notify@v2.3.3
        env:
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK_URL }}
          SLACK_TITLE: "deploy → Cloud Run starting"
          SLACK_MESSAGE: "${{ github.sha }} on ${{ github.ref_name }}"
          MSG_MINIMAL: actions url

      - name: Configure GCP credentials (OIDC)
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ vars.GCP_WIF_PROVIDER }}
          service_account: ${{ vars.GCP_DEPLOYER_SA }}

      - name: Setup gcloud
        uses: google-github-actions/setup-gcloud@v2

      - name: Login to Artifact Registry
        run: gcloud auth configure-docker ${{ vars.GCP_REGION }}-docker.pkg.dev --quiet

      - name: Build, tag, and push image to Artifact Registry
        id: build
        run: |
          IMG=${{ vars.GCP_REGION }}-docker.pkg.dev/${{ vars.GCP_PROJECT }}/deepcab/api:${{ github.ref_name }}
          docker build -f infra/docker/Dockerfile -t "$IMG" .
          docker push "$IMG"
          echo "image=$IMG" >> "$GITHUB_OUTPUT"

      - name: Render Cloud Run service spec (inject new image)
        run: |
          sed -e "s|__IMAGE__|${{ steps.build.outputs.image }}|" \
              -e "s|__MODEL_URI__|${{ inputs.model_uri }}|" \
              infra/gcp/cloud-run/service.yaml > service.rendered.yaml

      - name: Deploy Cloud Run service spec
        run: |
          gcloud run services replace service.rendered.yaml \
            --region=${{ vars.GCP_REGION }} \
            --platform=managed

      - name: Slack — success
        if: success()
        uses: rtcamp/action-slack-notify@v2.3.3
        env:
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK_URL }}
          SLACK_COLOR: good
          SLACK_TITLE: "deploy → Cloud Run ✓"
          SLACK_MESSAGE: "${{ steps.build.outputs.image }}"

      - name: Slack — failure
        if: failure()
        uses: rtcamp/action-slack-notify@v2.3.3
        env:
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK_URL }}
          SLACK_COLOR: danger
          SLACK_TITLE: "deploy → Cloud Run ✗"
          SLACK_MESSAGE: "see workflow run for details"
```

`deploy-gke.yml` has the same shape but swaps the "Render + Deploy" steps for `kustomize build … | kubectl apply -f -`.

### 5.3 Workload Identity Federation bootstrap

Documented in `infra/gcp/workload-identity/README.md`:

```bash
# 1. Create the WIF pool + provider (one-time per project)
gcloud iam workload-identity-pools create github-pool \
  --project=$PROJECT --location=global \
  --display-name="GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --project=$PROJECT --location=global \
  --workload-identity-pool=github-pool \
  --display-name="GitHub Actions" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository_owner == '$GH_OWNER'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# 2. Create deployer service account
gcloud iam service-accounts create deepcab-deployer \
  --display-name="deepCab GH Actions deployer"

# 3. Grant deploy roles
for ROLE in roles/run.admin roles/container.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:deepcab-deployer@$PROJECT.iam.gserviceaccount.com" \
    --role="$ROLE"
done

# 4. Allow the GH repo to impersonate the deployer SA
gcloud iam service-accounts add-iam-policy-binding \
  deepcab-deployer@$PROJECT.iam.gserviceaccount.com \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/$GH_OWNER/$GH_REPO"
```

GitHub variables to set (`gh variable set`):
- `GCP_PROJECT`
- `GCP_PROJECT_NUMBER`
- `GCP_REGION` (e.g. `us-central1`)
- `GCP_WIF_PROVIDER` (full resource path)
- `GCP_DEPLOYER_SA` (`deepcab-deployer@$PROJECT.iam.gserviceaccount.com`)

GitHub secrets:
- `SLACK_WEBHOOK_URL` (only)

## 6. Notebook bridge mechanics (Sub-project D part 1)

`notebooks/colab-train-and-push.ipynb` — six cells. User opens it on Colab (free or Pro+, GPU runtime), runs cells 1–3, gets the ngrok URL printed in cell 3. From VS Code:

1. `Cmd+Shift+P` → `Jupyter: Specify Jupyter Server for Connections`
2. Paste `<ngrok-url>?token=<token>`
3. Open the local copy of the notebook
4. Kernel picker → pick the remote ngrok server
5. From cell 4 onward, code runs on Colab's GPU but the editor is local VS Code

Cell contents:

```python
# Cell 1 — setup (run on Colab)
!pip install -q pyngrok jupyter_server
!git clone https://github.com/$GH_OWNER/deepCab.git /content/deepCab
%cd /content/deepCab/001-deepCab-api
!pip install -e .

# Cell 2 — auth (run on Colab)
from google.colab import auth, drive, userdata
auth.authenticate_user()        # for gcloud / gsutil
NGROK_AUTHTOKEN = userdata.get('NGROK_AUTHTOKEN')
GH_TOKEN        = userdata.get('GH_TOKEN')  # for workflow_dispatch
PROJECT         = userdata.get('GCP_PROJECT')

# Cell 3 — start jupyter server + ngrok TCP tunnel
import secrets as _s, subprocess, time
from pyngrok import ngrok
TOKEN = _s.token_urlsafe(24)
ngrok.set_auth_token(NGROK_AUTHTOKEN)
tunnel = ngrok.connect(8888, 'http')
subprocess.Popen([
    'jupyter', 'server',
    '--ip=0.0.0.0', '--port=8888', '--no-browser',
    f'--ServerApp.token={TOKEN}',
    '--ServerApp.allow_origin=*',
    '--ServerApp.disable_check_xsrf=True',
])
time.sleep(3)
print(f'Attach VS Code → "Jupyter: Specify Server" → {tunnel.public_url}?token={TOKEN}')

# Cell 4 — train (runs from VS Code attached)
from deepCab.training.train import run
from deepCab.schemas.config import TrainConfig, TorchMLPConfig, DataRef
result = run(TrainConfig(
    backend=TorchMLPConfig(epochs=50, lr=1e-3, batch_size=512),
    data=DataRef(size='full'),
))
print(result.metrics)

# Cell 5 — push artifact to GCS
import subprocess
RUN_ID = result.run_id
subprocess.run([
    'gsutil', '-m', 'cp', '-r',
    f'runs/{RUN_ID}',
    f'gs://deepcab-models/runs/{RUN_ID}/',
], check=True)
MODEL_URI = f'gs://deepcab-models/runs/{RUN_ID}/'
print(MODEL_URI)

# Cell 6 — trigger Cloud Run deploy
import requests
r = requests.post(
    f'https://api.github.com/repos/$GH_OWNER/deepCab/actions/workflows/deploy-cloud-run.yml/dispatches',
    headers={'Authorization': f'token {GH_TOKEN}', 'Accept': 'application/vnd.github+json'},
    json={'ref': 'main', 'inputs': {'model_uri': MODEL_URI}},
)
r.raise_for_status()
print('Deploy triggered.')
```

ngrok authtoken + GH_TOKEN + GCP_PROJECT are stored in Colab user-secrets (Colab's built-in `userdata`), not in the notebook.

## 7. React frontend (Sub-project D part 2)

### 7.1 Stack

- **Vite** (dev server + build)
- **React 18 + TypeScript**
- **Tailwind CSS** (utility styling — matches the no-design-time aesthetic of a learning project)
- **openapi-typescript** (generates `src/api/schemas.ts` from the running API's `/openapi.json`)
- **No state library** beyond React hooks. No router beyond `react-router-dom` if needed for the three pages.

### 7.2 Three pages

- **Predict.tsx** — form with `pickup_*`, `dropoff_*`, `passenger_count`, `pickup_datetime`. Submits `POST /predict`. Renders prediction value + 95% prediction interval (conformal). Shows the `request_id` for traceability.
- **Explain.tsx** — fetches `GET /explain/summary?run_id=LATEST`, renders a horizontal bar chart of the 5 SHAP groups (passenger, pickup_datetime, distance, pickup_location, dropoff_location).
- **Runs.tsx** — `GET /train` list of runs with backend, status, val_mae, timestamp. Status badge (`running` | `success` | `failed`).

### 7.3 API typing

```bash
npm run gen:types
# → curl http://api.deepcab.localhost/openapi.json | openapi-typescript - -o src/api/schemas.ts
```

`gen:types` is wired into CI: on any change to `deepCab/api/routers/**` or `deepCab/schemas/**`, a workflow regenerates `src/api/schemas.ts` and opens a PR. (Stretch — can be a follow-up.)

### 7.4 Deployment

- **Dev**: `react-dev` container in `docker-compose.dev.yml` running `npm run dev` (Vite on `:5173`), fronted by Traefik at `app.deepcab.localhost`. Hot Module Reload works through Traefik (WebSocket upgrade configured in dynamic.yml).
- **Prod**: `deploy-frontend-gh-pages.yml` builds with `VITE_API_BASE_URL` from `GCP_API_URL` GitHub variable, pushes `dist/` to `gh-pages` branch. (Alternative: also publish as a static container behind Traefik in `infra/compose/docker-compose.yml` for the full-stack deploy path. Defer.)

## 8. Real production-feel additions (Sub-project B detail)

Five additions beyond the existing stack:

| Service | Image | Why it earns its keep |
|---|---|---|
| traefik | `traefik:v3.1` | Replaces direct port mapping; gives real TLS locally (mkcert) and Let's Encrypt in staging; centralizes routing → middleware (rate-limit, auth, redirects) becomes a real lesson |
| loki | `grafana/loki:3.2.0` | Logs are the missing leg of observability (metrics + traces already present) |
| promtail | `grafana/promtail:3.2.0` | Ships container logs to Loki via Docker socket; zero per-service config |
| alertmanager | `prom/alertmanager:v0.27.0` | Standard Prom alerting; routes to Slack via webhook. This is where the API/runtime Slack notifications fire from |
| pgadmin | `dpage/pgadmin4:latest` (dev only) | Real prod stacks have a DB inspection tool; useful for the bootcamp lesson too |

ngrok also runs as a container (`ngrok/ngrok:latest`) in the dev profile only, configured from env var `NGROK_TUNNELS`.

`deepCab/obs/slack.py` (new, ~30 lines) — tiny helper used by:
- `registry/dispatcher.py:set_alias` — posts a Slack message on `@champion` change (Slack scope #4, no extra container needed)
- `flow_v2/retrain.py` — Prefect agent posts flow start/success/failure (Slack scope #2)

Alertmanager handles Slack scope #3 (runtime alerts from prom rules). GitHub Slack action handles Slack scope #1 (CI/CD events).

## 9. Audit + simplify (Sub-project E)

Runs **after** sub-projects A–D land. Two passes, each producing a section in `infra/AUDIT.md`.

### 9.1 Phase A — automated

```bash
# Baseline
radon cc -s -a deepCab/ > AUDIT.before.txt
radon mi  -s   deepCab/ >> AUDIT.before.txt
tokei deepCab/ > AUDIT.before.loc.txt

# Dead code
vulture deepCab/ --min-confidence 80 > AUDIT.dead.txt

# Duplicate code
pylint deepCab/ --disable=all --enable=duplicate-code --output-format=text > AUDIT.dup.txt

# Cyclic imports
pydeps deepCab --max-bacon=2 --show-cycles -T png -o AUDIT.cycles.png
```

Triage each finding: accept (apply fix) / reject (with reason) / defer (file an issue).

### 9.2 Phase B — targeted manual

Six high-payoff areas:

1. `registry/dispatcher.py` — both `save_artifact`/`load_artifact` (legacy) and `save_full_state`/`load_state_from_disk` (new) exist. Collapse legacy into new; delete the temp-dir path.
2. `schemas/settings.py` — `_maybe_read_file` resolves `file:` URIs at construction. Check whether `pydantic_settings.PydanticBaseSettingsSource` can do this natively; if so, remove the helper.
3. `agent/executor.py` + `agent/improve.py` + `agent/trace.py` — verify only one tracer writes per event. Confirm `Budget.restore` rebuilds from the trace correctly (already tested but worth a manual walk).
4. `training/preprocess.py` vs `features/pipeline.py` — clarify which is the canonical entry. Goal: `preprocess.py` calls `features/pipeline.py::preprocess_features` and adds split/clean. Eliminate any duplication of the encoder list.
5. `api/routers/*.py` — each router should be 1) request parse 2) call into a service 3) format response. If business logic creeps in, extract it.
6. Test suite — fixtures duplicated across `tests/api`, `tests/training`, `tests/models`. Consolidate to `tests/conftest.py` + one fixture per concept.

### 9.3 Acceptance criteria for the audit

- Net LOC change recorded (likely negative — we're removing legacy paths).
- Average cyclomatic complexity per function does not increase.
- All 103 tests still pass.
- No new dependencies added.
- `infra/AUDIT.md` lists every finding with an action.

## 10. Risks & open questions

| Risk | Mitigation |
|---|---|
| `.deepcab.localhost` won't resolve in Safari Tech Preview / curl without `/etc/hosts` | Add a `make hosts` target that appends entries idempotently; document the manual alternative |
| Free-tier ngrok URLs change on every restart | Document that paid ngrok ($8/mo) gives a stable subdomain; alternative is Cloudflare Tunnel (free, stable) — note it as a follow-up |
| Workload Identity setup is a one-time pain | `infra/gcp/workload-identity/README.md` is exhaustive; one-shot script available on request |
| GKE adds cluster cost ($72+/mo minimum) | Cloud Run is the default; GKE workflow is optional and only triggered with `workflow_dispatch` until needed |
| Colab disconnects after ~12h | Document checkpoint cadence in the notebook; small models train in <1h so unlikely to bite |
| React swap is a lot to bundle with infra changes | Each sub-project (A, B, C, D, E) is implementable as its own PR. Spec stays one design doc; plan will split into 5 sub-plans |
| `mkcert` requires a one-time CA install | `infra/README.md` covers it in 1 line; CI bypasses mkcert (no TLS needed in compose smoke) |

Open questions for user review:

1. **GKE node sizing** — minimal (`e2-standard-2`) for dev or skip entirely until needed? Default: minimal.
2. **Slack channels** — one channel for all events, or split by category (ci-cd / flows / alerts)? Default: one channel, `#deepcab-ops`, with prefix tags `[ci]`, `[flow]`, `[alert]`, `[mlflow]`.
3. **GitHub Pages vs containerized React in prod** — default to GitHub Pages (free, simple). If you want a single Cloud Run service that serves both API + React static, say so and we add nginx routing.
4. **Cloudflare Tunnel vs ngrok long-term** — ngrok now, Cloudflare Tunnel later? Or just commit to one?
5. **Should we wire the React app types via OpenAPI codegen in CI now, or defer?** Default: ship the script (`npm run gen:types`), defer the CI integration.

## 11. Docs to update

| File | Section | Surgical edit |
|---|---|---|
| `001-deepCab-api/CLAUDE.md` | "Architecture — refactor in flight" | Add `infra/` tree to the directory map; mark root `docker-compose*.yml` + root `Dockerfile` as moved; add `obs/slack.py` line |
| `001-deepCab-api/CLAUDE.md` | "Common commands" | Update every `docker compose -f` to `docker compose -f infra/compose/<file>`; add `make docker_dev_up`, `make docker_obs_up`, `make react_dev`, `make colab_kernel`, `make hosts`, `make wif_bootstrap` |
| `001-deepCab-api/CLAUDE.md` | "Required environment" | Add `GOOGLE_APPLICATION_CREDENTIALS`, `SLACK_WEBHOOK_URL`, `NGROK_AUTHTOKEN`, `TRAEFIK_ACME_EMAIL`, `PGADMIN_PASSWORD`. Document OIDC vs JSON path |
| `001-deepCab-api/CLAUDE.md` | "Conventions / gotchas" | Add: hostnames pattern (`*.deepcab.localhost`); secrets pattern (`infra/secrets/`); CI uses OIDC, never long-lived keys |
| `001-deepCab-api/README.md` | Quickstart | Point at `infra/`; one-liner: `make bootstrap && make docker_up` |
| `001-deepCab-api/README.md` | "What is this" | Mention React frontend + Colab notebook bridge |
| `001-deepCab-api/CONTRIBUTING.md` | "Common snags" | Add: Traefik hostnames; ngrok authtoken; mkcert one-time setup; OIDC vs JSON; Colab kernel attach steps |
| `003-deepCab-website/README.md` | Whole file | Rewrite Streamlit → React (Vite dev, build, env vars, `gen:types`) |
| `005-products/CLAUDE.md` | "Project Directory Layout" | One line: deepCab `003-` is now Vite+React (was Streamlit) |
| `005-products/DOCS.md` | deepCab entry | Same one-liner if the file exists |
| `infra/README.md` (NEW) | All | Compose layer order, bootstrap, `mkcert -install`, secrets seed instructions |
| `infra/gcp/workload-identity/README.md` (NEW) | All | One-time `gcloud` commands for WIF + SA binding (full block from §5.3) |

Doc edits execute in the same PR as the code change in each sub-project, not as a follow-up (per docs-current-protocol).

## 12. Sub-project boundaries (for the implementation plan)

Each sub-project is one PR. Estimated sizes (LOC ± documents):

| Sub-project | LOC est. | New files | Risk |
|---|---|---|---|
| A — `infra/` reorganization | ~0 net (moves only); Makefile + 1 README | 2 | Low — mechanical |
| B — Real-service additions | +400 (compose + configs + obs/slack.py) | ~12 | Medium — Traefik routing + cert dance |
| C — GCP CI/CD with OIDC + Slack | +500 (3 workflows + service.yaml + kustomize + WIF README) | ~8 | Medium-high — first-time WIF setup |
| D — React frontend + Colab notebook | +1500 (Vite scaffold + 3 pages + ipynb + Dockerfile) | ~30 | Medium — new tech surface |
| E — Audit + simplify | -500 to -2000 LOC (removals); AUDIT.md | 1 | Low — gated by 103 tests |

Net delta: probably +500 to +1500 LOC after audit deletions land.

## 13. Acceptance — what "done" looks like

Per sub-project:

- **A**: `make docker_up` works from `infra/compose/docker-compose.yml`; no compose files left at root; 103 tests still pass; CLAUDE.md paths updated.
- **B**: `make docker_up && make docker_obs_up && make docker_dev_up` brings up 18 containers; `https://api.deepcab.localhost/healthz` returns 200; Grafana shows logs from Loki + metrics from Prom + traces from Jaeger; one Prom alert fires a Slack message in a test channel.
- **C**: A tagged release `v0.1.0` triggers `deploy-cloud-run.yml`, which OIDC-auths to GCP, builds, pushes to GAR, deploys, posts Slack ✓. `deploy-gke.yml` ditto. No GCP JSON key ever stored in GitHub secrets.
- **D**: `make colab_kernel` prints a notebook URL; opening it on Colab + attaching from VS Code lets you run the train cell on Colab GPU. The React app at `app.deepcab.localhost` shows a working Predict form against the local API.
- **E**: `infra/AUDIT.md` exists; 103 tests still pass; net LOC delta documented; no new dependencies added.

End-to-end: clone fresh → `make bootstrap && make docker_up` → React frontend renders, predictions work, MLflow has runs, Prefect runs the retrain flow, Grafana shows the whole pipeline. Then `git tag v0.1.0 && git push --tags` → Cloud Run + GKE both deploy → Slack reports both.
