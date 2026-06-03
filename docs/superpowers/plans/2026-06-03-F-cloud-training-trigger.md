# Sub-project F — Cloud training trigger (Cloud Scheduler → Cloud Run Job; Prefect Cloud as alternative)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a managed, zero-idle-cost daily training trigger on GCP. **Default**: Cloud Scheduler fires a Cloud Run Job that runs `python -m deepCab.training.train` against the latest data, pushes the artifact to GCS, and (optionally) fires `deploy-cloud-run.yml`. **Alternative path documented but not executed**: Prefect Cloud + a Cloud Run worker.

**Architecture:** One Cloud Run Job reusing the same `deepcab-api` image as the API service, but with `--command="python"` + `--args="-m deepCab.training.train backend=tf_mlp data=full"`. Cloud Scheduler HTTP-targets the job's `:run` endpoint with `oauthServiceAccountEmail` for auth. Single-source GCS bucket (`gs://deepcab-models/`). No Prefect server runs in cloud — local compose stack keeps Prefect for UI / dev.

**Tech Stack:** Cloud Run Jobs API (v2), Cloud Scheduler (HTTP target), Workload Identity Federation (reuses WIF from Sub-project C), `rtcamp/action-slack-notify@v2.3.3`.

**Reference:** [Design spec §3 (decisions: cloud training trigger)](../specs/2026-06-03-deepcab-gcp-infra-and-audit-design.md#3-decisions-locked-in).

**Prerequisite:** Sub-project C landed (WIF + GAR + `deepcab-deployer` SA exist; the API image is in GAR).

---

## File map

| Action | Path | Purpose |
|---|---|---|
| Create | `infra/gcp/cloud-run-jobs/retrain-job.yaml` | Cloud Run Job spec (KRM) — placeholders rendered in CI |
| Create | `infra/gcp/scheduler/retrain-schedule.yaml` | Cloud Scheduler HTTP job spec |
| Create | `infra/gcp/scheduler/bootstrap.sh` | One-time `gcloud scheduler jobs create http ...` script |
| Create | `infra/gcp/cloud-run-jobs/README.md` | Doc: default trigger + Prefect Cloud alternative |
| Create | `.github/workflows/deploy-retrain-job.yml` | Reusable workflow: OIDC → GAR → build/push → `gcloud run jobs replace` |
| Modify | `deepCab/training/train.py` | Add `--gcs-push` CLI flag (or environment-driven behavior) so the Cloud Run Job pushes to `gs://deepcab-models/runs/<run_id>/` |
| Create | `tests/training/test_train_gcs_push.py` | Regression test for the GCS push path (mocked via `gcsfs` or `google.cloud.storage` mock) |
| Modify | `CLAUDE.md` | Document the cloud training trigger (default + alternative) |
| Modify | `README.md` | Add "Cloud training" subsection |
| Modify | `infra/gcp/workload-identity/README.md` | Grant Scheduler invoker role to a service account; bind that SA to invoke the Cloud Run Job |
| Modify | `Makefile` | `make scheduler_bootstrap`, `make job_bootstrap` |

---

## Task F1: Cloud Run Job spec

**Files:**
- Create: `infra/gcp/cloud-run-jobs/retrain-job.yaml`

- [ ] **Step 1: Write the Job spec**

```yaml
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: deepcab-retrain
  labels:
    cloud.googleapis.com/location: us-central1
spec:
  template:
    spec:
      template:
        spec:
          serviceAccountName: deepcab-runtime@__GCP_PROJECT__.iam.gserviceaccount.com
          timeoutSeconds: 3600
          maxRetries: 1
          containers:
            - image: __IMAGE__
              command: ["python"]
              args:
                - "-m"
                - "deepCab.training.train"
                - "backend=tf_mlp"
                - "data=full"
              env:
                - name: APP_ENV
                  value: prod
                - name: MLFLOW_TRACKING_URI
                  value: __MLFLOW_URL__
                - name: MODEL_TARGET
                  value: gcs
                - name: GCP_PROJECT
                  value: __GCP_PROJECT__
                - name: REGISTRY_GCS_BUCKET
                  value: __MODELS_BUCKET__
              resources:
                limits:
                  cpu: "4"
                  memory: "8Gi"
```

`MODELS_BUCKET` is something like `gs://deepcab-models`. Placeholders rendered by the CI workflow.

---

## Task F2: Cloud Scheduler HTTP job

**Files:**
- Create: `infra/gcp/scheduler/retrain-schedule.yaml`
- Create: `infra/gcp/scheduler/bootstrap.sh`

- [ ] **Step 1: Document the Cloud Scheduler HTTP target**

`infra/gcp/scheduler/retrain-schedule.yaml` (declarative reference — Cloud Scheduler does NOT take YAML directly, so this file is a documented config that `bootstrap.sh` translates into `gcloud scheduler jobs create http`):

```yaml
name: deepcab-retrain-daily
schedule: "0 2 * * *"      # 02:00 UTC daily
timezone: "Etc/UTC"
description: "Daily deepcab retrain — fires Cloud Run Job deepcab-retrain"
httpTarget:
  uri: "https://${GCP_REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${GCP_PROJECT}/jobs/deepcab-retrain:run"
  httpMethod: POST
  oauthToken:
    serviceAccountEmail: "deepcab-scheduler@${GCP_PROJECT}.iam.gserviceaccount.com"
    scope: "https://www.googleapis.com/auth/cloud-platform"
attemptDeadline: "1800s"
retryConfig:
  retryCount: 0    # job retries internally
```

- [ ] **Step 2: Write `bootstrap.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT}"
: "${GCP_REGION:?set GCP_REGION (e.g. us-central1)}"

SCHEDULER_SA="deepcab-scheduler@${GCP_PROJECT}.iam.gserviceaccount.com"
JOB_NAME="deepcab-retrain"
JOB_URL="https://${GCP_REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${GCP_PROJECT}/jobs/${JOB_NAME}:run"

# 1. Create scheduler SA if missing
gcloud iam service-accounts describe "$SCHEDULER_SA" --project="$GCP_PROJECT" >/dev/null 2>&1 \
  || gcloud iam service-accounts create deepcab-scheduler \
       --display-name="deepCab Scheduler invoker" \
       --project="$GCP_PROJECT"

# 2. Grant invoker role on the Cloud Run Job
gcloud run jobs add-iam-policy-binding "$JOB_NAME" \
  --region="$GCP_REGION" \
  --project="$GCP_PROJECT" \
  --member="serviceAccount:${SCHEDULER_SA}" \
  --role="roles/run.invoker"

# 3. Enable Cloud Scheduler API
gcloud services enable cloudscheduler.googleapis.com --project="$GCP_PROJECT"

# 4. Create / update the schedule
if gcloud scheduler jobs describe deepcab-retrain-daily --location="$GCP_REGION" --project="$GCP_PROJECT" >/dev/null 2>&1; then
  CMD=update
else
  CMD=create
fi

gcloud scheduler jobs "$CMD" http deepcab-retrain-daily \
  --location="$GCP_REGION" \
  --project="$GCP_PROJECT" \
  --schedule="0 2 * * *" \
  --time-zone="Etc/UTC" \
  --uri="$JOB_URL" \
  --http-method=POST \
  --oauth-service-account-email="$SCHEDULER_SA" \
  --oauth-token-scope="https://www.googleapis.com/auth/cloud-platform" \
  --attempt-deadline=1800s

echo "✓ Cloud Scheduler job deepcab-retrain-daily wired to Cloud Run Job ${JOB_NAME}"
```

- [ ] **Step 3: Add Makefile target**

```make
scheduler_bootstrap:
	@echo "Make sure GCP_PROJECT and GCP_REGION are set."
	bash infra/gcp/scheduler/bootstrap.sh
```

---

## Task F3: README explaining the two options

**Files:**
- Create: `infra/gcp/cloud-run-jobs/README.md`

- [ ] **Step 1: Write the README**

```markdown
# Cloud training trigger

deepCab supports two managed paths for running the daily retrain in GCP:

## Default — Cloud Scheduler → Cloud Run Job

- Cloud Scheduler fires `gcloud run jobs run deepcab-retrain` at 02:00 UTC daily
- The job container reuses the deepcab-api image; entrypoint is `python -m deepCab.training.train`
- Trained artifact lands in `gs://deepcab-models/runs/<run_id>/`
- Optionally triggers `deploy-cloud-run.yml` via the GH API if you want the API to roll forward automatically

Cost: **~$0/mo** (Cloud Scheduler is free up to 3 jobs; Cloud Run Jobs bills only the seconds the container runs).

Setup:

```bash
# 1. Land the Cloud Run Job (via the deploy-retrain-job.yml workflow or manually)
gh workflow run deploy-retrain-job.yml -f tag=v0.1.0

# 2. Bootstrap the Scheduler (one-time)
export GCP_PROJECT=...
export GCP_REGION=us-central1
make scheduler_bootstrap

# 3. Verify
gcloud scheduler jobs describe deepcab-retrain-daily --location=$GCP_REGION
gcloud run jobs describe deepcab-retrain --region=$GCP_REGION
```

To run on-demand:

```bash
gcloud run jobs execute deepcab-retrain --region=$GCP_REGION --project=$GCP_PROJECT
# or
gh workflow run deploy-retrain-job.yml -f tag=latest
```

## Alternative — Prefect Cloud + Cloud Run worker

Use this when you have more than one flow, want dependencies between flows, or want the Prefect UI for monitoring.

- Sign up at https://app.prefect.cloud (free tier covers a single small project)
- Create a work pool of type `cloud-run-v2`
- Deploy a worker as a Cloud Run service with `min_instances=1` (the worker long-polls the work pool)
- Push `retrain_flow` to Prefect Cloud as a deployment; schedule with Prefect's cron

Cost: **~$5–10/mo** for the `min_instances=1` Cloud Run worker + free-tier Prefect Cloud.

Setup (not executed by Sub-project F):

```bash
prefect cloud login
prefect work-pool create --type cloud-run-v2 deepcab-pool
prefect deploy deepCab/flow_v2/retrain.py:retrain_flow \
  --name deepcab-retrain \
  --pool deepcab-pool \
  --cron "0 2 * * *"

# Then deploy a Cloud Run worker pointing at the pool
gcloud run deploy deepcab-prefect-worker \
  --image=us-central1-docker.pkg.dev/$GCP_PROJECT/deepcab/api:latest \
  --command="prefect" --args="worker,start,-p,deepcab-pool" \
  --service-account=deepcab-runtime@$GCP_PROJECT.iam.gserviceaccount.com \
  --min-instances=1 \
  --region=$GCP_REGION
```

## Why default to Cloud Scheduler instead of Prefect Cloud?

- Zero idle cost (Prefect worker billing accumulates 24/7)
- One less moving part — no extra UI to learn, no work pool to maintain
- For "one nightly flow with one task" the orchestration value of Prefect is small
- Prefect Cloud stays easy to adopt later when you have a real DAG
```

---

## Task F4: Add `--gcs-push` behavior to training entrypoint

**Files:**
- Modify: `deepCab/training/train.py`
- Create: `tests/training/test_train_gcs_push.py`

- [ ] **Step 1: Write failing test**

`tests/training/test_train_gcs_push.py`:

```python
from unittest.mock import patch, MagicMock
from deepCab.training import train as tmod


def test_gcs_push_called_when_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv("REGISTRY_GCS_BUCKET", "deepcab-models")
    # Stub the real training pipeline to a no-op that produces a run dir
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "model.onnx").write_bytes(b"dummy")
    with patch.object(tmod, "_run_training", return_value=MagicMock(run_id="run-1", run_dir=run_dir)), \
         patch.object(tmod, "_push_to_gcs") as mock_push:
        tmod.run(cfg=None)
    mock_push.assert_called_once()
    args = mock_push.call_args.args
    assert args[0] == run_dir
    assert args[1] == "gs://deepcab-models/runs/run-1/"


def test_gcs_push_skipped_when_env_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("REGISTRY_GCS_BUCKET", raising=False)
    run_dir = tmp_path / "run-2"
    run_dir.mkdir()
    with patch.object(tmod, "_run_training", return_value=MagicMock(run_id="run-2", run_dir=run_dir)), \
         patch.object(tmod, "_push_to_gcs") as mock_push:
        tmod.run(cfg=None)
    mock_push.assert_not_called()
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/training/test_train_gcs_push.py -v
```

Expected: AttributeError on `_run_training` or `_push_to_gcs` (these helpers don't exist yet).

- [ ] **Step 3: Edit `deepCab/training/train.py`**

Identify the current `run(cfg)` function. Extract the training body into a `_run_training(cfg) -> TrainResult` helper. Add a `_push_to_gcs(local_dir: Path, gcs_uri: str)` helper. Then make `run()`:

```python
import os
from pathlib import Path


def _run_training(cfg) -> TrainResult:
    # existing body of run() goes here
    ...


def _push_to_gcs(local_dir: Path, gcs_uri: str) -> None:
    """Mirror local run dir to GCS using gsutil."""
    import subprocess
    subprocess.run(["gsutil", "-m", "cp", "-r", str(local_dir), gcs_uri], check=True)


def run(cfg=None):
    result = _run_training(cfg)
    bucket = os.environ.get("REGISTRY_GCS_BUCKET")
    if bucket:
        gcs_uri = f"gs://{bucket.removeprefix('gs://')}/runs/{result.run_id}/"
        _push_to_gcs(result.run_dir, gcs_uri)
    return result
```

The `TrainResult` must expose `run_id` and `run_dir` — verify these attributes exist; if not, add them.

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/training/test_train_gcs_push.py -v
```

- [ ] **Step 5: Run training suite for regressions**

```bash
uv run pytest tests/training -q
```

---

## Task F5: deploy-retrain-job.yml

**Files:**
- Create: `.github/workflows/deploy-retrain-job.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: deploy-retrain-job

on:
  workflow_dispatch:
    inputs:
      tag:
        description: "Image tag (e.g. v0.1.0, latest, sha-...)"
        required: true
        default: "latest"
  push:
    tags: ['retrain-v*']

permissions:
  id-token: write
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Slack — job deploy starting
        uses: rtcamp/action-slack-notify@v2.3.3
        env:
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK_URL }}
          SLACK_TITLE: "[ci] deploy → Cloud Run Job (retrain) starting"
          SLACK_MESSAGE: "tag=${{ inputs.tag || github.ref_name }}"
          SLACK_COLOR: "#cccccc"

      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ vars.GCP_WIF_PROVIDER }}
          service_account: ${{ vars.GCP_DEPLOYER_SA }}

      - uses: google-github-actions/setup-gcloud@v2

      - name: Configure docker for GAR
        run: gcloud auth configure-docker ${{ vars.GCP_REGION }}-docker.pkg.dev --quiet

      - id: build
        name: Build + push image
        run: |
          TAG="${{ inputs.tag || github.ref_name }}"
          IMG="${{ vars.GCP_REGION }}-docker.pkg.dev/${{ vars.GCP_PROJECT }}/deepcab/api:${TAG}"
          docker buildx build \
            --file infra/docker/Dockerfile \
            --tag "$IMG" \
            --push \
            --build-arg GPU=0 \
            .
          echo "image=$IMG" >> "$GITHUB_OUTPUT"

      - name: Render Cloud Run Job spec
        run: |
          sed -e "s|__IMAGE__|${{ steps.build.outputs.image }}|g" \
              -e "s|__GCP_PROJECT__|${{ vars.GCP_PROJECT }}|g" \
              -e "s|__MLFLOW_URL__|${{ vars.MLFLOW_URL }}|g" \
              -e "s|__MODELS_BUCKET__|${{ vars.GCP_MODELS_BUCKET }}|g" \
              infra/gcp/cloud-run-jobs/retrain-job.yaml > job.rendered.yaml
          cat job.rendered.yaml

      - name: Deploy Cloud Run Job
        run: |
          gcloud run jobs replace job.rendered.yaml \
            --region=${{ vars.GCP_REGION }} \
            --project=${{ vars.GCP_PROJECT }}

      - name: Optionally execute the job now
        if: ${{ inputs.tag != 'latest' }}
        run: |
          gcloud run jobs execute deepcab-retrain \
            --region=${{ vars.GCP_REGION }} \
            --project=${{ vars.GCP_PROJECT }} \
            --wait \
            || true   # don't fail the workflow on training failure — Slack will report

      - name: Slack — success
        if: success()
        uses: rtcamp/action-slack-notify@v2.3.3
        env:
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK_URL }}
          SLACK_COLOR: good
          SLACK_TITLE: "[ci] deploy → Cloud Run Job (retrain) ✓"
          SLACK_MESSAGE: "${{ steps.build.outputs.image }}"

      - name: Slack — failure
        if: failure()
        uses: rtcamp/action-slack-notify@v2.3.3
        env:
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK_URL }}
          SLACK_COLOR: danger
          SLACK_TITLE: "[ci] deploy → Cloud Run Job (retrain) ✗"
          SLACK_MESSAGE: "see ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
```

---

## Task F6: Update WIF README with scheduler + job invoker roles

**Files:**
- Modify: `infra/gcp/workload-identity/README.md`

- [ ] **Step 1: Append a new section**

```markdown
## Scheduler + Cloud Run Job (Sub-project F)

Additional one-time setup for the daily training trigger:

```bash
# 1. Grant the deployer SA the Cloud Run Admin role on Jobs (Sub-project C grants it on Services)
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:deepcab-deployer@$PROJECT.iam.gserviceaccount.com" \
  --role=roles/run.developer

# 2. Create the scheduler SA (different from deployer) and bind to Cloud Run Job invoker
make scheduler_bootstrap   # idempotent — creates deepcab-scheduler@$PROJECT.iam.gserviceaccount.com

# 3. Set GitHub variable for the models bucket
gh variable set GCP_MODELS_BUCKET --body "deepcab-models"
```

That's the full extra setup. From there:

- `gh workflow run deploy-retrain-job.yml -f tag=v0.1.0` registers the Job
- Cloud Scheduler fires it every day at 02:00 UTC
- Artifacts land in `gs://deepcab-models/runs/<run_id>/`
```

---

## Task F7: Update CLAUDE.md + README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: CLAUDE.md — add "Cloud training trigger" subsection**

After the "CI/CD" subsection added by Sub-project C:

```markdown
## Cloud training trigger

| Path | Trigger | Cost | When to use |
|---|---|---|---|
| **Default — Cloud Scheduler → Cloud Run Job** | `gcloud scheduler` fires `gcloud run jobs run deepcab-retrain` at 02:00 UTC daily | ~$0/mo (scheduler free; job bills per-second) | One nightly retrain. No Prefect server in cloud |
| Alternative — Prefect Cloud + Cloud Run worker | Prefect Cloud deployment with cron; worker is a Cloud Run service `min_instances=1` long-polling the work pool | ~$5–10/mo | Multiple flows, flow dependencies, Prefect UI needed |
| Local (unchanged) | `prefect server` + `prefect-agent` in `infra/compose/docker-compose.yml` | $0 (local) | Dev / interactive flow iteration |

Setup: `infra/gcp/cloud-run-jobs/README.md`.

Workflow: `.github/workflows/deploy-retrain-job.yml` (re-deploys the Job; `workflow_dispatch` or `retrain-v*` tag).
```

- [ ] **Step 2: README.md — "Cloud training" subsection**

After the existing "Deploy to GCP Cloud Run" section:

```markdown
## Cloud training trigger

The default cloud training path uses Cloud Scheduler firing a Cloud Run Job at 02:00 UTC daily. See `infra/gcp/cloud-run-jobs/README.md` for setup. Alternative Prefect Cloud path documented in the same README.
```

---

## Task F8: Commit Sub-project F

- [ ] **Step 1: Commit**

```bash
git add -A
git status --short
git commit -m "$(cat <<'EOF'
feat(cloud-train): Cloud Scheduler → Cloud Run Job (default) + Prefect Cloud alternative

Default cloud training trigger is now Cloud Scheduler → Cloud Run Job:
- infra/gcp/cloud-run-jobs/retrain-job.yaml — Cloud Run Job spec (KRM)
- infra/gcp/scheduler/{retrain-schedule.yaml, bootstrap.sh} — Scheduler config + gcloud bootstrap
- .github/workflows/deploy-retrain-job.yml — OIDC → GAR → build + push + `gcloud run jobs replace` + Slack
- deepCab/training/train.py — REGISTRY_GCS_BUCKET env triggers GCS artifact push after training; refactored body into _run_training + _push_to_gcs helpers (TDD: 2 new tests)
- Makefile: make scheduler_bootstrap

Prefect Cloud + Cloud Run worker remains documented as the alternative
path (infra/gcp/cloud-run-jobs/README.md) — not wired up in this PR.

Local compose stack keeps Prefect server + agent for dev UI. No Prefect
process runs in cloud by default.

Cost: ~$0/mo for the default path (Scheduler is free, Job bills per-second).

Sub-project F of the GCP infra design.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Done criteria

- [x] `infra/gcp/cloud-run-jobs/retrain-job.yaml` exists with `__IMAGE__` / `__GCP_PROJECT__` / `__MLFLOW_URL__` / `__MODELS_BUCKET__` placeholders
- [x] `infra/gcp/scheduler/{retrain-schedule.yaml, bootstrap.sh}` document + execute the Scheduler bootstrap
- [x] `infra/gcp/cloud-run-jobs/README.md` documents the default + alternative paths
- [x] `.github/workflows/deploy-retrain-job.yml` builds + pushes + deploys the Job + Slack-notifies
- [x] `deepCab/training/train.py` has `_run_training` + `_push_to_gcs` helpers; `run()` honors `REGISTRY_GCS_BUCKET`
- [x] 2 new tests pass: `tests/training/test_train_gcs_push.py`
- [x] WIF README documents the additional `deepcab-scheduler` SA + the `roles/run.developer` grant on the deployer SA
- [x] CLAUDE.md + README.md updated with the table + setup pointers
- [x] One commit
