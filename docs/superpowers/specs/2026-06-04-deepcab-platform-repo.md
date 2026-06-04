# deepCab platform repo — Terraform IaC + cross-repo CI

**Date:** 2026-06-04 (Plan G — post simplification audit)
**Status:** In-progress
**Scope:** Convert `002-deepCab-interface` → `002-deepCab-platform`. Move all GCP provisioning (WIF, GAR, Cloud Run, Cloud SQL, GCS buckets, Secret Manager, Cloud Scheduler) from `001-deepCab-api/infra/gcp/` into a new Terraform tree in 002. Keep the existing static landing page coexisting.

**Goal:** A real-world demo/learning platform repo. Real Terraform code that can be `apply`ed; documented per module; multi-env (dev/staging/prod). 001 keeps its Dockerfile + compose + image-build workflows; 002 owns everything GCP-provisioned.

## 1. Decisions locked in

| Decision | Choice |
|---|---|
| Rename local dir | `002-deepCab-interface` → `002-deepCab-platform` (DONE before this spec) |
| Rename GitHub repo | `deepCab-interface` → `deepCab-platform` via `gh repo rename` (DONE) |
| Envs | dev + staging + prod (3 envs) |
| Cross-repo deploy mechanism | **Independent**: 001 does image-only updates (`gcloud run services update --image=$IMG`); 002 owns service shape (CPU/memory/scale/env vars/IAM) via TF |
| Landing page | Keep `index.html` + `CNAME` + `images/` at repo root; platform stuff lives under subdirs (`terraform/`, `cloud-manifests/`, `docs/`) |
| TF backend | GCS-backed remote state per env (`gs://deepcab-tfstate-<env>/`); state bucket bootstrapping is a one-time manual step documented in RUNBOOK |
| Local-dev compose | STAYS in 001 (it's about how to build/run the app locally; not platform concern) |

## 2. Folder layout (target end state)

```
002-deepCab-platform/
├── index.html, CNAME, images/, README.md          # existing landing page — preserved
├── cloud-manifests/                               # raw YAML moved out of 001/infra/gcp/ (preserved as reference; TF supersedes)
│   ├── cloud-run/service.yaml
│   ├── cloud-run-jobs/retrain-job.yaml
│   ├── scheduler/{retrain-schedule.yaml,bootstrap.sh}
│   ├── gke/{base/,overlays/prod/}
│   └── workload-identity/{README.md,bootstrap.sh}  # deprecated by terraform/modules/wif but kept for diff comparison
├── terraform/
│   ├── modules/
│   │   ├── gar/                                    # Artifact Registry repo
│   │   ├── storage/                                # GCS buckets (mlflow_artifacts, deepcab_models, tfstate)
│   │   ├── cloud_sql/                              # Cloud SQL Postgres for MLflow tracking + Prefect (if used)
│   │   ├── secret_manager/                         # Slack webhook, OpenAI key, MLflow DB password seeds
│   │   ├── wif/                                    # Workload Identity Pool/Provider + deployer SA + runtime SA + scheduler SA
│   │   ├── cloud_run/                              # API service; Prefect server/worker (optional)
│   │   ├── cloud_run_job/                          # Retrain job
│   │   ├── scheduler/                              # Cloud Scheduler firing the Job
│   │   ├── vpc/                                    # Network + Cloud NAT (only if a private service needs it)
│   │   ├── gke/                                    # Cluster module (gated behind enable_gke = false default)
│   │   ├── dns/                                    # Cloud DNS A records for api.<env>.deepcab.com
│   │   └── iam/                                    # Cross-cutting bindings
│   ├── envs/
│   │   ├── dev/{main.tf,backend.tf,variables.tf,terraform.tfvars}
│   │   ├── staging/{main.tf,backend.tf,variables.tf,terraform.tfvars}
│   │   └── prod/{main.tf,backend.tf,variables.tf,terraform.tfvars}
│   └── README.md                                   # module reference
├── .github/workflows/
│   ├── platform-plan.yml                           # PRs touching terraform/ → terraform plan per env
│   └── platform-apply.yml                          # push to main → terraform apply (gated)
├── docs/
│   ├── ARCHITECTURE.md                             # diagram + data flow + cross-repo split
│   ├── ENVIRONMENTS.md                             # per-env table (sizes, costs, URLs)
│   ├── COSTS.md                                    # rough monthly cost per env
│   └── RUNBOOK.md                                  # bootstrap a new env, rotate secrets, etc.
├── Makefile                                        # make plan/apply/init/destroy per env
└── README.md                                       # overwritten: platform purpose + 5-min quickstart
```

## 3. Module-by-module sketch

### `terraform/modules/gar/`
- One Artifact Registry repo named `deepcab` in `var.region`.
- Output: `repo_url` (e.g. `us-central1-docker.pkg.dev/$PROJECT/deepcab`).
- Used by: deploy workflows in 001.

### `terraform/modules/storage/`
- Three GCS buckets:
  - `${project}-mlflow-artifacts-${env}` (MLflow artifact store)
  - `${project}-deepcab-models-${env}` (trained model artifacts)
  - `${project}-tfstate-${env}` (terraform state — bootstrapped manually first, then imported)
- Lifecycle rules: tfstate has versioning + 30-day delete; models have 90-day archival.

### `terraform/modules/cloud_sql/`
- Cloud SQL for Postgres 16 (`db-f1-micro` in dev, `db-g1-small` in staging, `db-custom-2-4096` in prod).
- Two databases: `mlflow` + `prefect` (optional, only if Prefect cloud setup is chosen later).
- Private IP via VPC + Cloud SQL Proxy from Cloud Run.
- Output: connection name, instance host.

### `terraform/modules/secret_manager/`
- Secrets seeded (with empty placeholder values; real values populated via `gcloud secrets versions add` post-apply):
  - `slack-webhook-url`
  - `openai-api-key`
  - `deepcab-api-key` (the X-API-Key for protected endpoints)
  - `mlflow-db-password` (auto-rotated maybe)
- IAM bindings: `deepcab-runtime` SA gets `roles/secretmanager.secretAccessor`.

### `terraform/modules/wif/`
- Replaces `001/infra/gcp/workload-identity/bootstrap.sh`.
- Inputs: `gh_owner`, `gh_repo` (the API repo that builds images).
- Resources:
  - `google_iam_workload_identity_pool` "github-pool"
  - `google_iam_workload_identity_pool_provider` "github-provider"
  - `google_service_account` "deepcab-deployer" + roles bindings (run.developer, artifactregistry.writer, etc.)
  - `google_service_account` "deepcab-runtime" + roles bindings (storage.objectViewer, secretmanager.secretAccessor, etc.)
  - `google_service_account` "deepcab-scheduler" + run.invoker
  - `google_service_account_iam_binding` "wif-binding" allowing the GH repo to impersonate deployer SA
  - `google_service_account_iam_member` "deployer→runtime impersonation"
- Outputs: full WIF provider path, deployer SA email, runtime SA email, scheduler SA email.

### `terraform/modules/cloud_run/`
- Service "deepcab-api":
  - Image: `var.api_image` (defaults to `${gar.repo_url}/api:latest` — 001 overrides via `services update --image`)
  - CPU/memory: per-env vars
  - Env vars: `APP_ENV`, `MLFLOW_TRACKING_URI`, `MODEL_TARGET=gcs`, `GCP_PROJECT`, `REGISTRY_GCS_BUCKET`
  - SA: runtime SA from `wif` module
  - Probes: `/healthz` startup + liveness
- (Optional) Service "deepcab-prefect-worker" gated by `enable_prefect_worker = false` default.

### `terraform/modules/cloud_run_job/`
- Job "deepcab-retrain":
  - Image: `var.retrain_image` (same image, different entry — `python -m deepCab.training.train`)
  - Args: configurable backend + data size per env
  - Resources: CPU 4 / mem 8Gi
  - SA: runtime SA

### `terraform/modules/scheduler/`
- Cloud Scheduler HTTP job "deepcab-retrain-daily":
  - Schedule: `var.cron_schedule` (default `"0 2 * * *"`)
  - Target: `https://${region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${project}/jobs/deepcab-retrain:run`
  - Auth: OAuth token using scheduler SA

### `terraform/modules/gke/` (optional, gated)
- `var.enable_gke = false` default.
- When enabled: a `e2-standard-2` regional cluster, 1-3 node autoscaling.
- Used only by the `deploy-gke.yml` workflow (which is dispatch-only).

### `terraform/modules/dns/`
- Cloud DNS managed zone for `deepcab.com` (or whatever domain is provided).
- A records: `api.${env}.deepcab.com`, `mlflow.${env}.deepcab.com`, `grafana.${env}.deepcab.com`.
- Defaults to disabled if `var.dns_zone_name == ""`.

### `terraform/modules/iam/`
- Cross-cutting bindings that don't fit cleanly into a single resource module.

## 4. Per-env composition

### `envs/dev/`
- Sizes: smallest tier across the board (db-f1-micro, Cloud Run cpu=1/mem=512Mi/min=0/max=2)
- DNS: disabled
- GKE: disabled
- Prefect worker: disabled
- Budget: ~$5-15/mo idle

### `envs/staging/`
- Sizes: db-g1-small, Cloud Run cpu=1/mem=1Gi/min=0/max=4
- DNS: enabled (api.staging.deepcab.com)
- GKE: disabled
- Prefect worker: disabled
- Budget: ~$20-40/mo

### `envs/prod/`
- Sizes: db-custom-2-4096, Cloud Run cpu=2/mem=2Gi/min=1/max=10
- DNS: enabled (api.deepcab.com)
- GKE: disabled by default (toggle via tfvars)
- Prefect worker: disabled by default
- Budget: ~$50-150/mo

## 5. Workflows

### `002/.github/workflows/platform-plan.yml`
- Trigger: PR with paths under `terraform/**`
- Per env (matrix dev+staging+prod), run `terraform init && terraform plan` and post the plan as a PR comment.
- Auth via OIDC → GCP using the `terraform-planner` SA (read-only).

### `002/.github/workflows/platform-apply.yml`
- Trigger: push to main with paths under `terraform/**`, OR `workflow_dispatch`
- Per env, sequentially: `terraform init && terraform apply -auto-approve`
- Auth via OIDC → GCP using the `terraform-applier` SA (full admin).
- Slack notification on success/failure.

### `001/.github/workflows/deploy-cloud-run.yml` (UPDATED in Plan G2)
- Drop the `Render Cloud Run service spec` step (TF owns the spec now).
- Replace `gcloud run services replace service.rendered.yaml` with:
  ```bash
  gcloud run services update deepcab-api \
    --image=${{ steps.build.outputs.image }} \
    --region=${{ vars.GCP_REGION }} \
    --project=${{ vars.GCP_PROJECT }}
  ```
- Image-only update; doesn't touch CPU/memory/env vars/IAM (TF owns those).
- Slack notifications preserved.

### `001/.github/workflows/deploy-retrain-job.yml` (UPDATED in Plan G2)
- Same pattern: `gcloud run jobs update deepcab-retrain --image=...` instead of `replace`.

## 6. Bootstrap order (one-time, documented in `002/docs/RUNBOOK.md`)

1. Create the GCS state bucket manually: `gcloud storage buckets create gs://deepcab-tfstate-dev` (chicken-and-egg).
2. In `002/terraform/envs/dev/`, run `terraform init` (initializes the backend).
3. Run `terraform apply` — provisions WIF, SAs, GAR, GCS buckets, Cloud SQL (5-10 min).
4. Populate secrets: `gcloud secrets versions add slack-webhook-url --data-file=-`.
5. From 001, push a `v0.1.0` tag — first image lands in GAR.
6. Back in 002, run `terraform apply` again (now that the image exists in GAR, Cloud Run service can reference it).
7. Hit the Cloud Run URL.

For staging/prod: same flow, swap `envs/dev` → `envs/staging` or `envs/prod`.

## 7. Cross-cutting concerns

- **State locking**: GCS backend uses Cloud Storage's strong consistency; no separate lock table needed.
- **Drift detection**: weekly cron in `platform-plan.yml` posts any drift to Slack.
- **Secret rotation**: documented in RUNBOOK; can be done out-of-band with `gcloud secrets versions add`.
- **Cost guardrails**: `prod` budget alert at $200/mo via TF.

## 8. Plan G execution

Two subagents:
- **G1**: in `002-deepCab-platform/`, do the heavy lift — move `001/infra/gcp/` content to `cloud-manifests/`; scaffold full Terraform tree; write workflows + docs.
- **G2**: in `001-deepCab-api/`, update `deploy-cloud-run.yml` + `deploy-retrain-job.yml` to use `services update --image` instead of `services replace`; add a deprecation note to `infra/gcp/workload-identity/README.md` pointing to `../../../002-deepCab-platform/terraform/modules/wif/`.

Each subagent commits in its own repo (disjoint git contexts — no index race possible).

## 9. Out of scope (explicit non-goals)

- Actually running `terraform apply` against a real GCP project (the user does that when ready)
- Provisioning the GCS state buckets via TF (chicken-and-egg; bootstrap is manual)
- Migrating MLflow from compose to Cloud SQL (the TF module declares the DB; the actual MLflow data migration is a separate pass)
- Setting up Grafana Cloud (could be a future module)
- Terraform Cloud / Atlantis (using bare GH Actions; could upgrade later)

## 10. Docs to update

| File | Edit |
|---|---|
| `001-deepCab-api/CLAUDE.md` | Update directory listing — `002` is now platform, not interface |
| `001-deepCab-api/README.md` | Quickstart points at 002 for cloud bootstrap |
| `001-deepCab-api/infra/gcp/workload-identity/README.md` | Add deprecation note → `002/terraform/modules/wif/` |
| Parent `001-deepCab/CLAUDE.md` | Same one-liner about 002 being platform |
| `002-deepCab-platform/README.md` | Full rewrite — platform purpose, quickstart, layered TF, links to docs/ |
| `002-deepCab-platform/docs/{ARCHITECTURE,ENVIRONMENTS,COSTS,RUNBOOK}.md` | New |

## 11. Done criteria

- `002-deepCab-platform/terraform/envs/dev/` runs `terraform init && terraform plan` cleanly against a real (or stub) GCP project
- `002-deepCab-platform/.github/workflows/platform-plan.yml` lints valid via YAML parse
- `001-deepCab-api/.github/workflows/deploy-cloud-run.yml` no longer references `services replace`; uses `services update --image=...`
- Both repos clean working tree
- All 178 tests still pass in 001 (Plan G doesn't touch app code)
