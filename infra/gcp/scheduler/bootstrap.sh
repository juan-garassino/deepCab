#!/usr/bin/env bash
# ⚠️ Deprecated as of 2026-06-04 (Plan G2)
#
# This bash bootstrap has been superseded by Terraform in the platform repo.
# See ../../../../002-deepCab-platform/terraform/modules/scheduler/ for the
# IaC equivalent, and ../../../../002-deepCab-platform/docs/RUNBOOK.md for
# the bootstrap order.
#
# This script is retained for reference but should NOT be used for new
# environments — let Terraform manage the Cloud Scheduler + scheduler SA.
#
# Bootstrap the Cloud Scheduler -> Cloud Run Job retrain trigger.
#
# Idempotent: each step is safe to re-run. Creates the deepcab-scheduler SA
# (if missing), grants it run.invoker on the deepcab-retrain Job, enables the
# Cloud Scheduler API, and creates-or-updates the deepcab-retrain-daily
# scheduler job firing at 02:00 UTC.
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
