#!/usr/bin/env bash
# Workload Identity Federation bootstrap for deepCab GitHub Actions → GCP.
#
# One-time setup. Idempotent-ish (gcloud commands will error if the resource
# already exists — that's fine, re-run after fixing the offender).
#
# Required env vars (export before running):
#   PROJECT         e.g. deepcab-prod-xxxx
#   PROJECT_NUMBER  e.g. 123456789012  (gcloud projects describe $PROJECT --format='value(projectNumber)')
#   GH_OWNER        e.g. juan-garassino
#   GH_REPO         e.g. deepCab
#   REGION          e.g. us-central1

set -euo pipefail

for var in PROJECT PROJECT_NUMBER GH_OWNER GH_REPO REGION; do
  if [[ -z "${!var:-}" ]]; then
    echo "ERROR: \$$var is not set. Export PROJECT, PROJECT_NUMBER, GH_OWNER, GH_REPO, REGION first." >&2
    exit 1
  fi
done

echo "==> Bootstrapping WIF for project=$PROJECT region=$REGION repo=$GH_OWNER/$GH_REPO"

# 1. Enable required APIs
gcloud services enable \
  iamcredentials.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  container.googleapis.com \
  iam.googleapis.com \
  --project="$PROJECT"

# 2. Create Artifact Registry repo for the api image
gcloud artifacts repositories create deepcab \
  --repository-format=docker \
  --location="$REGION" \
  --description="deepCab images" \
  --project="$PROJECT" || echo "  (artifact registry 'deepcab' may already exist — continuing)"

# 3. Create the Workload Identity Pool + Provider for GitHub Actions
gcloud iam workload-identity-pools create github-pool \
  --project="$PROJECT" --location=global \
  --display-name="GitHub Actions Pool" || echo "  (pool 'github-pool' may already exist — continuing)"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --project="$PROJECT" --location=global \
  --workload-identity-pool=github-pool \
  --display-name="GitHub Actions" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner == '$GH_OWNER'" \
  --issuer-uri="https://token.actions.githubusercontent.com" || echo "  (provider 'github-provider' may already exist — continuing)"

# 4. Create the deployer service account
gcloud iam service-accounts create deepcab-deployer \
  --project="$PROJECT" \
  --display-name="deepCab GH Actions deployer" || echo "  (SA 'deepcab-deployer' may already exist — continuing)"

# 5. Grant deploy roles
for ROLE in roles/run.admin \
            roles/container.admin \
            roles/artifactregistry.writer \
            roles/iam.serviceAccountUser \
            roles/storage.objectAdmin ; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:deepcab-deployer@$PROJECT.iam.gserviceaccount.com" \
    --role="$ROLE" \
    --condition=None >/dev/null
done

# 6. Bind GitHub repo to the deployer SA via WIF
gcloud iam service-accounts add-iam-policy-binding \
  "deepcab-deployer@$PROJECT.iam.gserviceaccount.com" \
  --project="$PROJECT" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/$GH_OWNER/$GH_REPO" \
  --condition=None >/dev/null

# 7. Create the runtime service account used by Cloud Run / GKE
gcloud iam service-accounts create deepcab-runtime \
  --project="$PROJECT" \
  --display-name="deepCab runtime SA" || echo "  (SA 'deepcab-runtime' may already exist — continuing)"

for ROLE in roles/storage.objectViewer \
            roles/aiplatform.user \
            roles/logging.logWriter \
            roles/monitoring.metricWriter ; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:deepcab-runtime@$PROJECT.iam.gserviceaccount.com" \
    --role="$ROLE" \
    --condition=None >/dev/null
done

# 8. Allow the deployer SA to impersonate the runtime SA
gcloud iam service-accounts add-iam-policy-binding \
  "deepcab-runtime@$PROJECT.iam.gserviceaccount.com" \
  --project="$PROJECT" \
  --role=roles/iam.serviceAccountUser \
  --member="serviceAccount:deepcab-deployer@$PROJECT.iam.gserviceaccount.com" \
  --condition=None >/dev/null

# 9. Print the values you need for GitHub variables
cat <<EOF

== Set these as repo variables (gh variable set) ==
GCP_PROJECT=$PROJECT
GCP_PROJECT_NUMBER=$PROJECT_NUMBER
GCP_REGION=$REGION
GCP_WIF_PROVIDER=projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider
GCP_DEPLOYER_SA=deepcab-deployer@$PROJECT.iam.gserviceaccount.com

== Set as repo SECRET ==
SLACK_WEBHOOK_URL=<your incoming webhook URL>

Done.
EOF
