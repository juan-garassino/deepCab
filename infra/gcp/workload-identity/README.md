# Workload Identity Federation bootstrap

One-time setup so GitHub Actions can deploy to GCP **without long-lived JSON keys**.

## Prerequisites

- `gcloud` CLI authenticated as a project owner.
- A GCP project with billing enabled.
- The deepCab repo on GitHub (note the org + repo name).

## Variables

```bash
export PROJECT=deepcab-prod-xxxx          # your GCP project id
export PROJECT_NUMBER=123456789012        # gcloud projects describe $PROJECT --format='value(projectNumber)'
export GH_OWNER=juan-garassino            # your github org/user
export GH_REPO=deepCab                    # your github repo (the parent — adjust if monorepo)
export REGION=us-central1
```

## Steps

```bash
# 1. Enable required APIs
gcloud services enable \
  iamcredentials.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  container.googleapis.com \
  iam.googleapis.com \
  --project=$PROJECT

# 2. Create Artifact Registry repo for the api image
gcloud artifacts repositories create deepcab \
  --repository-format=docker \
  --location=$REGION \
  --description="deepCab images" \
  --project=$PROJECT

# 3. Create the Workload Identity Pool + Provider for GitHub Actions
gcloud iam workload-identity-pools create github-pool \
  --project=$PROJECT --location=global \
  --display-name="GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --project=$PROJECT --location=global \
  --workload-identity-pool=github-pool \
  --display-name="GitHub Actions" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner == '$GH_OWNER'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# 4. Create the deployer service account
gcloud iam service-accounts create deepcab-deployer \
  --project=$PROJECT \
  --display-name="deepCab GH Actions deployer"

# 5. Grant deploy roles
for ROLE in roles/run.admin \
            roles/container.admin \
            roles/artifactregistry.writer \
            roles/iam.serviceAccountUser \
            roles/storage.objectAdmin ; do
  gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:deepcab-deployer@$PROJECT.iam.gserviceaccount.com" \
    --role="$ROLE"
done

# 6. Bind GitHub repo to the deployer SA via WIF
gcloud iam service-accounts add-iam-policy-binding \
  deepcab-deployer@$PROJECT.iam.gserviceaccount.com \
  --project=$PROJECT \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/$GH_OWNER/$GH_REPO"

# 7. Create the runtime service account used by Cloud Run / GKE
gcloud iam service-accounts create deepcab-runtime \
  --project=$PROJECT \
  --display-name="deepCab runtime SA"

for ROLE in roles/storage.objectViewer \
            roles/aiplatform.user \
            roles/logging.logWriter \
            roles/monitoring.metricWriter ; do
  gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:deepcab-runtime@$PROJECT.iam.gserviceaccount.com" \
    --role="$ROLE"
done

# 8. Allow the deployer SA to impersonate the runtime SA
gcloud iam service-accounts add-iam-policy-binding \
  deepcab-runtime@$PROJECT.iam.gserviceaccount.com \
  --project=$PROJECT \
  --role=roles/iam.serviceAccountUser \
  --member="serviceAccount:deepcab-deployer@$PROJECT.iam.gserviceaccount.com"

# 9. Print the values you need for GitHub variables
echo ""
echo "== Set these as repo variables (gh variable set) =="
echo "GCP_PROJECT=$PROJECT"
echo "GCP_PROJECT_NUMBER=$PROJECT_NUMBER"
echo "GCP_REGION=$REGION"
echo "GCP_WIF_PROVIDER=projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
echo "GCP_DEPLOYER_SA=deepcab-deployer@$PROJECT.iam.gserviceaccount.com"
echo ""
echo "== Set as repo SECRET =="
echo "SLACK_WEBHOOK_URL=<your incoming webhook URL>"
```

## GitHub variables + secrets to set

```bash
gh variable set GCP_PROJECT --body "$PROJECT"
gh variable set GCP_PROJECT_NUMBER --body "$PROJECT_NUMBER"
gh variable set GCP_REGION --body "$REGION"
gh variable set GCP_WIF_PROVIDER --body "projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
gh variable set GCP_DEPLOYER_SA --body "deepcab-deployer@$PROJECT.iam.gserviceaccount.com"
gh variable set MLFLOW_URL --body "https://mlflow.example.com"   # adjust to your MLflow deployment

gh secret set SLACK_WEBHOOK_URL    # paste the URL when prompted
```

## Verification

After the first push of a `v*` tag, watch `.github/workflows/deploy-cloud-run.yml` succeed. If WIF is misconfigured the auth step fails with a clear message like "Permission 'iam.serviceAccounts.getAccessToken' denied".

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
