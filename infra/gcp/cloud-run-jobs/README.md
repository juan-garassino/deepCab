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
