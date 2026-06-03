# `secrets/` — local dev defaults

`docker-compose.yml` mounts each file in `./secrets/` at `/run/secrets/<name>`
inside the corresponding service. The directory is **gitignored**; this
`secrets.example/` is the template.

## First-run setup

```bash
mkdir -p secrets
echo "dev-pg-pw"      > secrets/postgres_password
echo "dev-minio-pw"   > secrets/minio_root_password
openssl rand -hex 16  > secrets/deepcab_api_key
echo "sk-..."         > secrets/openai_api_key
chmod 600 secrets/*
```

## What references them

| Secret               | Consumed by                                                              |
|----------------------|--------------------------------------------------------------------------|
| `postgres_password`  | `postgres` (POSTGRES_PASSWORD_FILE), `mlflow` (URI inline), `prefect`    |
| `minio_root_password`| `minio` (MINIO_ROOT_PASSWORD_FILE), `mlflow` (AWS_SECRET_ACCESS_KEY)     |
| `deepcab_api_key`    | `api` (X-API-Key gate on /train and /agent)                              |
| `openai_api_key`     | `api` (OpenAI SDK; fallback to deepcab_api_key for backward compat)      |

## Production (swarm / k8s)

Replace `file: ./secrets/<name>` with `external: true` in `docker-compose.yml`
and provision via your cluster's secret store. The `_maybe_read_file` helper
in `deepCab/schemas/settings.py` resolves any `file:/run/secrets/<name>`
env value the same way regardless of source.
