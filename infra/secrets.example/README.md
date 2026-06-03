# `infra/secrets/` — local dev defaults

`infra/compose/docker-compose.yml` mounts each file in `infra/secrets/` at
`/run/secrets/<name>` inside the corresponding service. The directory is
**gitignored**; this `infra/secrets.example/` is the template.

To seed defaults, copy each example file:

```
cp infra/secrets.example/<file> infra/secrets/<file_without_example_suffix>
```

## First-run setup

```bash
mkdir -p infra/secrets
echo "dev-pg-pw"      > infra/secrets/postgres_password
echo "dev-minio-pw"   > infra/secrets/minio_root_password
openssl rand -hex 16  > infra/secrets/deepcab_api_key
echo "sk-..."         > infra/secrets/openai_api_key
chmod 600 infra/secrets/*
```

## What references them

| Secret               | Consumed by                                                              |
|----------------------|--------------------------------------------------------------------------|
| `postgres_password`  | `postgres` (POSTGRES_PASSWORD_FILE), `mlflow` (URI inline), `prefect`    |
| `minio_root_password`| `minio` (MINIO_ROOT_PASSWORD_FILE), `mlflow` (AWS_SECRET_ACCESS_KEY)     |
| `deepcab_api_key`    | `api` (X-API-Key gate on /train and /agent)                              |
| `openai_api_key`     | `api` (OpenAI SDK; fallback to deepcab_api_key for backward compat)      |

## Production (swarm / k8s)

Replace `file: ../secrets/<name>` with `external: true` in
`infra/compose/docker-compose.yml` and provision via your cluster's secret store. The `_maybe_read_file` helper
in `deepCab/schemas/settings.py` resolves any `file:/run/secrets/<name>`
env value the same way regardless of source.
