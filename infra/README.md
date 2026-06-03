# infra/

All container, compose, and cloud deployment artifacts.

## Layers

| File | Run with | Purpose |
|---|---|---|
| `compose/docker-compose.yml` | `make docker_up` | Core: traefik + postgres + minio + redis + mlflow + prefect + prefect-agent + api |
| `compose/docker-compose.obs.yml` | `make docker_obs_up` | + otel + jaeger + prom + alertmanager + grafana + loki + promtail |
| `compose/docker-compose.dev.yml` | `make docker_dev_up` | + ngrok + pgadmin + react-dev (never in CI/prod) |
| `compose/docker-compose.gpu.yml` | `make docker_gpu_up` | GPU override |

## First-time setup

```bash
cp -r secrets.example/* secrets/
make hosts                # appends *.deepcab.localhost to /etc/hosts (no-op on most browsers)
mkcert -install           # local CA for TLS — see https://github.com/FiloSottile/mkcert
mkcert -cert-file secrets/traefik-cert.pem -key-file secrets/traefik-key.pem '*.deepcab.localhost'
```

## GCP deploys

See `gcp/workload-identity/README.md` for the one-time Workload Identity Federation bootstrap, then push a `v*` tag to fire `.github/workflows/deploy-cloud-run.yml`.
