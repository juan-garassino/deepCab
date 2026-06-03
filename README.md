# deepCab

NYC taxi-fare MLOps **learning hub**. One repo demonstrating, end-to-end:

- 6 model backends behind one sklearn-compatible interface (TF MLP · Torch MLP · XGBoost · LightGBM · CatBoost · FT-Transformer)
- Pydantic schemas as the **single source of truth** for FastAPI bodies, OpenAI agent tool params, Hydra structured configs, and MLflow param logs
- Modern data stack: Polars + DuckDB + Hive-partitioned Parquet + Pandera + content-hash SQLite lineage
- Hydra configs · Optuna HPO · time-series CV · Adaptive Conformal Inference · ONNX serving with INT8 quant · SHAP explainability
- FastAPI: routers, DI, middleware, BG tasks, SSE, X-API-Key gate
- Hand-rolled OpenAI tool-calling agent: planner + executor split, MLflow-run memory, budget-capped self-improve loop, append-only JSONL trace
- Prefect 3 nightly retrain flow · MLflow 2.x with `@champion` / `@challenger` aliases · auto-emit MODEL_CARD.md per run
- Compose: core / obs / gpu / secrets · GitHub Actions CI + train-smoke matrix · OTel + Prometheus + Grafana provisioned

## Quickstart

```bash
make bootstrap         # uv sync + seed secrets + (optional) CSV→Parquet migrate
make run_train         # python -m deepCab.training.train backend=tf_mlp data=1k
make run_api           # API serves /predict immediately — autoloader rehydrates STATE from runs/LATEST
# silicon Macs: run `make actions_reinstall_silicon` before `make bootstrap` for tensorflow-macos wheels
```

Full stack via compose (files under `infra/compose/`):

```bash
make docker_up                # core (traefik + api + mlflow + postgres + minio + redis + prefect)
make docker_obs_up            # core + observability (+ otel + jaeger + prom + alertmanager + grafana + loki + promtail)
make docker_dev_up            # core + obs + dev extras (ngrok, pgadmin)
# or raw:
docker compose -f infra/compose/docker-compose.yml up -d
docker compose -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.obs.yml up -d
```

### Full obs stack with Slack alerts

First time only — set up local DNS + TLS so Traefik can route `*.deepcab.localhost`:

```bash
make hosts    # adds /etc/hosts entries (sudo)
make mkcert   # writes mkcert-issued cert into infra/secrets/
echo "https://hooks.slack.com/services/T0/B0/YOUR-WEBHOOK" > infra/secrets/slack_webhook_url
make docker_obs_up
```

Then hit `https://grafana.deepcab.localhost` (admin/admin) — Prometheus + Loki + Jaeger
datasources are auto-provisioned. Alertmanager posts to `#deepcab-ops` on rule
fires; the same Slack webhook receives `notify_alias_change` and `notify_flow_event`
messages from the in-process helper (`deepCab/obs/slack.py`).

## API endpoints

`http://localhost:8000/docs` for the live OpenAPI spec.

| Endpoint                | Auth      | Notes                                                                       |
|-------------------------|-----------|-----------------------------------------------------------------------------|
| `GET  /`                | open      | Service banner                                                              |
| `GET  /version`         | open      | Package version                                                             |
| `GET  /healthz`         | open      | Liveness                                                                    |
| `GET  /readyz`          | open      | Reports `model_loaded` + `backend_kind`                                      |
| `GET  /metrics`         | open      | Prometheus text format                                                       |
| `POST /predict`         | open      | Single row → fare + ACI interval (when calibrated)                          |
| `POST /predict/batch`   | open      | Batched                                                                      |
| `POST /predict/stream`  | open      | SSE per-row stream                                                           |
| `POST /explain`         | open      | SHAP attribution aggregated to 5 user-meaningful groups                     |
| `GET  /explain/summary` | open      | Cached global mean(|SHAP|) per group                                        |
| `POST /train`           | X-API-Key | BackgroundTask; returns `task_id`                                           |
| `GET  /train/{task_id}` | open      | Status (pending/running/succeeded/failed) + run_id                          |
| `POST /agent`           | X-API-Key | One-shot agent turn (SSE)                                                   |
| `POST /agent/improve`   | X-API-Key | Self-improve loop with `BudgetCap` (SSE)                                    |

### curl examples

```bash
KEY=$(cat infra/secrets/deepcab_api_key)
ROW='{"row":{"pickup_datetime":"2014-01-15T05:00:00","pickup_longitude":-73.97,"pickup_latitude":40.78,"dropoff_longitude":-73.99,"dropoff_latitude":40.74,"passenger_count":2}}'

# predict (after make run_train)
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d "$ROW"

# explain
curl -X POST http://localhost:8000/explain -H 'Content-Type: application/json' \
  -d "${ROW%\}}, \"mode\":\"per_row\"}"

# train with a different backend
curl -X POST http://localhost:8000/train -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"config":{"backend":{"kind":"lgbm","n_estimators":200},"data":{"size":"1k","validation_size":"1k"},"seed":42}}'
```

## Agent

```bash
# REPL — needs OPENAI_API_KEY in .env.dev or infra/secrets/openai_api_key
python -m deepCab.agent.cli
> /improve reduce val_mae below 3.0
```

Or via HTTP:

```bash
curl -X POST http://localhost:8000/agent/improve -H "X-API-Key: $KEY" -H 'Content-Type: application/json' -N \
  -d '{"goal":"reduce val_mae","budget":{"max_iters":3,"max_tool_calls":20,"max_usd":1.0}}'
```

Available tools: `preprocess · train · evaluate · predict · explain · tune · list_runs · compare_runs · propose_next_experiment`. Each is a Pydantic-validated function whose schema doubles as a FastAPI body — same source of truth.

## Hydra training

```bash
python -m deepCab.training.train backend=tf_mlp data=1k
python -m deepCab.training.train backend=xgb data=10k seed=7
# override any backend field
python -m deepCab.training.train backend=lgbm backend.num_leaves=63 backend.learning_rate=0.03
```

## Prefect flow (scheduled retrain)

```bash
make run_flow       # one-shot in-process
make flow_serve     # serves the nightly cron schedule
# Prefect UI: http://localhost:4200
```

## Observability

| UI            | URL (Traefik)                              | Direct (no Traefik)     | Login           |
|---------------|--------------------------------------------|-------------------------|-----------------|
| Traefik       | http://localhost:8080                      | http://localhost:8080   | — (dev)         |
| MLflow        | https://mlflow.deepcab.localhost           | —                       | —               |
| Prefect       | https://prefect.deepcab.localhost          | —                       | —               |
| Jaeger        | https://jaeger.deepcab.localhost           | http://localhost:16686  | —               |
| Grafana       | https://grafana.deepcab.localhost          | http://localhost:3000   | admin / admin   |
| Prometheus    | (no Traefik label)                         | http://localhost:9090   | —               |
| Alertmanager  | https://alertmanager.deepcab.localhost     | —                       | —               |
| MinIO Console | https://minio.deepcab.localhost            | http://localhost:9001   | minio / (file)  |
| pgAdmin (dev) | https://pgadmin.deepcab.localhost          | —                       | (file)          |

Grafana auto-provisions three dashboards: **deepCab — API** (latency p50/p95/p99, error rate), **deepCab — Agent** (tool calls, tokens, USD), **deepCab — Training** (runs, epoch duration). Promtail ships every container's stdout/stderr into Loki — visible under Grafana → Explore → Loki.

Alert rules in `infra/compose/conf/prometheus-rules.yml`: `HighErrorRate` (5xx > 5%), `ApiDown` (no scrape 2m), `LatencyP99High` (p99 > 1s for 10m). Alertmanager fans them out to Slack via `infra/secrets/slack_webhook_url`.

## GPU

```bash
make docker_gpu_up
# or raw:
docker compose -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.gpu.yml up -d --build api prefect-agent
# requires nvidia-container-toolkit on the host
```

## Tests

```bash
make actions_reinstall        # uv sync --extra dev
uv run pytest tests/ -q       # ~100 tests, ~13 skipped (backends/onnx/openai gated)
```

## How to extend

- **New backend**: subclass `AbstractEstimator` in `deepCab/models/<name>.py`, add a `*Config` to `deepCab/schemas/config.py` (`kind: Literal["..."]`), register in `deepCab/models/_kinds.py::BACKENDS`. ONNX export goes in `deepCab/models/onnx_export.py`. One line per backend everywhere.
- **New agent tool**: add `(InputModel, fn, "description")` to `deepCab/agent/tools.py::_TOOLS`. The OpenAI schema + FastAPI body materialize automatically.
- **New endpoint**: drop a router under `deepCab/api/routers/` and include it in `deepCab/api/app.py::create_app`.

## Architecture

See `/Users/juan-garassino/Code/005-products/006-deep-projects/001-deepCab/CLAUDE.md` for the per-module tree and design decisions per phase.

## Production deferred

These are intentionally **not** in the MVP — documented + scoped post-MVP:

- Multi-worker uvicorn (STATE → Redis migration)
- Drift detection (Evidently / NannyML on `/monitor`)
- Logged inference table for online-feedback drift
- Per-user rate-limiting + quotas
- LLM-driven `propose_next_experiment`
- Tune-across-backends
