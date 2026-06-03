# Sub-project B — Real production-feel container additions

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Traefik (reverse proxy + local TLS), Loki + Promtail (logs), Alertmanager (Prom→Slack), pgAdmin (dev), ngrok (dev), and a tiny in-process Slack helper. Stack reaches 18 containers with real production posture.

**Architecture:** Traefik fronts every HTTP service via `*.deepcab.localhost` hostnames + mkcert local TLS. Logs flow `container → promtail (docker socket) → loki → grafana`. Prom alerts route via Alertmanager → Slack webhook. Runtime Slack notifications from MLflow alias changes and Prefect flow events go through a small `deepCab/obs/slack.py` helper.

**Tech Stack:** Traefik v3.1, Loki/Promtail 3.2, Alertmanager v0.27, pgAdmin 4, ngrok agent, FastAPI (for the test endpoint that posts Slack), Python `requests`.

**Reference:** [Design spec §4.2, §4.3, §8](../specs/2026-06-03-deepcab-gcp-infra-and-audit-design.md).

**Prerequisite:** Sub-project A landed (`infra/` exists).

---

## File map

| Action | Path | Purpose |
|---|---|---|
| Create | `infra/compose/docker-compose.dev.yml` | dev-only services (ngrok, pgadmin, react-dev stub) |
| Modify | `infra/compose/docker-compose.yml` | add traefik service; add traefik labels to api/mlflow/prefect/minio |
| Modify | `infra/compose/docker-compose.obs.yml` | add loki, promtail, alertmanager; add traefik labels to grafana/jaeger |
| Create | `infra/compose/conf/traefik/traefik.yml` | static config |
| Create | `infra/compose/conf/traefik/dynamic.yml` | dynamic config (cert, middlewares) |
| Create | `infra/compose/conf/loki-config.yaml` | Loki config |
| Create | `infra/compose/conf/promtail-config.yaml` | Promtail Docker SD config |
| Create | `infra/compose/conf/alertmanager.yml` | Slack receiver |
| Create | `infra/compose/conf/prometheus-rules.yml` | API alert rules |
| Modify | `infra/compose/conf/prometheus.yml` | reference alertmanager + rules |
| Create | `infra/secrets.example/slack_webhook_url.example` | placeholder |
| Create | `infra/secrets.example/gcp_sa_key.example` | placeholder |
| Create | `infra/secrets.example/ngrok_authtoken.example` | placeholder |
| Create | `infra/secrets.example/pgadmin_password.example` | placeholder |
| Create | `infra/secrets.example/traefik_acme_email.example` | placeholder |
| Create | `deepCab/obs/slack.py` | in-process Slack notifier |
| Modify | `deepCab/registry/dispatcher.py` | call `slack.notify_alias_change` from `set_alias` |
| Modify | `deepCab/flow_v2/retrain.py` | call `slack.notify_flow_event` on start/success/failure |
| Modify | `deepCab/schemas/settings.py` | add `obs.slack_webhook_url` field |
| Create | `tests/obs/test_slack.py` | unit tests for the helper |
| Modify | `Makefile` | add `make hosts`, `make mkcert`, `make docker_dev_up`, `make docker_dev_down` |
| Modify | `CLAUDE.md`, `README.md`, `CONTRIBUTING.md` | document new services + setup |

---

## Task B1: Add Traefik to core compose

**Files:**
- Modify: `infra/compose/docker-compose.yml`
- Create: `infra/compose/conf/traefik/traefik.yml`
- Create: `infra/compose/conf/traefik/dynamic.yml`

- [ ] **Step 1: Write `infra/compose/conf/traefik/traefik.yml` (static config)**

```yaml
api:
  dashboard: true
  insecure: true   # dev only; dashboard at :8080

entryPoints:
  web:
    address: ":80"
  websecure:
    address: ":443"

providers:
  docker:
    exposedByDefault: false
    network: deepcab_default
  file:
    directory: /etc/traefik/dynamic
    watch: true

# Let's Encrypt enabled in staging/prod only; locally we use mkcert via dynamic config.
certificatesResolvers:
  letsencrypt:
    acme:
      email: ${TRAEFIK_ACME_EMAIL:-noreply@example.com}
      storage: /letsencrypt/acme.json
      httpChallenge:
        entryPoint: web
```

- [ ] **Step 2: Write `infra/compose/conf/traefik/dynamic.yml` (mkcert TLS)**

```yaml
tls:
  certificates:
    - certFile: /certs/traefik-cert.pem
      keyFile: /certs/traefik-key.pem
  stores:
    default:
      defaultCertificate:
        certFile: /certs/traefik-cert.pem
        keyFile: /certs/traefik-key.pem
```

- [ ] **Step 3: Add Traefik service to `docker-compose.yml`**

Insert before `postgres`:

```yaml
  traefik:
    image: traefik:v3.1
    restart: unless-stopped
    command:
      - "--configFile=/etc/traefik/traefik.yml"
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"    # dashboard
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./conf/traefik/traefik.yml:/etc/traefik/traefik.yml:ro
      - ./conf/traefik/dynamic.yml:/etc/traefik/dynamic/dynamic.yml:ro
      - ../secrets/traefik-cert.pem:/certs/traefik-cert.pem:ro
      - ../secrets/traefik-key.pem:/certs/traefik-key.pem:ro
      - traefik_acme:/letsencrypt
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.traefik.rule=Host(`traefik.deepcab.localhost`)"
      - "traefik.http.routers.traefik.tls=true"
      - "traefik.http.routers.traefik.service=api@internal"
```

Add `traefik_acme:` under the top-level `volumes:` block.

- [ ] **Step 4: Add labels to api, mlflow, prefect, minio**

For each, add a `labels:` block under the service. Example for `api`:

```yaml
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=Host(`api.deepcab.localhost`)"
      - "traefik.http.routers.api.tls=true"
      - "traefik.http.services.api.loadbalancer.server.port=8000"
```

Repeat with host `mlflow.deepcab.localhost` (port 5000), `prefect.deepcab.localhost` (port 4200), `minio.deepcab.localhost` (port 9001 — the console).

Remove the host-side `ports:` blocks for `mlflow` and `prefect` (Traefik now routes them); keep `minio: 9000` and `9001` for direct S3 access during testing.

- [ ] **Step 5: Verify compose syntax**

```bash
docker compose -f infra/compose/docker-compose.yml config --quiet && echo OK
```

---

## Task B2: mkcert helper + hosts target

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add make targets**

Append to `Makefile`:

```make
HOSTS_LINE := 127.0.0.1 api.deepcab.localhost mlflow.deepcab.localhost prefect.deepcab.localhost grafana.deepcab.localhost jaeger.deepcab.localhost minio.deepcab.localhost traefik.deepcab.localhost pgadmin.deepcab.localhost app.deepcab.localhost

hosts:
	@if ! grep -q "deepcab.localhost" /etc/hosts; then \
	  echo "$(HOSTS_LINE)" | sudo tee -a /etc/hosts ; \
	else \
	  echo "✓ /etc/hosts already configured" ; \
	fi

mkcert:
	@command -v mkcert >/dev/null || { echo "install mkcert first: brew install mkcert"; exit 1; }
	mkcert -install
	mkcert -cert-file infra/secrets/traefik-cert.pem -key-file infra/secrets/traefik-key.pem '*.deepcab.localhost' localhost
	@echo "✓ TLS cert in infra/secrets/"
```

Chrome/Firefox/Safari resolve `*.localhost` to 127.0.0.1 natively, so `make hosts` is mostly a safety net for curl + Edge.

---

## Task B3: Add Loki + Promtail + Alertmanager to obs compose

**Files:**
- Modify: `infra/compose/docker-compose.obs.yml`
- Create: `infra/compose/conf/loki-config.yaml`
- Create: `infra/compose/conf/promtail-config.yaml`
- Create: `infra/compose/conf/alertmanager.yml`
- Create: `infra/compose/conf/prometheus-rules.yml`
- Modify: `infra/compose/conf/prometheus.yml`

- [ ] **Step 1: Write `infra/compose/conf/loki-config.yaml`**

```yaml
auth_enabled: false
server:
  http_listen_port: 3100
common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory
schema_config:
  configs:
    - from: 2024-01-01
      store: tsdb
      object_store: filesystem
      schema: v13
      index:
        prefix: index_
        period: 24h
limits_config:
  reject_old_samples: false
  retention_period: 168h
```

- [ ] **Step 2: Write `infra/compose/conf/promtail-config.yaml`**

```yaml
server:
  http_listen_port: 9080
positions:
  filename: /tmp/positions.yaml
clients:
  - url: http://loki:3100/loki/api/v1/push
scrape_configs:
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '/(.*)'
        target_label: container
      - source_labels: ['__meta_docker_container_log_stream']
        target_label: stream
```

- [ ] **Step 3: Write `infra/compose/conf/alertmanager.yml`**

```yaml
global:
  resolve_timeout: 5m

route:
  receiver: slack
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 1h

receivers:
  - name: slack
    slack_configs:
      - api_url_file: /etc/alertmanager/slack_webhook_url
        channel: '#deepcab-ops'
        send_resolved: true
        title: "[alert] {{ .CommonLabels.alertname }}"
        text: |
          {{ range .Alerts }}
          *severity:* {{ .Labels.severity }}
          *summary:* {{ .Annotations.summary }}
          *description:* {{ .Annotations.description }}
          {{ end }}
```

- [ ] **Step 4: Write `infra/compose/conf/prometheus-rules.yml`**

```yaml
groups:
  - name: deepcab-api
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) by (job) / sum(rate(http_requests_total[5m])) by (job) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "5xx rate above 5%"
          description: "{{ $labels.job }} returned {{ printf \"%.2f\" $value }} 5xx-rate for 5m."

      - alert: ApiDown
        expr: up{job="deepcab-api"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "deepcab-api is unreachable"
          description: "prometheus has not scraped /metrics for 2 minutes."

      - alert: LatencyP99High
        expr: histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, job)) > 1.0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "p99 latency above 1s"
          description: "p99 = {{ printf \"%.2f\" $value }}s for 10m on {{ $labels.job }}."
```

- [ ] **Step 5: Update `infra/compose/conf/prometheus.yml`**

Add alerting section (top-level) and rule_files reference:

```yaml
rule_files:
  - /etc/prometheus/rules.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

Keep existing `scrape_configs:`.

- [ ] **Step 6: Add three services to `docker-compose.obs.yml`**

```yaml
  loki:
    image: grafana/loki:3.2.0
    restart: unless-stopped
    command: -config.file=/etc/loki/local-config.yaml
    volumes:
      - ./conf/loki-config.yaml:/etc/loki/local-config.yaml:ro
      - loki_data:/loki

  promtail:
    image: grafana/promtail:3.2.0
    restart: unless-stopped
    command: -config.file=/etc/promtail/config.yml
    volumes:
      - ./conf/promtail-config.yaml:/etc/promtail/config.yml:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    depends_on:
      - loki

  alertmanager:
    image: prom/alertmanager:v0.27.0
    restart: unless-stopped
    command:
      - --config.file=/etc/alertmanager/alertmanager.yml
    secrets:
      - slack_webhook_url
    volumes:
      - ./conf/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager_data:/alertmanager
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.alertmanager.rule=Host(`alertmanager.deepcab.localhost`)"
      - "traefik.http.routers.alertmanager.tls=true"
      - "traefik.http.services.alertmanager.loadbalancer.server.port=9093"
```

In the obs compose, point alertmanager to the secret file. Add a top-level `secrets:` block:

```yaml
secrets:
  slack_webhook_url:
    file: ../secrets/slack_webhook_url
```

Then in the `alertmanager` service: `secrets: [slack_webhook_url]` and the file lands at `/run/secrets/slack_webhook_url`. Update the path in `alertmanager.yml`:

```yaml
api_url_file: /run/secrets/slack_webhook_url
```

Add to volumes:
```yaml
volumes:
  loki_data:
  alertmanager_data:
```

- [ ] **Step 7: Update prometheus service to mount the rules file**

In the existing `prometheus:` service in the obs compose:

```yaml
    volumes:
      - ./conf/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./conf/prometheus-rules.yml:/etc/prometheus/rules.yml:ro
      - prometheus_data:/prometheus
```

Add `prometheus_data:` to volumes block.

- [ ] **Step 8: Update grafana provisioning to include Loki datasource**

Edit `infra/compose/conf/grafana/provisioning/datasources/datasources.yml` (create file if missing):

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
```

- [ ] **Step 9: Verify**

```bash
docker compose -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.obs.yml config --quiet && echo OK
```

---

## Task B4: Create dev compose with ngrok + pgadmin

**Files:**
- Create: `infra/compose/docker-compose.dev.yml`

- [ ] **Step 1: Write the file**

```yaml
name: deepcab

secrets:
  ngrok_authtoken:
    file: ../secrets/ngrok_authtoken
  pgadmin_password:
    file: ../secrets/pgadmin_password

services:
  ngrok:
    image: ngrok/ngrok:latest
    restart: unless-stopped
    command: http api:8000 --log=stdout
    secrets:
      - ngrok_authtoken
    environment:
      NGROK_AUTHTOKEN_FILE: /run/secrets/ngrok_authtoken
    depends_on:
      - api
    labels:
      - "traefik.enable=false"

  pgadmin:
    image: dpage/pgadmin4:latest
    restart: unless-stopped
    secrets:
      - pgadmin_password
    environment:
      PGADMIN_DEFAULT_EMAIL: dev@deepcab.localhost
      PGADMIN_DEFAULT_PASSWORD_FILE: /run/secrets/pgadmin_password
      PGADMIN_CONFIG_SERVER_MODE: "False"
    depends_on:
      - postgres
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.pgadmin.rule=Host(`pgadmin.deepcab.localhost`)"
      - "traefik.http.routers.pgadmin.tls=true"
      - "traefik.http.services.pgadmin.loadbalancer.server.port=80"
    volumes:
      - pgadmin_data:/var/lib/pgadmin

volumes:
  pgadmin_data:
```

(Note: `react-dev` service gets added by Sub-project D, not here, to avoid building React without the package.json yet.)

- [ ] **Step 2: Add Makefile target**

```make
docker_dev_up:
	docker compose -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.obs.yml -f infra/compose/docker-compose.dev.yml up -d

docker_dev_down:
	docker compose -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.obs.yml -f infra/compose/docker-compose.dev.yml down
```

- [ ] **Step 3: Verify**

```bash
docker compose -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.obs.yml -f infra/compose/docker-compose.dev.yml config --quiet && echo OK
```

---

## Task B5: Add missing secret example files

**Files:**
- Create: `infra/secrets.example/slack_webhook_url.example`
- Create: `infra/secrets.example/gcp_sa_key.example`
- Create: `infra/secrets.example/ngrok_authtoken.example`
- Create: `infra/secrets.example/pgadmin_password.example`
- Create: `infra/secrets.example/traefik_acme_email.example`

- [ ] **Step 1: Create placeholder files**

```bash
echo 'https://hooks.slack.com/services/REPLACE/ME/PLEASE' > infra/secrets.example/slack_webhook_url.example
echo '{"type":"service_account","project_id":"REPLACE","private_key":"-----BEGIN ...","client_email":"sa@REPLACE.iam.gserviceaccount.com"}' > infra/secrets.example/gcp_sa_key.example
echo 'REPLACE_WITH_YOUR_NGROK_AUTHTOKEN' > infra/secrets.example/ngrok_authtoken.example
echo 'REPLACE_WITH_STRONG_PASSWORD' > infra/secrets.example/pgadmin_password.example
echo 'you@example.com' > infra/secrets.example/traefik_acme_email.example
```

- [ ] **Step 2: Update `infra/secrets.example/README.md`**

Append a table listing each new secret + its provider/source.

---

## Task B6: Write `deepCab/obs/slack.py` (TDD)

**Files:**
- Create: `deepCab/obs/slack.py`
- Create: `tests/obs/test_slack.py`

- [ ] **Step 1: Write the failing test**

Create `tests/obs/__init__.py` (empty) and `tests/obs/test_slack.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from deepCab.obs import slack


def test_post_noop_when_webhook_unset(monkeypatch):
    monkeypatch.setattr(slack, "_webhook_url", lambda: None)
    with patch("requests.post") as mock_post:
        slack.post("hello", tag="ci")
    mock_post.assert_not_called()


def test_post_formats_with_tag(monkeypatch):
    monkeypatch.setattr(slack, "_webhook_url", lambda: "https://hooks.slack.com/abc")
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        slack.post("training started", tag="flow", extra={"run_id": "r-7"})

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "https://hooks.slack.com/abc"
    payload = kwargs["json"]
    assert payload["text"].startswith("[flow]")
    assert "training started" in payload["text"]
    assert "r-7" in payload["text"]


def test_post_swallows_network_errors(monkeypatch):
    monkeypatch.setattr(slack, "_webhook_url", lambda: "https://hooks.slack.com/abc")
    with patch("requests.post", side_effect=Exception("boom")):
        slack.post("x", tag="ci")   # must not raise


def test_notify_alias_change_uses_mlflow_tag(monkeypatch):
    monkeypatch.setattr(slack, "_webhook_url", lambda: "https://hooks.slack.com/abc")
    with patch("deepCab.obs.slack.post") as mock_post:
        slack.notify_alias_change(model="deepcab", alias="champion", version="3")
    mock_post.assert_called_once()
    kwargs = mock_post.call_args.kwargs
    assert kwargs["tag"] == "mlflow"


def test_notify_flow_event_uses_flow_tag(monkeypatch):
    monkeypatch.setattr(slack, "_webhook_url", lambda: "https://hooks.slack.com/abc")
    with patch("deepCab.obs.slack.post") as mock_post:
        slack.notify_flow_event(flow="retrain", state="success", run_id="r-1")
    kwargs = mock_post.call_args.kwargs
    assert kwargs["tag"] == "flow"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/obs/test_slack.py -v
```

Expected: ImportError on `from deepCab.obs import slack`.

- [ ] **Step 3: Write `deepCab/obs/slack.py`**

```python
from __future__ import annotations

import logging
from typing import Any, Mapping

import requests

from deepCab.schemas.settings import get_settings

log = logging.getLogger(__name__)


def _webhook_url() -> str | None:
    s = get_settings()
    return getattr(s.obs, "slack_webhook_url", None) or None


def post(text: str, *, tag: str, extra: Mapping[str, Any] | None = None) -> None:
    url = _webhook_url()
    if not url:
        return

    body = f"[{tag}] {text}"
    if extra:
        body += " — " + " ".join(f"{k}={v}" for k, v in extra.items())
    payload = {"text": body}

    try:
        r = requests.post(url, json=payload, timeout=3)
        if r.status_code >= 300:
            log.warning("slack webhook returned %s: %s", r.status_code, r.text[:200])
    except Exception as exc:
        log.warning("slack webhook failed: %s", exc)


def notify_alias_change(*, model: str, alias: str, version: str) -> None:
    post(
        f"alias `@{alias}` → {model} v{version}",
        tag="mlflow",
        extra={"model": model, "alias": alias, "version": version},
    )


def notify_flow_event(*, flow: str, state: str, run_id: str) -> None:
    post(
        f"flow `{flow}` → {state}",
        tag="flow",
        extra={"run_id": run_id},
    )
```

- [ ] **Step 4: Add `obs.slack_webhook_url` to settings**

Open `deepCab/schemas/settings.py`. Find the `ObsSettings` class (or similar — there should be an obs sub-section). Add:

```python
class ObsSettings(BaseSettings):
    # ... existing fields ...
    slack_webhook_url: str | None = None

    model_config = SettingsConfigDict(env_prefix="OBS_", env_file=_env_file(), extra="ignore")
```

If `_maybe_read_file` is in use for the `file:` URI pattern, ensure it applies to `slack_webhook_url` too (so `OBS_SLACK_WEBHOOK_URL=file:/run/secrets/slack_webhook_url` works in compose).

Also add `OBS_SLACK_WEBHOOK_URL=file:/run/secrets/slack_webhook_url` to the api service env in `docker-compose.yml`, and `OBS_SLACK_WEBHOOK_URL=file:/run/secrets/slack_webhook_url` to the prefect-agent service env.

- [ ] **Step 5: Run tests, verify pass**

```bash
uv run pytest tests/obs/test_slack.py -v
```

Expected: 5 passing.

- [ ] **Step 6: Run full test suite for regressions**

```bash
uv run pytest tests/ -q --ignore=tests/all
```

Expected: 103 + 5 = 108 passing.

---

## Task B7: Wire `slack.notify_alias_change` into `registry/dispatcher.py`

**Files:**
- Modify: `deepCab/registry/dispatcher.py`
- Create: `tests/registry/test_set_alias_slack.py`

- [ ] **Step 1: Write failing test**

```python
from unittest.mock import patch
from deepCab.registry import dispatcher


def test_set_alias_posts_slack(monkeypatch):
    with patch("deepCab.obs.slack.notify_alias_change") as mock_notify, \
         patch("deepCab.registry.dispatcher._set_alias_backend") as mock_backend:
        dispatcher.set_alias(model="deepcab", alias="champion", version="7")
    mock_backend.assert_called_once_with(model="deepcab", alias="champion", version="7")
    mock_notify.assert_called_once_with(model="deepcab", alias="champion", version="7")
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/registry/test_set_alias_slack.py -v
```

Expected: AttributeError on `_set_alias_backend` OR import error.

- [ ] **Step 3: Refactor `set_alias`**

In `deepCab/registry/dispatcher.py`, find `set_alias`. Rename the existing impl to `_set_alias_backend(*, model, alias, version)` (preserve its body unchanged). Then define a new `set_alias` that delegates + notifies:

```python
def set_alias(*, model: str, alias: str, version: str) -> None:
    _set_alias_backend(model=model, alias=alias, version=version)
    from deepCab.obs import slack  # local import to avoid circular at module load
    slack.notify_alias_change(model=model, alias=alias, version=version)
```

- [ ] **Step 4: Run target test, verify pass**

```bash
uv run pytest tests/registry/test_set_alias_slack.py -v
```

- [ ] **Step 5: Run full suite**

```bash
uv run pytest tests/ -q --ignore=tests/all
```

Expected: 109 passing (108 + 1).

---

## Task B8: Wire `slack.notify_flow_event` into `flow_v2/retrain.py`

**Files:**
- Modify: `deepCab/flow_v2/retrain.py`
- Create: `tests/flow_v2/test_retrain_slack.py`

- [ ] **Step 1: Write failing test**

```python
from unittest.mock import patch, MagicMock


def test_retrain_flow_emits_slack_on_success(monkeypatch):
    from deepCab.flow_v2 import retrain
    with patch("deepCab.obs.slack.notify_flow_event") as mock_notify, \
         patch.object(retrain, "_preprocess", return_value=MagicMock()), \
         patch.object(retrain, "_train",      return_value=MagicMock(run_id="r-9")), \
         patch.object(retrain, "_evaluate",   return_value={"val_mae": 3.4}):
        result = retrain.retrain_flow.fn()   # call the underlying function (Prefect 3 .fn)
    assert result is not None
    calls = [c.kwargs for c in mock_notify.call_args_list]
    assert any(c.get("state") == "running" for c in calls)
    assert any(c.get("state") == "success" for c in calls)


def test_retrain_flow_emits_slack_on_failure():
    from deepCab.flow_v2 import retrain
    with patch("deepCab.obs.slack.notify_flow_event") as mock_notify, \
         patch.object(retrain, "_preprocess", side_effect=RuntimeError("boom")):
        try:
            retrain.retrain_flow.fn()
        except RuntimeError:
            pass
    calls = [c.kwargs for c in mock_notify.call_args_list]
    assert any(c.get("state") == "failed" for c in calls)
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/flow_v2/test_retrain_slack.py -v
```

Expected: failure (no slack hooks yet).

- [ ] **Step 3: Edit `deepCab/flow_v2/retrain.py`**

Find `retrain_flow` (the `@flow` decorated function). Rename the existing per-step calls to `_preprocess`, `_train`, `_evaluate` if they aren't already named that way. Wrap the body:

```python
import uuid
from prefect import flow
from deepCab.obs import slack


@flow(name="deepcab-retrain")
def retrain_flow(cfg=None):
    run_id = str(uuid.uuid4())[:8]
    slack.notify_flow_event(flow="retrain", state="running", run_id=run_id)
    try:
        x = _preprocess(cfg)
        result = _train(x, cfg)
        metrics = _evaluate(result, cfg)
        slack.notify_flow_event(flow="retrain", state="success", run_id=result.run_id)
        return metrics
    except Exception:
        slack.notify_flow_event(flow="retrain", state="failed", run_id=run_id)
        raise
```

- [ ] **Step 4: Verify**

```bash
uv run pytest tests/flow_v2/test_retrain_slack.py tests/flow_v2 -v
```

- [ ] **Step 5: Run full suite**

```bash
uv run pytest tests/ -q --ignore=tests/all
```

Expected: 111 passing (109 + 2).

---

## Task B9: Bring up the stack end-to-end

**Files:** none — manual verification.

- [ ] **Step 1: Seed real secrets**

```bash
mkdir -p infra/secrets
cp infra/secrets.example/postgres_password.example infra/secrets/postgres_password
cp infra/secrets.example/minio_root_password.example infra/secrets/minio_root_password
cp infra/secrets.example/deepcab_api_key.example infra/secrets/deepcab_api_key
cp infra/secrets.example/openai_api_key.example infra/secrets/openai_api_key
echo "https://hooks.slack.com/services/T0/B0/PUT-A-REAL-TEST-WEBHOOK-HERE" > infra/secrets/slack_webhook_url
echo "REPLACE" > infra/secrets/pgadmin_password
echo "REPLACE_WITH_AUTHTOKEN" > infra/secrets/ngrok_authtoken
echo "you@example.com" > infra/secrets/traefik_acme_email
```

- [ ] **Step 2: Run mkcert**

```bash
make mkcert
ls infra/secrets/traefik-cert.pem infra/secrets/traefik-key.pem
```

- [ ] **Step 3: Bring up obs + core**

```bash
make docker_obs_up
sleep 30
docker compose -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.obs.yml ps
```

Expected: all 14 services healthy or "Up".

- [ ] **Step 4: Hit the routes**

```bash
curl -k https://api.deepcab.localhost/healthz       # api
curl -k https://mlflow.deepcab.localhost/           # mlflow
curl -k https://prefect.deepcab.localhost/          # prefect
curl -k https://grafana.deepcab.localhost/login     # grafana
curl -k https://jaeger.deepcab.localhost/           # jaeger
```

Each should return 200 (or a redirect / login page). Any error → `docker compose logs traefik` and check the Traefik dashboard at http://localhost:8080.

- [ ] **Step 5: Fire a test Slack alert**

In Grafana (Explore → Loki), confirm logs appear from every container.

In a separate shell, force an Alertmanager alert:

```bash
curl -k -X POST -H 'Content-Type: application/json' -d '[{"labels":{"alertname":"TestAlert","severity":"info"},"annotations":{"summary":"smoke test","description":"if you see this in Slack, B9 passes"}}]' http://localhost:9093/api/v2/alerts
```

Confirm Slack receives the message in `#deepcab-ops` within 30s.

- [ ] **Step 6: Bring it down**

```bash
make docker_obs_down
```

---

## Task B10: Update docs

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`

- [ ] **Step 1: CLAUDE.md — Architecture / docker-compose section**

In the new docker-compose block from Sub-project A, expand the obs line:

```markdown
- `infra/compose/docker-compose.obs.yml` — adds otel-collector, jaeger, prometheus, **alertmanager**, grafana, **loki**, **promtail**. Datasources: Prometheus + Loki + Jaeger.
```

Add a dev line:

```markdown
- `infra/compose/docker-compose.dev.yml` — adds ngrok, pgadmin (dev only — never CI/prod).
```

- [ ] **Step 2: CLAUDE.md — Required environment**

Add: `OBS_SLACK_WEBHOOK_URL` (file URI), `TRAEFIK_ACME_EMAIL`, `PGADMIN_PASSWORD`, `NGROK_AUTHTOKEN`. Note these as Docker-secret-mounted in compose.

- [ ] **Step 3: CLAUDE.md — Common commands**

Add `make hosts`, `make mkcert`, `make docker_dev_up`/`docker_dev_down`. Note that the first time, run `make hosts && make mkcert` before any `make docker_*_up`.

- [ ] **Step 4: README.md — Quickstart**

After the existing quickstart, add a short "If you want the full obs stack with Slack alerts" sub-section pointing at `make docker_obs_up`.

- [ ] **Step 5: CONTRIBUTING.md — Common snags**

Add snags:
- Traefik dashboard at http://localhost:8080
- `*.deepcab.localhost` resolution (Chrome/FF/Safari OK; curl needs `make hosts`)
- mkcert one-time install
- Loki "no data" → check Promtail logs

---

## Task B11: Commit Sub-project B

- [ ] **Step 1: Stage + status**

```bash
git add -A
git status --short
```

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(infra): add traefik, loki/promtail, alertmanager, pgadmin, ngrok, slack helper

Production-feel stack additions:
- Traefik v3.1 reverse proxy + mkcert local TLS (api/mlflow/prefect/grafana/jaeger/minio/pgadmin/alertmanager all on *.deepcab.localhost)
- Loki + Promtail for log aggregation (Docker SD)
- Alertmanager → Slack webhook for prom alerts
- docker-compose.dev.yml with ngrok + pgadmin (dev-only)
- deepCab/obs/slack.py — in-process Slack helper used by:
  - registry/dispatcher.py:set_alias → notify_alias_change (mlflow tag)
  - flow_v2/retrain.py → notify_flow_event (flow tag)
- 5 secret example files: slack_webhook_url, gcp_sa_key, ngrok_authtoken, pgadmin_password, traefik_acme_email
- prometheus-rules.yml: HighErrorRate, ApiDown, LatencyP99High
- Makefile: make hosts, make mkcert, make docker_dev_up/down

Sub-project B of the GCP infra design.

111 tests pass (was 103 + 8 new).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Done criteria

- [x] Traefik routes 9 hostnames under `*.deepcab.localhost` with mkcert TLS
- [x] Loki + Promtail aggregate container logs into Grafana
- [x] Alertmanager posts to Slack on test alert
- [x] pgAdmin reaches postgres via internal network
- [x] ngrok service starts when `docker-compose.dev.yml` is included
- [x] `deepCab/obs/slack.py` has 5 unit tests passing
- [x] `set_alias` and `retrain_flow` fire Slack notifications
- [x] `make docker_obs_up` brings up all 14 services healthy
- [x] CLAUDE.md, README.md, CONTRIBUTING.md updated
- [x] 111 tests pass
- [x] One commit
