# Sub-project A — `infra/` reorganization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all container/compose/secret artifacts under `infra/` without changing any container behavior. CI/tests continue to pass.

**Architecture:** Pure file move + Makefile/CLAUDE.md path updates. No new code, no new dependencies. `.github/workflows/` stays at repo root because GitHub discovers it there.

**Tech Stack:** Bash, docker compose, Make.

**Reference:** [Design spec §4.1](../specs/2026-06-03-deepcab-gcp-infra-and-audit-design.md#41-folder-layout-target-end-state).

---

## File map

| From | To |
|---|---|
| `Dockerfile` | `infra/docker/Dockerfile` |
| `.dockerignore` (if exists) | `infra/docker/.dockerignore` |
| `docker-compose.yml` | `infra/compose/docker-compose.yml` |
| `docker-compose.obs.yml` | `infra/compose/docker-compose.obs.yml` |
| `docker-compose.gpu.yml` | `infra/compose/docker-compose.gpu.yml` |
| `compose/otel-collector.yaml` | `infra/compose/conf/otel-collector.yaml` |
| `compose/prometheus.yml` | `infra/compose/conf/prometheus.yml` |
| `compose/grafana/` (tree) | `infra/compose/conf/grafana/` |
| `secrets.example/` (tree) | `infra/secrets.example/` |

New files: `infra/README.md`, `infra/.gitkeep`-shaped `secrets/` directory entry in `.gitignore`.

---

## Task A1: Create `infra/` skeleton

**Files:**
- Create: `infra/docker/`, `infra/compose/conf/`, `infra/gcp/`, `infra/secrets/`, `infra/secrets.example/`
- Create: `infra/README.md`

- [ ] **Step 1: Verify clean working tree**

```bash
cd 001-deepCab-api
git status --short
```

Expected: empty (or only this plan file).

- [ ] **Step 2: Create directory skeleton**

```bash
mkdir -p infra/docker infra/compose/conf infra/gcp infra/secrets infra/secrets.example
```

- [ ] **Step 3: Write `infra/README.md`**

```markdown
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
```

- [ ] **Step 4: Stage skeleton**

```bash
git add infra/README.md
```

(Empty dirs don't track; that's fine — they get populated next.)

---

## Task A2: Move Dockerfile + compose files

**Files:**
- Move: `Dockerfile` → `infra/docker/Dockerfile`
- Move: `docker-compose.yml`, `docker-compose.obs.yml`, `docker-compose.gpu.yml` → `infra/compose/`

- [ ] **Step 1: Move with git mv**

```bash
git mv Dockerfile infra/docker/Dockerfile
git mv docker-compose.yml infra/compose/docker-compose.yml
git mv docker-compose.obs.yml infra/compose/docker-compose.obs.yml
git mv docker-compose.gpu.yml infra/compose/docker-compose.gpu.yml
```

- [ ] **Step 2: Update Dockerfile context paths**

Open `infra/docker/Dockerfile`. The `COPY pyproject.toml uv.lock ./` and `COPY deepCab deepCab` lines stay the same — but the build context must now be the repo root (`001-deepCab-api/`). No edits needed to the Dockerfile itself; just remember the build invocation changes.

- [ ] **Step 3: Update compose file build contexts**

In all three compose files, change every:

```yaml
build:
  context: .
```

to:

```yaml
build:
  context: ../..
  dockerfile: infra/docker/Dockerfile
```

Find and edit `infra/compose/docker-compose.yml`:

```bash
grep -n "context: \." infra/compose/docker-compose.yml
```

For each match, replace per the pattern above. Repeat for `docker-compose.obs.yml` and `docker-compose.gpu.yml`.

- [ ] **Step 4: Update secret file paths in compose**

In `infra/compose/docker-compose.yml`, change every:

```yaml
file: ./secrets/postgres_password
```

to:

```yaml
file: ../secrets/postgres_password
```

(One level up from `infra/compose/` to reach `infra/secrets/`.)

Repeat for `minio_root_password`, `deepcab_api_key`, `openai_api_key`.

- [ ] **Step 5: Verify compose syntax**

```bash
docker compose -f infra/compose/docker-compose.yml config --quiet
docker compose -f infra/compose/docker-compose.obs.yml config --quiet
docker compose -f infra/compose/docker-compose.gpu.yml config --quiet
```

Expected: no output (success). Any error → re-read the path changes.

---

## Task A3: Move conf/ tree

**Files:**
- Move: `compose/otel-collector.yaml` → `infra/compose/conf/otel-collector.yaml`
- Move: `compose/prometheus.yml` → `infra/compose/conf/prometheus.yml`
- Move: `compose/grafana/` → `infra/compose/conf/grafana/`

- [ ] **Step 1: Move with git mv**

```bash
git mv compose/otel-collector.yaml infra/compose/conf/otel-collector.yaml
git mv compose/prometheus.yml infra/compose/conf/prometheus.yml
git mv compose/grafana infra/compose/conf/grafana
rmdir compose
```

- [ ] **Step 2: Update conf paths in `docker-compose.obs.yml`**

Find:

```yaml
- ./compose/otel-collector.yaml:/etc/otelcol/config.yaml:ro
- ./compose/prometheus.yml:/etc/prometheus/prometheus.yml:ro
- ./compose/grafana/provisioning:/etc/grafana/provisioning:ro
```

Replace with:

```yaml
- ./conf/otel-collector.yaml:/etc/otelcol/config.yaml:ro
- ./conf/prometheus.yml:/etc/prometheus/prometheus.yml:ro
- ./conf/grafana/provisioning:/etc/grafana/provisioning:ro
```

(Now relative to `infra/compose/` where the obs compose file lives.)

- [ ] **Step 3: Verify**

```bash
docker compose -f infra/compose/docker-compose.obs.yml config --quiet
```

Expected: success.

---

## Task A4: Move secrets.example

**Files:**
- Move: `secrets.example/` → `infra/secrets.example/`

- [ ] **Step 1: Move with git mv**

```bash
git mv secrets.example/README.md infra/secrets.example/README.md
git mv secrets.example/deepcab_api_key.example infra/secrets.example/deepcab_api_key.example
git mv secrets.example/minio_root_password.example infra/secrets.example/minio_root_password.example
git mv secrets.example/openai_api_key.example infra/secrets.example/openai_api_key.example
git mv secrets.example/postgres_password.example infra/secrets.example/postgres_password.example
rmdir secrets.example
```

- [ ] **Step 2: Update `.gitignore`**

Find `secrets/` or `001-deepCab-api/secrets/` in `.gitignore` and replace with `infra/secrets/`.

If no entry exists, append:

```
infra/secrets/
```

- [ ] **Step 3: Update `infra/secrets.example/README.md`**

Change the one-time-copy instruction line from:

```
cp secrets.example/<file> secrets/<file_without_example_suffix>
```

to:

```
cp infra/secrets.example/<file> infra/secrets/<file_without_example_suffix>
```

---

## Task A5: Update Makefile

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Replace every compose path**

```bash
grep -n "docker-compose" Makefile
grep -n "docker compose" Makefile
```

For each line that uses one of the moved files, prepend `infra/compose/`. Examples (existing → new):

```make
# before
docker compose up -d
# after
docker compose -f infra/compose/docker-compose.yml up -d
```

```make
# before
docker compose -f docker-compose.yml -f docker-compose.obs.yml up -d
# after
docker compose -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.obs.yml up -d
```

```make
# before
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build api
# after
docker compose -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.gpu.yml up -d --build api
```

- [ ] **Step 2: Add new make targets**

Append to the Makefile (alphabetize under existing targets if there's an obvious section):

```make
docker_up:
	docker compose -f infra/compose/docker-compose.yml up -d

docker_down:
	docker compose -f infra/compose/docker-compose.yml down

docker_obs_up:
	docker compose -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.obs.yml up -d

docker_obs_down:
	docker compose -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.obs.yml down

docker_gpu_up:
	docker compose -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.gpu.yml up -d --build api
```

- [ ] **Step 3: Verify `make list`**

```bash
make list 2>&1 | head -40
```

Expected: new targets appear, no syntax errors.

---

## Task A6: Update CLAUDE.md, README.md, CONTRIBUTING.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`

- [ ] **Step 1: CLAUDE.md — Architecture map**

Find the directory tree under "Architecture — refactor in flight" and the `docker-compose` paragraph below it. Insert `infra/` tree per spec §4.1 above the existing tree (or replace if it overlaps). Mark old root paths as moved.

The "docker-compose at `001-deepCab-api/`:" paragraph should be rewritten:

```markdown
**Container artifacts** live under `infra/`:
- `infra/docker/Dockerfile` — api image (was at repo root)
- `infra/compose/docker-compose.yml` — core stack: traefik, postgres, minio, redis, mlflow, prefect, prefect-agent, api
- `infra/compose/docker-compose.obs.yml` — adds otel, jaeger, prometheus, alertmanager, grafana, loki, promtail
- `infra/compose/docker-compose.dev.yml` — adds ngrok, pgadmin, react-dev (dev only)
- `infra/compose/docker-compose.gpu.yml` — GPU override (nvidia runtime, GPU=1 build arg)
- `infra/compose/conf/{otel-collector.yaml,prometheus.yml,grafana/provisioning/}` — service configs
- `infra/secrets.example/` — template; copy to `infra/secrets/` (gitignored) for first run
```

- [ ] **Step 2: CLAUDE.md — Common commands**

Find the "Common commands" section. Replace every `docker compose ...` example with the `-f infra/compose/...` form. Add the new `make docker_up`, `make docker_obs_up`, `make docker_gpu_up` shortcuts.

- [ ] **Step 3: README.md — Quickstart**

Find the install/run instructions and replace any reference to `docker-compose.yml` at repo root with the `infra/compose/` path. Mention the new `make docker_up` shortcut.

- [ ] **Step 4: CONTRIBUTING.md — Common snags**

Append a snag entry:

```markdown
### "docker compose can't find docker-compose.yml at root"

The compose files moved under `infra/compose/` in Sub-project A (2026-06-03). Run `make docker_up` instead of bare `docker compose up`, or pass `-f infra/compose/docker-compose.yml` explicitly.
```

---

## Task A7: Verify nothing broke

**Files:** none — verification only.

- [ ] **Step 1: Config check all compose files**

```bash
docker compose -f infra/compose/docker-compose.yml config --quiet && echo "core OK"
docker compose -f infra/compose/docker-compose.obs.yml config --quiet && echo "obs OK"
docker compose -f infra/compose/docker-compose.gpu.yml config --quiet && echo "gpu OK"
```

Expected: three "OK" lines.

- [ ] **Step 2: Run the test suite**

```bash
uv run pytest tests/ -q --ignore=tests/all
```

Expected: 103 passing, 13 gated-skipped (the STANDING.md baseline). If anything fails, the path migration broke a test fixture — fix before committing.

- [ ] **Step 3: Seed secrets if missing, then dry-run a container**

```bash
test -d infra/secrets || cp -r infra/secrets.example/ infra/secrets/
# Trim ".example" suffix on every file in infra/secrets/
for f in infra/secrets/*.example; do mv "$f" "${f%.example}"; done
docker compose -f infra/compose/docker-compose.yml build api
```

Expected: image builds successfully.

---

## Task A8: Commit Sub-project A

**Files:** none — git only.

- [ ] **Step 1: Stage everything**

```bash
git add -A
git status --short
```

Expected: a bunch of renames (`R  Dockerfile -> infra/docker/Dockerfile`), some modifications (Makefile, CLAUDE.md, README.md, CONTRIBUTING.md, .gitignore), one new file (`infra/README.md`).

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
chore(infra): reorganize container artifacts under infra/

Move Dockerfile + 3 compose files + compose/ conf tree + secrets.example/
under infra/. Update Makefile paths and add make docker_up / docker_obs_up
/ docker_gpu_up shortcuts. Update CLAUDE.md, README.md, CONTRIBUTING.md.

.github/workflows/ stays at repo root (GitHub-discovered).

Sub-project A of the GCP infra design
(docs/superpowers/specs/2026-06-03-deepcab-gcp-infra-and-audit-design.md).

103 tests still pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Sanity check**

```bash
git log --oneline -1
git status --short
```

Expected: one new commit, clean working tree.

---

## Done criteria

- [x] All compose files live under `infra/compose/`
- [x] Dockerfile lives under `infra/docker/`
- [x] `compose/` tree moved under `infra/compose/conf/`
- [x] `secrets.example/` moved under `infra/secrets.example/`
- [x] Makefile uses `infra/compose/...` paths and exposes `docker_up`, `docker_obs_up`, `docker_gpu_up`
- [x] `infra/README.md` exists
- [x] `.gitignore` ignores `infra/secrets/`
- [x] CLAUDE.md, README.md, CONTRIBUTING.md reflect the new layout
- [x] `pytest` still green (103 passing)
- [x] One commit on the branch
