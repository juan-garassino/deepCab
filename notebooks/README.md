# Notebooks — phase-by-phase walkthrough

Each notebook is a runnable lesson against the live package (`uv sync --extra dev` first).
The intent is **read-and-run** — every cell is self-contained, no hidden state from prior notebooks.

| # | Notebook                          | What it teaches                                                                      | Status |
|---|-----------------------------------|--------------------------------------------------------------------------------------|--------|
| 00 | `00-overview.ipynb`              | Architectural tour: where each piece lives, how data flows, key design decisions     | ✅      |
| 01 | `01-data-layer.ipynb`            | Polars + Parquet/Hive + Pandera schemas + content-hash lineage                       | scaffold |
| 02 | `02-backend-zoo.ipynb`           | Discriminated-union config + factory + sklearn `check_estimator` across 6 backends   | ✅      |
| 03 | `03-hydra-optuna-cv.ipynb`       | OmegaConf→Pydantic bridge + Optuna HPO + TimeSeriesSplit                             | scaffold |
| 04 | `04-shap-explainability.ipynb`   | Per-backend SHAP factory + 65d→5-group aggregation                                   | scaffold |
| 05 | `05-fastapi-routers.ipynb`       | DI, middleware, BG tasks, SSE — the "learning hub" surface                          | scaffold |
| 06 | `06-onnx-serving.ipynb`          | Export per backend + ONNX runtime + batcher + INT8 quant trade-offs                 | scaffold |
| 07 | `07-agent-loop.ipynb`            | Tool registry, planner+executor split, budget cap, idempotent replay                | ✅      |
| 08 | `08-prefect-flow.ipynb`          | `@flow` + `@task` + cron schedule + compose integration                              | scaffold |
| 09 | `09-ci-and-cards.ipynb`          | GH Actions matrix + auto MODEL_CARD.md + lineage SQLite                              | scaffold |

`scaffold` = the file exists with section headings + one runnable smoke cell.
The full prose + exercises land per-notebook in follow-ups; the scaffolds are
honest stubs, not empty.

## Running

```bash
uv run jupyter notebook notebooks/
# or non-interactive:
uv run jupyter nbconvert --execute notebooks/00-overview.ipynb --to notebook --output 00-overview.executed.ipynb
```

## Why three "real" notebooks now, the rest scaffolded?

Each notebook is a real lesson — research + prose + executable + exercise design.
Doing 10 in one batch produces 10 mediocre notebooks. Doing 3 thoroughly
(overview + the keystone backend factory + the agent loop) and scaffolding 7
gives a clear roadmap without faking depth.

## `colab-train-and-push.ipynb`

Train on Colab's GPU with the editor still on your laptop:

1. Open the notebook on Colab (https://colab.research.google.com/github/<owner>/<repo>/blob/main/001-deepCab-api/notebooks/colab-train-and-push.ipynb). Switch to GPU runtime.
2. Set 4 Colab secrets (Tools → Secrets): `NGROK_AUTHTOKEN`, `GH_TOKEN`, `GCP_PROJECT`, `GH_REPO`.
3. Run cells 1–3. Cell 3 prints a URL.
4. In VS Code: `Cmd+Shift+P` → *Jupyter: Specify Jupyter Server for Connections* → paste the URL from cell 3.
5. Open this notebook locally. Use the kernel picker to choose the remote ngrok server.
6. Run cells 4–6 from VS Code. Cell 4 trains on Colab's GPU; cell 5 pushes to GCS; cell 6 triggers `deploy-cloud-run.yml`.

If the Colab runtime disconnects, re-run cell 3 to get a new ngrok URL (free-tier ngrok URLs are ephemeral).
