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
