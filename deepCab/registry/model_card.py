"""Auto-emit MODEL_CARD.md per registered model.

Inputs: TrainConfig + val metrics + Provenance manifest + (optional) SHAP top
features. Output: a markdown file the audit trail can read directly. Fields
follow the Hugging Face / Google Model-Card schema loosely — intended use,
data, evaluation, ethical considerations, provenance.

Phase 10's training/train.py hook will call `write_model_card(...)` after a
successful run and log it as an MLflow artifact alongside provenance.json."""

from __future__ import annotations

from pathlib import Path

from deepCab.schemas.config import TrainConfig

TEMPLATE = """# Model Card — {model_name} v{version}

## Intended use
NYC taxi-fare prediction. Input: pickup datetime + 4 lon/lat coords + passenger count.
Output: predicted fare in USD.

## Backend
- Kind: `{backend_kind}`
- Config:
```yaml
{backend_yaml}
```

## Training data
- Dataset slice: `{data_size}` (train) / `{val_size}` (val)
- Input parquet hash: `{input_hash}`
- Preprocessor: 65-d feature space (passenger scaler + time OHE/sin/cos + haversine/manhattan + geohash one-hots)

## Evaluation
| Metric  | Value |
|---------|------:|
{metric_rows}

## Top feature attributions (mean |SHAP|, 5-group aggregation)
{shap_rows}

## Provenance
- Git SHA: `{git_sha}`
- Config hash: `{config_hash}`
- Python: `{python}`
- Platform: `{platform}`
- ONNX opset: `{onnx_opset}`
- CUDA available: `{cuda_available}` (`{cuda_version}`)

## Ethical / fairness notes
This model is trained on a 2009-2015 NYC taxi sample. Predictions may not
generalize outside this geographic + temporal window. No protected-attribute
features are used; predictions should not be relied upon for policy decisions.

## Limitations
- Coordinates outside the NYC bounding box (lat ∈ {nyc_lat}, lon ∈ {nyc_lon}) are rejected upstream.
- Fares > $400 or with passenger_count ∉ [1, 8] are filtered in `clean_data`.
- SHAP aggregation merges pickup_lat with pickup_lon (and same for dropoff)
  into a `*_location` group — see `deepCab/explain/aggregate.py`.
"""


def write_model_card(
    *,
    out_path: Path,
    model_name: str,
    version: int,
    cfg: TrainConfig,
    metrics: dict[str, float],
    provenance: dict,
    shap_top: dict[str, float] | None = None,
) -> Path:
    import yaml

    from deepCab.schemas.data import NYC_LAT, NYC_LON

    metric_rows = "\n".join(f"| {k}  | {v:.4f} |" for k, v in metrics.items())
    if shap_top:
        shap_rows = "\n".join(
            f"- **{name}**: {value:.4f}"
            for name, value in sorted(shap_top.items(), key=lambda kv: -abs(kv[1]))
        )
    else:
        shap_rows = "_(no SHAP summary captured)_"

    body = TEMPLATE.format(
        model_name=model_name,
        version=version,
        backend_kind=cfg.backend.kind,
        backend_yaml=yaml.safe_dump(cfg.backend.model_dump(), sort_keys=False).strip(),
        data_size=cfg.data.size,
        val_size=cfg.data.validation_size,
        input_hash=provenance.get("config_hash", "?"),
        metric_rows=metric_rows or "| _no metrics_ | |",
        shap_rows=shap_rows,
        git_sha=provenance.get("git_sha") or "unknown",
        config_hash=provenance.get("config_hash", "?"),
        python=provenance.get("python", "?"),
        platform=provenance.get("platform", "?"),
        onnx_opset=provenance.get("onnx_opset", "?"),
        cuda_available=provenance.get("cuda_available", False),
        cuda_version=provenance.get("cuda_version") or "n/a",
        nyc_lat=NYC_LAT,
        nyc_lon=NYC_LON,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)
    return out_path
