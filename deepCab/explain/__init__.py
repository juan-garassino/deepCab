"""SHAP explainability layer: per-backend factory, 65d -> 5-group aggregation,
cached global summary. Endpoint binding lives in api/routers/explain.py.

Submodules are imported directly (e.g.
`from deepCab.explain.explainer import explain_row`)."""
