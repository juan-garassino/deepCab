"""SHAP explainability layer: per-backend factory, 65d -> 5-group aggregation,
cached global summary. Endpoint binding lives in Phase 6 api/routers/explain.py."""
from deepCab.explain.aggregate import COLUMN_GROUPS, aggregate_global, aggregate_shap  # noqa: F401
from deepCab.explain.cache import GlobalSummary, clear_cache, get_global_summary  # noqa: F401
from deepCab.explain.explainer import Explanation, explain_batch, explain_row, make_explainer  # noqa: F401
