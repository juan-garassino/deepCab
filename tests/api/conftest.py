"""Shared fixtures for the API test suite.

B.6 consolidation: `_clean_state` was duplicated between `test_routers.py`
(reset model + tasks) and `test_graphql.py` (reset model only). Hoisted here
in its strictest form so both files inherit the same isolation."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset in-process STATE (model handle + train-task table) around every
    API test so no residual fitted model or pending task bleeds between cases."""
    from deepCab.api.state import STATE

    STATE.model = None
    STATE.tasks.clear()
    yield
    STATE.model = None
    STATE.tasks.clear()
