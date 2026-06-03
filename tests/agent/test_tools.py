"""Tool registry: schema generation matches Pydantic, unknown tools come back
as {"error": ...} (never raise), args validation surfaces as error data."""
from __future__ import annotations

from deepCab.agent.tools import dispatch, openai_tools, tool_names


def test_openai_tools_has_one_entry_per_registered_tool() -> None:
    tools = openai_tools()
    names = {t["function"]["name"] for t in tools}
    assert names == set(tool_names())
    # Each entry must carry a JSONSchema-style parameters block (Pydantic-derived)
    for t in tools:
        assert "parameters" in t["function"]
        assert "properties" in t["function"]["parameters"]


def test_dispatch_unknown_tool_returns_error_dict() -> None:
    out = dispatch("nope", {})
    assert "error" in out
    assert "unknown tool" in out["error"]


def test_dispatch_bad_args_returns_validation_error() -> None:
    out = dispatch("preprocess", {"data": "not-a-dataref"})
    assert "error" in out
    assert "validation" in out["error"].lower()
