"""Tests for use-case orchestration runtime helpers."""

from types import SimpleNamespace
from typing import cast

import pytest
from smolagents import Tool

from agentic_internet.agents.multi_model_serpapi import MultiModelSerpAPISystem
from agentic_internet.agents.orchestration_runtime import (
    parse_worker_model_overrides,
    resolve_tool_bundles,
    resolve_use_case_tools,
    summarize_use_case,
)
from agentic_internet.agents.use_cases import get_use_case_recipe
from agentic_internet.config.settings import settings


def _tool(name: str) -> Tool:
    return cast(Tool, SimpleNamespace(name=name, description=f"{name} desc"))


def test_parse_worker_model_overrides():
    parsed = parse_worker_model_overrides(["code_analyst=qwen-coder", "technical_researcher=sonar"])
    assert parsed == {"code_analyst": "qwen-coder", "technical_researcher": "sonar"}


def test_parse_worker_model_overrides_rejects_bad_format():
    with pytest.raises(ValueError, match="worker=model"):
        parse_worker_model_overrides(["qwen-coder"])


def test_resolve_tool_bundles_dedupes_tools():
    tools, missing = resolve_tool_bundles([_tool("web_search"), _tool("news_search")], ("web", "web"))
    assert [tool.name for tool in tools] == ["web_search", "news_search"]
    assert missing == ()


def test_resolve_use_case_tools_reports_missing_worker_bundle():
    recipe = get_use_case_recipe("technical_due_diligence")
    resolved = resolve_use_case_tools(recipe, [_tool("web_search"), _tool("python_executor"), _tool("data_analysis")])

    assert resolved.tool_names_for_worker("code_analyst") == ("python_executor", "data_analysis")
    assert "technical_researcher:scraper" in resolved.missing_bundles


def test_summarize_use_case_includes_k_and_workers():
    recipe = get_use_case_recipe("technical_due_diligence")
    resolved = resolve_use_case_tools(recipe, [_tool("web_search"), _tool("python_executor"), _tool("data_analysis")])
    summary = summarize_use_case(recipe, resolved)

    assert summary["id"] == "technical_due_diligence"
    assert summary["k"] == 3
    assert [worker["name"] for worker in summary["workers"]] == [
        "technical_researcher",
        "code_analyst",
        "risk_synthesizer",
    ]


@pytest.mark.parametrize(("enabled", "expected"), [(False, False), (True, True)])
def test_use_case_inventory_respects_code_execution_setting(monkeypatch, enabled, expected):
    monkeypatch.setattr("agentic_internet.agents.multi_model_serpapi.ModelManager", lambda: SimpleNamespace())
    monkeypatch.setattr(settings.tools, "code_execution_enabled", enabled)
    system = MultiModelSerpAPISystem(serpapi_key="")

    names = {tool.name for tool in system.create_use_case_tool_inventory()}

    assert ("python_executor" in names) is expected
    assert ("data_analysis" in names) is expected
    assert "web_search" in names
