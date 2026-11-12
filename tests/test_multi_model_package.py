"""Compatibility tests for the split multi-model orchestration package."""

import pytest

from agentic_internet.agents import multi_model, multi_model_serpapi

PUBLIC_CLASS_NAMES = [
    "ModelManager",
    "SearchResult",
    "LocalResult",
    "ShoppingResult",
    "NewsResult",
    "ScholarResult",
    "ImageResult",
    "ContextWindow",
    "AgentMemory",
    "TaskContext",
    "ContextEngineeringMixin",
    "GoogleSearchTool",
    "GoogleShoppingTool",
    "GoogleMapsLocalTool",
    "GoogleScholarTool",
    "MultiEngineSearchTool",
    "AgentTool",
    "MultiModelSerpAPISystem",
    "LiteLLMModel",
    "GoogleSearch",
    "BingSearch",
    "YahooSearch",
    "BaiduSearch",
    "GoogleScholarSearch",
    "CodeAgent",
    "ToolCallingAgent",
    "Tool",
]


@pytest.mark.parametrize("name", PUBLIC_CLASS_NAMES)
def test_public_class_identity_is_preserved(name):
    assert getattr(multi_model, name) is getattr(multi_model_serpapi, name)


def test_non_reusable_search_params_identity_is_preserved():
    assert multi_model.NON_REUSABLE_SEARCH_PARAMS is multi_model_serpapi.NON_REUSABLE_SEARCH_PARAMS
