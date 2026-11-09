"""Regression tests for SerpAPI request memory behavior."""

from typing import ClassVar
from unittest.mock import patch

from agentic_internet.agents.multi_model_serpapi import ContextEngineeringMixin, GoogleSearchTool


class FakeGoogleSearch:
    """Capture request parameters and return one valid organic result."""

    calls: ClassVar[list[dict]] = []

    def __init__(self, params):
        self.params = dict(params)
        self.__class__.calls.append(self.params)

    def get_dict(self):
        return {"organic_results": [{"title": "Result", "link": "https://example.com"}]}


def test_sequential_searches_preserve_current_query_and_explicit_params():
    FakeGoogleSearch.calls = []
    engine = ContextEngineeringMixin()
    tool = GoogleSearchTool(api_key="synthetic-key", context_engine=engine)

    with patch("agentic_internet.agents.multi_model.tools.GoogleSearch", FakeGoogleSearch):
        tool.forward("first-query", language="en")
        tool.forward("second-query", language="fr")

    assert [(call["q"], call["hl"]) for call in FakeGoogleSearch.calls] == [
        ("first-query", "en"),
        ("second-query", "fr"),
    ]
    remembered = engine.memory.get_search_recommendations("web_search", "google")["best_params"]
    assert remembered == {"gl": "us", "hl": "fr", "num": 10}
    assert "api_key" not in remembered
    assert "q" not in remembered
