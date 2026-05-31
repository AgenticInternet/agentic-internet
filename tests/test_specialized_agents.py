"""Tests for specialized agent helper logic.

The heavy ``InternetAgent.__init__`` (which initializes a model) is bypassed via
``__new__`` so that only the pure helper logic is exercised.
"""

from agentic_internet.agents.specialized_agents import (
    BrowserAutomationAgent,
    ContentCreationAgent,
)


def _agent_with_run(return_value: str) -> BrowserAutomationAgent:
    agent = BrowserAutomationAgent.__new__(BrowserAutomationAgent)
    agent.run = lambda task: return_value  # type: ignore[method-assign]
    return agent


def _content_agent_with_run(return_value: str) -> ContentCreationAgent:
    agent = ContentCreationAgent.__new__(ContentCreationAgent)
    agent.run = lambda task: return_value  # type: ignore[method-assign]
    return agent


class TestScrapeStructuredData:
    def test_parses_valid_json_object(self):
        agent = _agent_with_run('{"title": "Example", "price": 10}')
        result = agent.scrape_structured_data("http://example.com", {"title": "string"})
        assert result == {"title": "Example", "price": 10}

    def test_non_dict_json_wrapped_as_raw(self):
        agent = _agent_with_run("[1, 2, 3]")
        result = agent.scrape_structured_data("http://example.com", {})
        assert result == {"raw_result": "[1, 2, 3]"}

    def test_invalid_json_wrapped_as_raw(self):
        agent = _agent_with_run("not json at all")
        result = agent.scrape_structured_data("http://example.com", {})
        assert result == {"raw_result": "not json at all"}


class TestWriteArticle:
    def test_returns_metadata_with_content(self):
        agent = _content_agent_with_run("Article body")
        result = agent.write_article("AI", style="formal", word_count=300)
        assert result["topic"] == "AI"
        assert result["style"] == "formal"
        assert result["content"] == "Article body"
        assert result["word_count_target"] == 300
