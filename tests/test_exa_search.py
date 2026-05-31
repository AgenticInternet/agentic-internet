"""Tests for the Exa search tools."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agentic_internet.tools.exa_search import (
    INTEGRATION_NAME,
    ExaFindSimilarTool,
    ExaResult,
    ExaSearchTool,
    _build_contents_kwargs,
    _format_results,
    _normalize_category,
    _normalize_search_type,
)


def _make_raw(**kwargs):
    """Build a stand-in for an exa_py Result object."""
    defaults = {
        "title": None,
        "url": "",
        "text": None,
        "summary": None,
        "highlights": None,
        "author": None,
        "published_date": None,
        "score": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestExaResult:
    def test_from_sdk_with_full_fields(self):
        raw = _make_raw(
            title="A title",
            url="https://example.com",
            text="full text",
            summary="a summary",
            highlights=["h1", "h2"],
            author="Alice",
            published_date="2024-01-01",
            score=0.9,
        )
        r = ExaResult.from_sdk(raw)
        assert r.title == "A title"
        assert r.url == "https://example.com"
        assert r.text == "full text"
        assert r.summary == "a summary"
        assert r.highlights == ["h1", "h2"]
        assert r.author == "Alice"
        assert r.published_date == "2024-01-01"
        assert r.score == 0.9

    def test_from_sdk_with_missing_fields_uses_defaults(self):
        raw = _make_raw(url="https://example.com")
        r = ExaResult.from_sdk(raw)
        assert r.title == "No title"
        assert r.url == "https://example.com"
        assert r.text is None
        assert r.summary is None
        assert r.highlights == []

    def test_snippet_prefers_summary(self):
        r = ExaResult(
            title="t",
            url="u",
            summary="summary content",
            highlights=["highlight content"],
            text="full text content",
        )
        assert r.snippet() == "summary content"

    def test_snippet_falls_back_to_highlights(self):
        r = ExaResult(
            title="t",
            url="u",
            highlights=["first highlight", "second highlight"],
            text="full text content",
        )
        snippet = r.snippet()
        assert "first highlight" in snippet
        assert "second highlight" in snippet

    def test_snippet_falls_back_to_text(self):
        r = ExaResult(title="t", url="u", text="full text content")
        assert r.snippet() == "full text content"

    def test_snippet_returns_no_description_when_empty(self):
        r = ExaResult(title="t", url="u")
        assert r.snippet() == "No description"

    def test_snippet_truncates_long_content(self):
        r = ExaResult(title="t", url="u", text="x" * 1000)
        snippet = r.snippet(max_chars=100)
        assert snippet.endswith("...")
        assert len(snippet) <= 103


class TestFormatResults:
    def test_empty_results(self):
        assert _format_results([]) == "No search results found."

    def test_includes_title_url_and_snippet(self):
        results = [
            ExaResult(
                title="Hello",
                url="https://example.com",
                summary="A summary",
                author="Bob",
                published_date="2024-01-01",
            )
        ]
        out = _format_results(results)
        assert "Hello" in out
        assert "https://example.com" in out
        assert "A summary" in out
        assert "Bob" in out
        assert "2024-01-01" in out


class TestBuildContentsKwargs:
    def test_all_disabled(self):
        kwargs = _build_contents_kwargs(text=False, text_max_chars=None, highlights=False, summary=False)
        assert kwargs == {}

    def test_text_with_max_chars(self):
        kwargs = _build_contents_kwargs(text=True, text_max_chars=500, highlights=False, summary=False)
        assert kwargs == {"text": {"max_characters": 500}}

    def test_text_without_max_chars(self):
        kwargs = _build_contents_kwargs(text=True, text_max_chars=None, highlights=False, summary=False)
        assert kwargs == {"text": True}

    def test_highlights_enabled(self):
        kwargs = _build_contents_kwargs(text=False, text_max_chars=None, highlights=True, summary=False)
        assert kwargs == {"highlights": True}

    def test_summary_without_query(self):
        kwargs = _build_contents_kwargs(text=False, text_max_chars=None, highlights=False, summary=True)
        assert kwargs == {"summary": True}

    def test_summary_with_query(self):
        kwargs = _build_contents_kwargs(
            text=False,
            text_max_chars=None,
            highlights=False,
            summary=True,
            summary_query="What is X?",
        )
        assert kwargs == {"summary": {"query": "What is X?"}}

    def test_all_enabled_simultaneously(self):
        kwargs = _build_contents_kwargs(text=True, text_max_chars=200, highlights=True, summary=True)
        assert kwargs == {
            "text": {"max_characters": 200},
            "highlights": True,
            "summary": True,
        }


class TestNormalizers:
    def test_search_type_passthrough(self):
        assert _normalize_search_type("neural") == "neural"
        assert _normalize_search_type("auto") == "auto"

    def test_search_type_unknown_returns_none(self):
        assert _normalize_search_type("keyword") is None
        assert _normalize_search_type("bogus") is None

    def test_search_type_none_returns_none(self):
        assert _normalize_search_type(None) is None

    def test_category_passthrough(self):
        assert _normalize_category("research paper") == "research paper"
        assert _normalize_category("news") == "news"

    def test_category_unknown_returns_none(self):
        assert _normalize_category("bogus") is None
        assert _normalize_category("") is None


class TestExaSearchTool:
    @patch("agentic_internet.tools.exa_search.HAS_EXA", False)
    def test_disabled_when_sdk_missing(self):
        tool = ExaSearchTool.__new__(ExaSearchTool)
        tool.api_key = "fake"
        assert tool.is_available() is False
        assert "not available" in tool.forward("test").lower()

    @patch("agentic_internet.tools.exa_search.HAS_EXA", True)
    def test_disabled_when_api_key_missing(self):
        tool = ExaSearchTool.__new__(ExaSearchTool)
        tool.api_key = None
        assert tool.is_available() is False
        assert "EXA_API_KEY" in tool.forward("test")

    @patch("agentic_internet.tools.exa_search.HAS_EXA", True)
    @patch("agentic_internet.tools.exa_search.Exa")
    def test_forward_calls_search_and_contents_with_filters(self, mock_exa_cls):
        mock_client = MagicMock()
        mock_client.headers = {}
        mock_response = SimpleNamespace(
            results=[
                _make_raw(
                    title="Result 1",
                    url="https://example.com/a",
                    summary="Summary 1",
                )
            ]
        )
        mock_client.search_and_contents.return_value = mock_response
        mock_exa_cls.return_value = mock_client

        tool = ExaSearchTool.__new__(ExaSearchTool)
        tool.api_key = "fake-key"
        tool.num_results = 5
        tool.include_text = False
        tool.include_highlights = True
        tool.include_summary = True
        tool.summary_query = None
        tool.text_max_chars = 500

        result = tool.forward(
            "ai agents",
            search_type="neural",
            category="research paper",
            include_domains=["arxiv.org"],
            exclude_domains=["spam.com"],
            start_published_date="2024-01-01",
            end_published_date="2024-12-31",
        )

        assert "Result 1" in result
        assert "https://example.com/a" in result

        mock_exa_cls.assert_called_once_with(api_key="fake-key")
        assert mock_client.headers["x-exa-integration"] == INTEGRATION_NAME

        kwargs = mock_client.search_and_contents.call_args.kwargs
        assert kwargs["query"] == "ai agents"
        assert kwargs["num_results"] == 5
        assert kwargs["type"] == "neural"
        assert kwargs["category"] == "research paper"
        assert kwargs["include_domains"] == ["arxiv.org"]
        assert kwargs["exclude_domains"] == ["spam.com"]
        assert kwargs["start_published_date"] == "2024-01-01"
        assert kwargs["end_published_date"] == "2024-12-31"
        assert kwargs["highlights"] is True
        assert kwargs["summary"] is True

    @patch("agentic_internet.tools.exa_search.HAS_EXA", True)
    @patch("agentic_internet.tools.exa_search.Exa")
    def test_forward_drops_unknown_search_type(self, mock_exa_cls):
        mock_client = MagicMock()
        mock_client.headers = {}
        mock_client.search_and_contents.return_value = SimpleNamespace(results=[])
        mock_exa_cls.return_value = mock_client

        tool = ExaSearchTool.__new__(ExaSearchTool)
        tool.api_key = "fake-key"
        tool.num_results = 5
        tool.include_text = False
        tool.include_highlights = False
        tool.include_summary = False
        tool.summary_query = None
        tool.text_max_chars = 500

        tool.forward("query", search_type="keyword")

        kwargs = mock_client.search_and_contents.call_args.kwargs
        assert "type" not in kwargs

    @patch("agentic_internet.tools.exa_search.HAS_EXA", True)
    @patch("agentic_internet.tools.exa_search.Exa")
    def test_forward_returns_error_on_exception(self, mock_exa_cls):
        mock_client = MagicMock()
        mock_client.headers = {}
        mock_client.search_and_contents.side_effect = RuntimeError("API down")
        mock_exa_cls.return_value = mock_client

        tool = ExaSearchTool.__new__(ExaSearchTool)
        tool.api_key = "fake-key"
        tool.num_results = 5
        tool.include_text = False
        tool.include_highlights = True
        tool.include_summary = True
        tool.summary_query = None
        tool.text_max_chars = 500

        result = tool.forward("query")
        assert "Error" in result
        assert "API down" in result

    @patch("agentic_internet.tools.exa_search.HAS_EXA", True)
    @patch("agentic_internet.tools.exa_search.Exa")
    def test_forward_handles_missing_content_fields(self, mock_exa_cls):
        """Result with no summary/highlights/text falls back to 'No description'."""
        mock_client = MagicMock()
        mock_client.headers = {}
        mock_client.search_and_contents.return_value = SimpleNamespace(
            results=[_make_raw(title="Bare", url="https://example.com/bare")]
        )
        mock_exa_cls.return_value = mock_client

        tool = ExaSearchTool.__new__(ExaSearchTool)
        tool.api_key = "fake-key"
        tool.num_results = 5
        tool.include_text = False
        tool.include_highlights = False
        tool.include_summary = False
        tool.summary_query = None
        tool.text_max_chars = 500

        result = tool.forward("query")
        assert "Bare" in result
        assert "No description" in result


class TestExaFindSimilarTool:
    @patch("agentic_internet.tools.exa_search.HAS_EXA", True)
    @patch("agentic_internet.tools.exa_search.Exa")
    def test_forward_calls_find_similar_and_contents(self, mock_exa_cls):
        mock_client = MagicMock()
        mock_client.headers = {}
        mock_client.find_similar_and_contents.return_value = SimpleNamespace(
            results=[
                _make_raw(
                    title="Similar",
                    url="https://example.com/similar",
                    highlights=["a", "b"],
                )
            ]
        )
        mock_exa_cls.return_value = mock_client

        tool = ExaFindSimilarTool.__new__(ExaFindSimilarTool)
        tool.api_key = "fake-key"
        tool.num_results = 3
        tool.include_text = False
        tool.include_highlights = True
        tool.include_summary = False
        tool.summary_query = None
        tool.text_max_chars = 500

        result = tool.forward("https://example.com/source", exclude_source_domain=True)

        assert "Similar" in result
        assert "https://example.com/similar" in result
        assert mock_client.headers["x-exa-integration"] == INTEGRATION_NAME

        kwargs = mock_client.find_similar_and_contents.call_args.kwargs
        assert kwargs["url"] == "https://example.com/source"
        assert kwargs["num_results"] == 3
        assert kwargs["exclude_source_domain"] is True
        assert kwargs["highlights"] is True

    @patch("agentic_internet.tools.exa_search.HAS_EXA", False)
    def test_disabled_when_sdk_missing(self):
        tool = ExaFindSimilarTool.__new__(ExaFindSimilarTool)
        tool.api_key = "fake"
        assert tool.is_available() is False
        assert "not available" in tool.forward("https://example.com").lower()


class TestDefaultToolsRegistration:
    """Verify Exa tools are only registered when EXA_API_KEY is set."""

    def test_not_registered_when_api_key_missing(self, monkeypatch):
        from agentic_internet.config.settings import settings

        monkeypatch.setattr(settings, "exa_api_key", None)

        from agentic_internet.agents.internet_agent import InternetAgent

        agent = InternetAgent.__new__(InternetAgent)
        agent.verbose = False
        agent.tools = []
        tools = agent._get_default_tools()
        names = {t.name for t in tools}
        assert "exa_search" not in names
        assert "exa_find_similar" not in names

    def test_registered_when_api_key_set(self, monkeypatch):
        from agentic_internet.config.settings import settings

        monkeypatch.setattr(settings, "exa_api_key", "fake-key")
        # Avoid pulling in the browser tool which requires a separate key
        monkeypatch.setattr(settings.tools, "browser_enabled", False)

        from agentic_internet.agents.internet_agent import InternetAgent

        agent = InternetAgent.__new__(InternetAgent)
        agent.verbose = False
        agent.tools = []
        tools = agent._get_default_tools()
        names = {t.name for t in tools}
        assert "exa_search" in names
        assert "exa_find_similar" in names
