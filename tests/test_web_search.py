"""Tests for web search tools."""

from unittest.mock import MagicMock, patch

from agentic_internet.tools.web_search import (
    NewsSearchTool,
    WebScraperTool,
    WebSearchTool,
    _search_ddgs,
    _search_serpapi,
    _search_with_fallback,
    _validate_url,
)


class TestValidateUrl:
    def test_valid_http(self):
        assert _validate_url("http://example.com") is None

    def test_valid_https(self):
        assert _validate_url("https://example.com/path?q=1") is None

    def test_invalid_scheme(self):
        err = _validate_url("ftp://example.com")
        assert err is not None
        assert "scheme" in err.lower()

    def test_no_scheme(self):
        err = _validate_url("example.com")
        assert err is not None

    def test_empty_url(self):
        err = _validate_url("")
        assert err is not None


class TestSearchSerpapi:
    @patch("agentic_internet.tools.web_search.HAS_SERPAPI", True)
    @patch("agentic_internet.tools.web_search.GoogleSearch")
    def test_returns_formatted_results(self, mock_search_cls):
        mock_search_cls.return_value.get_dict.return_value = {
            "organic_results": [
                {"title": "Test Title", "snippet": "Test snippet", "link": "https://example.com"}
            ]
        }
        result = _search_serpapi("test query", "fake_key")
        assert result is not None
        assert "Test Title" in result
        assert "https://example.com" in result

    @patch("agentic_internet.tools.web_search.HAS_SERPAPI", True)
    @patch("agentic_internet.tools.web_search.GoogleSearch")
    def test_returns_none_on_empty_results(self, mock_search_cls):
        mock_search_cls.return_value.get_dict.return_value = {"organic_results": []}
        result = _search_serpapi("test query", "fake_key")
        assert result is None

    @patch("agentic_internet.tools.web_search.HAS_SERPAPI", False)
    def test_returns_none_when_not_available(self):
        result = _search_serpapi("test", "key")
        assert result is None

    @patch("agentic_internet.tools.web_search.HAS_SERPAPI", True)
    @patch("agentic_internet.tools.web_search.GoogleSearch")
    def test_returns_none_on_exception(self, mock_search_cls):
        mock_search_cls.return_value.get_dict.side_effect = Exception("API error")
        result = _search_serpapi("test", "key")
        assert result is None


class TestSearchDdgs:
    @patch("agentic_internet.tools.web_search.HAS_DDGS", True)
    @patch("agentic_internet.tools.web_search.DDGS")
    def test_returns_formatted_results(self, mock_ddgs_cls):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.text.return_value = [
            {"title": "DDG Title", "body": "DDG body", "href": "https://ddg.com"}
        ]
        mock_ddgs_cls.return_value = mock_ctx

        result = _search_ddgs("test query")
        assert "DDG Title" in result

    @patch("agentic_internet.tools.web_search.HAS_DDGS", False)
    def test_returns_error_when_not_available(self):
        result = _search_ddgs("test")
        assert "not available" in result.lower()


class TestSearchWithFallback:
    @patch("agentic_internet.tools.web_search._search_serpapi")
    @patch("agentic_internet.tools.web_search._search_ddgs")
    def test_uses_serpapi_when_available(self, mock_ddgs, mock_serpapi):
        mock_serpapi.return_value = "SerpAPI result"
        result = _search_with_fallback("query", "key")
        assert result == "SerpAPI result"
        mock_ddgs.assert_not_called()

    @patch("agentic_internet.tools.web_search._search_serpapi")
    @patch("agentic_internet.tools.web_search._search_ddgs")
    def test_falls_back_to_ddgs(self, mock_ddgs, mock_serpapi):
        mock_serpapi.return_value = None
        mock_ddgs.return_value = "DDG result"
        result = _search_with_fallback("query", "key")
        assert result == "DDG result"

    @patch("agentic_internet.tools.web_search._search_ddgs")
    def test_skips_serpapi_without_key(self, mock_ddgs):
        mock_ddgs.return_value = "DDG result"
        result = _search_with_fallback("query", None)
        assert result == "DDG result"


class TestWebSearchTool:
    @patch("agentic_internet.tools.web_search._search_with_fallback")
    def test_forward_calls_fallback(self, mock_fallback):
        mock_fallback.return_value = "results"
        tool = WebSearchTool.__new__(WebSearchTool)
        tool.serpapi_key = None
        tool.use_orchestrator = False
        tool.orchestrator = None
        result = tool.forward("test query")
        assert result == "results"


class TestWebScraperTool:
    def test_rejects_invalid_url(self):
        tool = WebScraperTool()
        result = tool.forward("not-a-url")
        assert "Invalid" in result or "scheme" in result.lower()

    def test_rejects_ftp_url(self):
        tool = WebScraperTool()
        result = tool.forward("ftp://example.com/file")
        assert "Invalid" in result or "scheme" in result.lower()

    @patch("agentic_internet.tools.web_search.httpx.Client")
    def test_successful_scrape(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Hello World</p></body></html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        tool = WebScraperTool()
        result = tool.forward("https://example.com")
        assert "Hello World" in result


class TestNewsSearchTool:
    @patch("agentic_internet.tools.web_search._search_with_fallback")
    def test_forward_uses_news_params(self, mock_fallback):
        mock_fallback.return_value = "news results"
        tool = NewsSearchTool.__new__(NewsSearchTool)
        tool.serpapi_key = "key"
        result = tool.forward("test news")
        mock_fallback.assert_called_once_with("test news", "key", tbm="nws", news=True)
        assert result == "news results"
