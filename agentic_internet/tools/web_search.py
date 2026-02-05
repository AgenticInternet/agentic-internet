"""Web search tool for agents using SerpAPI (with DuckDuckGo fallback)."""

import logging
import os
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from smolagents import Tool

logger = logging.getLogger(__name__)

try:
    from serpapi import GoogleSearch
    HAS_SERPAPI = True
except ImportError:
    HAS_SERPAPI = False

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

def _search_serpapi(query: str, serpapi_key: str, *, tbm: str | None = None, num: int = 5) -> str | None:
    """Shared SerpAPI search logic. Returns formatted results or None on failure."""
    if not HAS_SERPAPI or not serpapi_key:
        return None
    try:
        params: dict[str, Any] = {"q": query, "api_key": serpapi_key, "num": num}
        if tbm:
            params["tbm"] = tbm
        search = GoogleSearch(params)
        results = search.get_dict()

        result_key = "news_results" if tbm == "nws" else "organic_results"
        items = results.get(result_key, [])
        if not items:
            return None

        formatted = []
        for i, item in enumerate(items[:num], 1):
            title = item.get("title", "No title")
            snippet = item.get("snippet", item.get("body", "No description"))
            link = item.get("link", item.get("href", "No URL"))

            parts = [f"{i}. **{title}**"]
            source = item.get("source")
            if source:
                source_name = source.get("name", str(source)) if isinstance(source, dict) else str(source)
                parts.append(f"   Source: {source_name}")
            date = item.get("date")
            if date:
                parts.append(f"   Date: {date}")
            parts.append(f"   {snippet}")
            parts.append(f"   URL: {link}")
            formatted.append("\n".join(parts))

        return "\n\n".join(formatted)
    except Exception as e:
        logger.warning("SerpAPI search failed: %s", e)
        return None


def _search_ddgs(query: str, *, news: bool = False, num: int = 5) -> str:
    """Shared DuckDuckGo search logic."""
    if not HAS_DDGS:
        return "DuckDuckGo search is not available. Install with: pip install duckduckgo-search"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=num)) if news else list(ddgs.text(query, max_results=num))

            if not results:
                return "No search results found."

            formatted = []
            for i, item in enumerate(results, 1):
                title = item.get("title", "No title")
                body = item.get("body", item.get("snippet", "No description"))
                link = item.get("href", item.get("url", "No URL"))

                parts = [f"{i}. **{title}**"]
                source = item.get("source")
                if source:
                    parts.append(f"   Source: {source}")
                date = item.get("date")
                if date:
                    parts.append(f"   Date: {date}")
                parts.append(f"   {body}")
                parts.append(f"   URL: {link}")
                formatted.append("\n".join(parts))

            return "\n\n".join(formatted)
    except Exception as e:
        return f"Error performing DuckDuckGo search: {e}"


def _search_with_fallback(
    query: str, serpapi_key: str | None, *, tbm: str | None = None, news: bool = False
) -> str:
    """Try SerpAPI first, fall back to DuckDuckGo."""
    if serpapi_key:
        result = _search_serpapi(query, serpapi_key, tbm=tbm)
        if result:
            return result
    return _search_ddgs(query, news=news)


class WebSearchTool(Tool):
    """
    Tool for searching the web using SerpAPI (preferred) or DuckDuckGo (fallback).
    Can use orchestrated search with multiple agents if enabled.
    """

    name = "web_search"
    description = """
    Search the web for information using SerpAPI (if available) or DuckDuckGo.
    Input should be a search query string.
    Returns a list of search results with titles, snippets, and URLs.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to look up on the web",
        }
    }
    output_type = "string"

    def __init__(self, use_orchestrator: bool = False, orchestrator: Any = None):
        super().__init__()
        self.serpapi_key: str | None = os.getenv("SERPAPI_API_KEY")
        self.use_orchestrator = use_orchestrator
        self.orchestrator = orchestrator

        if not HAS_SERPAPI and not HAS_DDGS:
            logger.warning(
                "Neither SerpAPI nor DuckDuckGo search libraries are installed. "
                "Install with: pip install google-search-results duckduckgo-search"
            )

    def forward(self, query: str) -> str:
        """Execute web search and return results."""
        if self.use_orchestrator and self.orchestrator:
            return self._search_with_orchestrator(query)
        return _search_with_fallback(query, self.serpapi_key)

    def _search_with_orchestrator(self, query: str) -> str:
        """Search using the orchestrator with multiple agents."""

        try:
            results = self.orchestrator.search(query, parallel=True)

            if results.get("synthesis"):
                return f"**Orchestrated Search Results (Synthesized):**\n\n{results['synthesis']}"

            if results.get("agent_results"):
                formatted = ["**Orchestrated Search Results:**\n"]
                for agent_name, agent_data in results["agent_results"].items():
                    formatted.append(f"\n**{agent_name} ({agent_data.get('specialization', 'general')}):**")
                    formatted.append(agent_data.get("result", "No results"))
                return "\n".join(formatted)

            logger.warning("Orchestrator returned no results, falling back to direct search")
        except Exception as e:
            logger.warning("Orchestrated search failed: %s", e)

        return _search_with_fallback(query, self.serpapi_key)


def _validate_url(url: str) -> str | None:
    """Validate and normalise a URL. Returns error message or None if valid."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"Invalid URL scheme '{parsed.scheme}'. Only http and https are supported."
        if not parsed.netloc:
            return "Invalid URL: missing hostname."
    except Exception:
        return f"Malformed URL: {url}"
    return None


class WebScraperTool(Tool):
    """Tool for scraping content from web pages."""

    name = "web_scraper"
    description = """
    Scrape and extract text content from a web page.
    Input should be a URL to scrape.
    Returns the extracted text content from the page.
    """
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the webpage to scrape",
        }
    }
    output_type = "string"

    MAX_CONTENT_LENGTH = 3000
    REQUEST_TIMEOUT = 30.0

    def forward(self, url: str) -> str:
        """Scrape content from a web page."""
        error = _validate_url(url)
        if error:
            return error

        try:
            with httpx.Client(follow_redirects=True, timeout=self.REQUEST_TIMEOUT) as client:
                response = client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                )
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                for element in soup(["script", "style"]):
                    element.decompose()

                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = " ".join(chunk for chunk in chunks if chunk)

                if len(text) > self.MAX_CONTENT_LENGTH:
                    text = text[: self.MAX_CONTENT_LENGTH] + "... [truncated]"

                return f"Content from {url}:\n\n{text}"

        except httpx.HTTPStatusError as e:
            return f"HTTP error {e.response.status_code} when accessing {url}"
        except Exception as e:
            return f"Error scraping {url}: {e}"


class NewsSearchTool(Tool):
    """Tool for searching recent news articles using SerpAPI (preferred) or DuckDuckGo (fallback)."""

    name = "news_search"
    description = """
    Search for recent news articles on a topic.
    Input should be a search query string.
    Returns recent news articles with titles, descriptions, and sources.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The news topic to search for",
        }
    }
    output_type = "string"

    def __init__(self) -> None:
        super().__init__()
        self.serpapi_key: str | None = os.getenv("SERPAPI_API_KEY")

        if not HAS_SERPAPI and not HAS_DDGS:
            logger.warning(
                "Neither SerpAPI nor DuckDuckGo search libraries are installed. "
                "Install with: pip install google-search-results duckduckgo-search"
            )

    def forward(self, query: str) -> str:
        """Search for news articles."""
        return _search_with_fallback(query, self.serpapi_key, tbm="nws", news=True)
