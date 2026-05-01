"""Exa AI-powered search tool for agents using neural and keyword search."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from smolagents import Tool

logger = logging.getLogger(__name__)

try:
    from exa_py import Exa
    HAS_EXA = True
except ImportError:
    HAS_EXA = False

INTEGRATION_NAME = "agentic-internet"

VALID_SEARCH_TYPES = ("auto", "neural", "fast", "deep-lite", "deep", "deep-reasoning", "instant")
VALID_CATEGORIES = (
    "company",
    "research paper",
    "news",
    "personal site",
    "financial report",
    "people",
)


@dataclass
class ExaResult:
    """Typed view of a single Exa search result."""

    title: str
    url: str
    text: str | None = None
    summary: str | None = None
    highlights: list[str] = field(default_factory=list)
    author: str | None = None
    published_date: str | None = None
    score: float | None = None

    @classmethod
    def from_sdk(cls, raw: Any) -> "ExaResult":
        """Build from an exa_py Result object (or anything with the expected attrs)."""
        highlights = getattr(raw, "highlights", None) or []
        return cls(
            title=getattr(raw, "title", None) or "No title",
            url=getattr(raw, "url", None) or "",
            text=getattr(raw, "text", None),
            summary=getattr(raw, "summary", None),
            highlights=list(highlights),
            author=getattr(raw, "author", None),
            published_date=getattr(raw, "published_date", None),
            score=getattr(raw, "score", None),
        )

    def snippet(self, max_chars: int = 500) -> str:
        """Pick the best available content for a short snippet, cascading through fields."""
        if self.summary:
            content = self.summary
        elif self.highlights:
            content = " ... ".join(self.highlights)
        elif self.text:
            content = self.text
        else:
            return "No description"
        return content if len(content) <= max_chars else content[:max_chars] + "..."


def _build_client(api_key: str) -> Any:
    """Create an Exa client and tag it with the integration header."""
    client = Exa(api_key=api_key)
    client.headers["x-exa-integration"] = INTEGRATION_NAME
    return client


def _format_results(results: list[ExaResult]) -> str:
    """Format Exa results matching the existing web_search output style."""
    if not results:
        return "No search results found."

    formatted = []
    for i, r in enumerate(results, 1):
        parts = [f"{i}. **{r.title}**"]
        if r.author:
            parts.append(f"   Author: {r.author}")
        if r.published_date:
            parts.append(f"   Date: {r.published_date}")
        parts.append(f"   {r.snippet()}")
        parts.append(f"   URL: {r.url}")
        formatted.append("\n".join(parts))
    return "\n\n".join(formatted)


def _build_contents_kwargs(
    *,
    text: bool,
    text_max_chars: int | None,
    highlights: bool,
    summary: bool,
    summary_query: str | None = None,
) -> dict[str, Any]:
    """Build the contents kwargs for search_and_contents.

    The Exa API allows multiple content types to be requested simultaneously.
    """
    kwargs: dict[str, Any] = {}
    if text:
        kwargs["text"] = {"max_characters": text_max_chars} if text_max_chars else True
    if highlights:
        kwargs["highlights"] = True
    if summary:
        kwargs["summary"] = {"query": summary_query} if summary_query else True
    return kwargs


def _normalize_search_type(search_type: str | None) -> str | None:
    """Validate the search type, fall back to None (Exa default) if unrecognized."""
    if search_type is None:
        return None
    if search_type in VALID_SEARCH_TYPES:
        return search_type
    logger.warning("Unknown Exa search type %r, falling back to default", search_type)
    return None


def _normalize_category(category: str | None) -> str | None:
    """Validate the category, fall back to None if unrecognized."""
    if category is None or category == "":
        return None
    if category in VALID_CATEGORIES:
        return category
    logger.warning("Unknown Exa category %r, ignoring", category)
    return None


class ExaSearchTool(Tool):
    """
    Tool for searching the web using Exa's AI-powered neural search.

    Exa provides embeddings-based semantic search with optional summaries,
    highlights, and full-text content. Works well for research-style queries
    where keyword matching is not enough.
    """

    name = "exa_search"
    description = """
    Search the web using Exa's AI-powered neural search.
    Input is a search query. Optionally specify search_type (auto, neural, fast),
    a category filter (e.g. "research paper", "news", "company"),
    or domain filters. Returns ranked results with titles, summaries, and URLs.
    Best for research, semantic, and intent-based queries.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to look up on the web",
        },
        "search_type": {
            "type": "string",
            "description": (
                "Search method: 'auto' (default), 'neural', 'fast', or 'deep'. "
                "Use 'neural' for semantic queries, 'fast' for low-latency keyword-style search."
            ),
            "nullable": True,
        },
        "category": {
            "type": "string",
            "description": (
                "Optional content category filter: 'company', 'research paper', "
                "'news', 'personal site', 'financial report', 'people'."
            ),
            "nullable": True,
        },
        "include_domains": {
            "type": "array",
            "description": "Optional list of domains to restrict results to (e.g. ['arxiv.org']).",
            "nullable": True,
        },
        "exclude_domains": {
            "type": "array",
            "description": "Optional list of domains to exclude from results.",
            "nullable": True,
        },
        "start_published_date": {
            "type": "string",
            "description": "Optional ISO 8601 date; only return results published on/after this date.",
            "nullable": True,
        },
        "end_published_date": {
            "type": "string",
            "description": "Optional ISO 8601 date; only return results published on/before this date.",
            "nullable": True,
        },
    }
    output_type = "string"

    DEFAULT_NUM_RESULTS = 5
    DEFAULT_TEXT_MAX_CHARS = 500

    def __init__(
        self,
        *,
        num_results: int | None = None,
        include_text: bool = False,
        include_highlights: bool = True,
        include_summary: bool = True,
        summary_query: str | None = None,
        text_max_chars: int | None = None,
    ):
        super().__init__()
        self.api_key: str | None = os.getenv("EXA_API_KEY")
        self.num_results = num_results or self.DEFAULT_NUM_RESULTS
        self.include_text = include_text
        self.include_highlights = include_highlights
        self.include_summary = include_summary
        self.summary_query = summary_query
        self.text_max_chars = text_max_chars or self.DEFAULT_TEXT_MAX_CHARS

        if not HAS_EXA:
            logger.warning(
                "exa-py is not installed. Install with: pip install exa-py"
            )

    def is_available(self) -> bool:
        """Check whether the tool can run (SDK installed and API key set)."""
        return HAS_EXA and bool(self.api_key)

    def forward(
        self,
        query: str,
        search_type: str | None = None,
        category: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
    ) -> str:
        """Execute Exa search and return formatted results."""
        if not HAS_EXA:
            return "Exa search is not available. Install with: pip install exa-py"
        if not self.api_key:
            return "Exa search is not available. Set EXA_API_KEY in your environment."

        try:
            client = _build_client(self.api_key)
            contents_kwargs = _build_contents_kwargs(
                text=self.include_text,
                text_max_chars=self.text_max_chars,
                highlights=self.include_highlights,
                summary=self.include_summary,
                summary_query=self.summary_query,
            )

            kwargs: dict[str, Any] = {
                "query": query,
                "num_results": self.num_results,
                **contents_kwargs,
            }
            normalized_type = _normalize_search_type(search_type)
            if normalized_type:
                kwargs["type"] = normalized_type
            normalized_category = _normalize_category(category)
            if normalized_category:
                kwargs["category"] = normalized_category
            if include_domains:
                kwargs["include_domains"] = include_domains
            if exclude_domains:
                kwargs["exclude_domains"] = exclude_domains
            if start_published_date:
                kwargs["start_published_date"] = start_published_date
            if end_published_date:
                kwargs["end_published_date"] = end_published_date

            response = client.search_and_contents(**kwargs)
            raw_results = getattr(response, "results", None) or []
            results = [ExaResult.from_sdk(r) for r in raw_results]
            return _format_results(results)
        except Exception as e:
            logger.warning("Exa search failed: %s", e)
            return f"Error performing Exa search: {e}"


class ExaFindSimilarTool(Tool):
    """Tool for finding pages semantically similar to a given URL using Exa."""

    name = "exa_find_similar"
    description = """
    Find web pages semantically similar to a given URL using Exa.
    Useful for "more like this" discovery, finding related research papers,
    competitors of a company, or articles covering the same topic.
    Returns ranked similar pages with titles, summaries, and URLs.
    """
    inputs = {
        "url": {
            "type": "string",
            "description": "The reference URL to find similar pages to.",
        },
        "exclude_source_domain": {
            "type": "boolean",
            "description": "If true, exclude results from the same domain as the source URL.",
            "nullable": True,
        },
    }
    output_type = "string"

    DEFAULT_NUM_RESULTS = 5
    DEFAULT_TEXT_MAX_CHARS = 500

    def __init__(
        self,
        *,
        num_results: int | None = None,
        include_text: bool = False,
        include_highlights: bool = True,
        include_summary: bool = True,
        summary_query: str | None = None,
        text_max_chars: int | None = None,
    ):
        super().__init__()
        self.api_key: str | None = os.getenv("EXA_API_KEY")
        self.num_results = num_results or self.DEFAULT_NUM_RESULTS
        self.include_text = include_text
        self.include_highlights = include_highlights
        self.include_summary = include_summary
        self.summary_query = summary_query
        self.text_max_chars = text_max_chars or self.DEFAULT_TEXT_MAX_CHARS

        if not HAS_EXA:
            logger.warning(
                "exa-py is not installed. Install with: pip install exa-py"
            )

    def is_available(self) -> bool:
        return HAS_EXA and bool(self.api_key)

    def forward(self, url: str, exclude_source_domain: bool | None = None) -> str:
        """Execute Exa find_similar and return formatted results."""
        if not HAS_EXA:
            return "Exa search is not available. Install with: pip install exa-py"
        if not self.api_key:
            return "Exa search is not available. Set EXA_API_KEY in your environment."

        try:
            client = _build_client(self.api_key)
            contents_kwargs = _build_contents_kwargs(
                text=self.include_text,
                text_max_chars=self.text_max_chars,
                highlights=self.include_highlights,
                summary=self.include_summary,
                summary_query=self.summary_query,
            )

            kwargs: dict[str, Any] = {
                "url": url,
                "num_results": self.num_results,
                **contents_kwargs,
            }
            if exclude_source_domain is not None:
                kwargs["exclude_source_domain"] = exclude_source_domain

            response = client.find_similar_and_contents(**kwargs)
            raw_results = getattr(response, "results", None) or []
            results = [ExaResult.from_sdk(r) for r in raw_results]
            return _format_results(results)
        except Exception as e:
            logger.warning("Exa find_similar failed: %s", e)
            return f"Error performing Exa find_similar: {e}"
