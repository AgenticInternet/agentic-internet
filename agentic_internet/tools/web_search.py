"""Web search tool for agents using SerpAPI (with DuckDuckGo fallback)."""

from typing import Optional, List, Dict, Any
from smolagents import Tool
import os
import httpx
from bs4 import BeautifulSoup
from rich.console import Console

# Try to import both search libraries
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

console = Console()

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
            "description": "The search query to look up on the web"
        }
    }
    output_type = "string"
    
    def __init__(self, use_orchestrator: bool = False, orchestrator=None):
        """Initialize the web search tool.
        
        Args:
            use_orchestrator: Whether to use orchestrated search with multiple agents
            orchestrator: Optional SearchOrchestrator instance to use
        """
        super().__init__()
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        self.use_serpapi = HAS_SERPAPI and self.serpapi_key
        self.use_orchestrator = use_orchestrator
        self.orchestrator = orchestrator
        
        if not HAS_SERPAPI and not HAS_DDGS:
            console.print(
                "[yellow]Warning: Neither SerpAPI nor DuckDuckGo search libraries are installed. "
                "Install with: pip install google-search-results duckduckgo-search[/yellow]"
            )
    
    def _search_with_serpapi(self, query: str) -> str:
        """Search using SerpAPI."""
        try:
            search = GoogleSearch({
                "q": query,
                "api_key": self.serpapi_key,
                "num": 5
            })
            results = search.get_dict()
            
            organic_results = results.get("organic_results", [])
            if not organic_results:
                return None  # Fall back to DuckDuckGo
            
            formatted_results = []
            for i, result in enumerate(organic_results[:5], 1):
                formatted_results.append(
                    f"{i}. **{result.get('title', 'No title')}**\n"
                    f"   {result.get('snippet', 'No description')}\n"
                    f"   URL: {result.get('link', 'No URL')}"
                )
            
            return "\n\n".join(formatted_results)
        except Exception as e:
            console.print(f"[yellow]SerpAPI search failed: {e}. Falling back to DuckDuckGo.[/yellow]")
            return None
    
    def _search_with_ddgs(self, query: str) -> str:
        """Search using DuckDuckGo."""
        if not HAS_DDGS:
            return "DuckDuckGo search is not available. Install with: pip install duckduckgo-search"
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                
                if not results:
                    return "No search results found."
                
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append(
                        f"{i}. **{result.get('title', 'No title')}**\n"
                        f"   {result.get('body', 'No description')}\n"
                        f"   URL: {result.get('href', 'No URL')}"
                    )
                
                return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Error performing DuckDuckGo search: {str(e)}"
    
    def forward(self, query: str) -> str:
        """Execute web search and return results."""
        # If orchestrator is enabled and available, use it
        if self.use_orchestrator and self.orchestrator:
            return self._search_with_orchestrator(query)
        
        # Otherwise use direct search
        # Try SerpAPI first if available
        if self.use_serpapi:
            result = self._search_with_serpapi(query)
            if result:
                return result
        
        # Fall back to DuckDuckGo
        return self._search_with_ddgs(query)
    
    def _search_with_orchestrator(self, query: str) -> str:
        """Search using the orchestrator with multiple agents."""
        try:
            # Import here to avoid circular dependency
            import json
            
            # Run orchestrated search
            results = self.orchestrator.search(query, parallel=True)
            
            # Check if we have synthesis
            if "synthesis" in results and results["synthesis"]:
                return f"**Orchestrated Search Results (Synthesized):**\n\n{results['synthesis']}"
            
            # Otherwise format individual agent results
            if "agent_results" in results and results["agent_results"]:
                formatted = ["**Orchestrated Search Results:**\n"]
                for agent_name, agent_data in results["agent_results"].items():
                    formatted.append(f"\n**{agent_name} ({agent_data.get('specialization', 'general')}):**")
                    formatted.append(agent_data.get('result', 'No results'))
                return "\n".join(formatted)
            
            # If no results, fall back to regular search
            console.print("[yellow]Orchestrator returned no results, falling back to direct search[/yellow]")
            if self.use_serpapi:
                result = self._search_with_serpapi(query)
                if result:
                    return result
            return self._search_with_ddgs(query)
            
        except Exception as e:
            console.print(f"[yellow]Orchestrated search failed: {e}. Falling back to direct search.[/yellow]")
            # Fall back to regular search
            if self.use_serpapi:
                result = self._search_with_serpapi(query)
                if result:
                    return result
            return self._search_with_ddgs(query)


class WebScraperTool(Tool):
    """
    Tool for scraping content from web pages.
    """
    name = "web_scraper"
    description = """
    Scrape and extract text content from a web page.
    Input should be a URL to scrape.
    Returns the extracted text content from the page.
    """
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the webpage to scrape"
        }
    }
    output_type = "string"
    
    def forward(self, url: str) -> str:
        """Scrape content from a web page."""
        try:
            # Use httpx for better async support
            with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                response = client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                )
                response.raise_for_status()
                
                # Parse HTML with BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text and clean it up
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Limit text length to avoid token limits
                max_length = 3000
                if len(text) > max_length:
                    text = text[:max_length] + "... [truncated]"
                
                return f"Content from {url}:\n\n{text}"
                
        except httpx.HTTPStatusError as e:
            return f"HTTP error {e.response.status_code} when accessing {url}"
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"


class NewsSearchTool(Tool):
    """
    Tool for searching recent news articles using SerpAPI (preferred) or DuckDuckGo (fallback).
    """
    name = "news_search"
    description = """
    Search for recent news articles on a topic.
    Input should be a search query string.
    Returns recent news articles with titles, descriptions, and sources.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The news topic to search for"
        }
    }
    output_type = "string"
    
    def __init__(self):
        """Initialize the news search tool."""
        super().__init__()
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        self.use_serpapi = HAS_SERPAPI and self.serpapi_key
        
        if not HAS_SERPAPI and not HAS_DDGS:
            console.print(
                "[yellow]Warning: Neither SerpAPI nor DuckDuckGo search libraries are installed. "
                "Install with: pip install google-search-results duckduckgo-search[/yellow]"
            )
    
    def _search_news_with_serpapi(self, query: str) -> str:
        """Search news using SerpAPI."""
        try:
            search = GoogleSearch({
                "q": query,
                "api_key": self.serpapi_key,
                "tbm": "nws",  # News search
                "num": 5
            })
            results = search.get_dict()
            
            news_results = results.get("news_results", [])
            if not news_results:
                return None  # Fall back to DuckDuckGo
            
            formatted_results = []
            for i, article in enumerate(news_results[:5], 1):
                # Handle source field - it might be a string or dict
                source = article.get('source', 'Unknown')
                if isinstance(source, dict):
                    source_name = source.get('name', 'Unknown')
                else:
                    source_name = str(source) if source else 'Unknown'
                
                formatted_results.append(
                    f"{i}. **{article.get('title', 'No title')}**\n"
                    f"   Source: {source_name}\n"
                    f"   Date: {article.get('date', 'Unknown date')}\n"
                    f"   {article.get('snippet', 'No description')}\n"
                    f"   URL: {article.get('link', 'No URL')}"
                )
            
            return "\n\n".join(formatted_results)
        except Exception as e:
            console.print(f"[yellow]SerpAPI news search failed: {e}. Falling back to DuckDuckGo.[/yellow]")
            return None
    
    def _search_news_with_ddgs(self, query: str) -> str:
        """Search news using DuckDuckGo."""
        if not HAS_DDGS:
            return "DuckDuckGo search is not available. Install with: pip install duckduckgo-search"
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=5))
                
                if not results:
                    return "No news articles found."
                
                formatted_results = []
                for i, article in enumerate(results, 1):
                    formatted_results.append(
                        f"{i}. **{article.get('title', 'No title')}**\n"
                        f"   Source: {article.get('source', 'Unknown')}\n"
                        f"   Date: {article.get('date', 'Unknown date')}\n"
                        f"   {article.get('body', 'No description')}\n"
                        f"   URL: {article.get('url', 'No URL')}"
                    )
                
                return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Error searching news with DuckDuckGo: {str(e)}"
    
    def forward(self, query: str) -> str:
        """Search for news articles."""
        # Try SerpAPI first if available
        if self.use_serpapi:
            result = self._search_news_with_serpapi(query)
            if result:
                return result
        
        # Fall back to DuckDuckGo
        return self._search_news_with_ddgs(query)
