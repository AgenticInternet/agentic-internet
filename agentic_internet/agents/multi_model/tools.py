"""SerpAPI tool adapters and worker-as-tool wrapper."""

# ruff: noqa: RUF012, RUF015 - preserve the extracted smolagents implementation.

import json
from typing import Any

from serpapi import BaiduSearch, BingSearch, GoogleScholarSearch, GoogleSearch, YahooSearch
from smolagents import Tool


# Advanced SerpAPI Tools
class GoogleSearchTool(Tool):
    name = "google_search"
    description = "Advanced Google search with location, language, and filter support using SerpAPI."
    inputs = {
        "query": {"type": "string", "description": "Search query"},
        "location": {"type": "string", "description": "Geographic location for search (optional)", "nullable": True},
        "language": {"type": "string", "description": "Language code (default: 'en')", "nullable": True},
        "country": {"type": "string", "description": "Country code (default: 'us')", "nullable": True},
        "num_results": {"type": "integer", "description": "Number of results (default: 10)", "nullable": True},
        "date_filter": {
            "type": "string",
            "description": "Date filter: qdr:d (day), qdr:w (week), qdr:m (month), qdr:y (year)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, api_key: str, context_engine=None):
        super().__init__()
        self.api_key = api_key
        self.context_engine = context_engine

    def forward(
        self,
        query: str,
        location: str | None = None,
        language: str = "en",
        country: str = "us",
        num_results: int = 10,
        date_filter: str | None = None,
    ) -> str:
        """Execute Google search with advanced parameters"""
        if not self.api_key:
            return "SerpAPI client not initialized. Please provide an API key."

        try:
            params: dict[str, Any] = {
                "api_key": self.api_key,
                "q": query,
                "hl": language,
                "gl": country,
                "num": num_results,
            }

            if location:
                params["location"] = location
            if date_filter:
                params["tbs"] = date_filter

            # Get recommendations from memory if context engine exists
            if self.context_engine:
                recommendations = self.context_engine.memory.get_search_recommendations("web_search", "google")
                if recommendations.get("best_params"):
                    params = {**recommendations["best_params"], **params}

            search = GoogleSearch(params)
            results = search.get_dict()

            # Process results
            organic_results = results.get("organic_results", [])
            processed_results = []

            for i, result in enumerate(organic_results):
                processed_results.append(
                    {
                        "position": i + 1,
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "source": result.get("source", ""),
                    }
                )

            # Update context and memory if available
            if self.context_engine:
                success_metrics = {
                    "success": len(processed_results) > 0,
                    "result_count": len(processed_results),
                    "params": params,
                }
                self.context_engine.memory.remember_search_pattern("web_search", "google", success_metrics)

                if hasattr(self.context_engine, "current_task_context") and self.context_engine.current_task_context:
                    self.context_engine.current_task_context.log_search("google", query, len(processed_results), True)

            return json.dumps({"results": processed_results, "total": len(processed_results)}, indent=2)

        except Exception as e:
            error_msg = f"Google search failed: {e!s}"
            if (
                self.context_engine
                and hasattr(self.context_engine, "current_task_context")
                and self.context_engine.current_task_context
            ):
                self.context_engine.current_task_context.log_search("google", query, 0, False)
            return error_msg


class GoogleShoppingTool(Tool):
    name = "google_shopping"
    description = "Search Google Shopping for products with prices and reviews."
    inputs = {
        "query": {"type": "string", "description": "Product search query"},
        "location": {"type": "string", "description": "Shopping location", "nullable": True},
        "min_price": {"type": "integer", "description": "Minimum price filter", "nullable": True},
        "max_price": {"type": "integer", "description": "Maximum price filter", "nullable": True},
        "sort_by": {"type": "string", "description": "Sort by: r (rating), rv (reviews), p (price)", "nullable": True},
    }
    output_type = "string"

    def __init__(self, api_key: str, context_engine=None):
        super().__init__()
        self.api_key = api_key
        self.context_engine = context_engine

    def forward(
        self,
        query: str,
        location: str | None = None,
        min_price: int | None = None,
        max_price: int | None = None,
        sort_by: str | None = None,
    ) -> str:
        """Search Google Shopping with filters"""
        if not self.api_key:
            return "SerpAPI client not initialized. Please provide an API key."

        try:
            params: dict[str, Any] = {"api_key": self.api_key, "q": query}

            if location:
                params["location"] = location
            if min_price:
                params["min_price"] = min_price
            if max_price:
                params["max_price"] = max_price
            if sort_by:
                params["sort_by"] = sort_by

            # Google Shopping needs to be handled through GoogleSearch with shopping params
            search = GoogleSearch(params)
            search.params_dict["tbm"] = "shop"
            results = search.get_dict()
            shopping_results = results.get("shopping_results", [])

            processed_results = []
            for result in shopping_results:
                processed_results.append(
                    {
                        "title": result.get("title", ""),
                        "price": result.get("price", ""),
                        "source": result.get("source", ""),
                        "link": result.get("link", ""),
                        "rating": result.get("rating"),
                        "reviews": result.get("reviews"),
                        "shipping": result.get("shipping"),
                    }
                )

            # Update memory if available
            if self.context_engine:
                success_metrics = {
                    "success": len(processed_results) > 0,
                    "result_count": len(processed_results),
                    "params": params,
                }
                self.context_engine.memory.remember_search_pattern("shopping", "google", success_metrics)

            return json.dumps({"shopping_results": processed_results, "total": len(processed_results)}, indent=2)

        except Exception as e:
            return f"Google Shopping search failed: {e!s}"


class GoogleMapsLocalTool(Tool):
    name = "google_maps_local"
    description = "Search Google Maps for local businesses and places."
    inputs = {
        "query": {"type": "string", "description": "Local business search query"},
        "location": {"type": "string", "description": "Search location (required)"},
        "type": {"type": "string", "description": "Place type filter (restaurant, hotel, etc.)", "nullable": True},
    }
    output_type = "string"

    def __init__(self, api_key: str, context_engine=None):
        super().__init__()
        self.api_key = api_key
        self.context_engine = context_engine

    def forward(self, query: str, location: str, type: str | None = None) -> str:
        """Search Google Maps for local businesses"""
        if not self.api_key:
            return "SerpAPI client not initialized. Please provide an API key."

        try:
            params = {
                "api_key": self.api_key,
                "q": query,
                "ll": f"@{location}" if "," not in location else location,
                "type": "search",
            }

            if type:
                params["type"] = type

            # Google Maps search through regular Google Search
            search = GoogleSearch(params)
            results = search.get_dict()
            local_results = results.get("local_results", [])

            processed_results = []
            for result in local_results:
                processed_results.append(
                    {
                        "title": result.get("title", ""),
                        "address": result.get("address", ""),
                        "phone": result.get("phone"),
                        "rating": result.get("rating"),
                        "reviews": result.get("reviews"),
                        "website": result.get("website"),
                        "place_id": result.get("place_id"),
                    }
                )

            return json.dumps({"local_results": processed_results, "total": len(processed_results)}, indent=2)

        except Exception as e:
            return f"Google Maps search failed: {e!s}"


class GoogleScholarTool(Tool):
    name = "google_scholar"
    description = "Search Google Scholar for academic papers and citations."
    inputs = {
        "query": {"type": "string", "description": "Academic search query"},
        "year_low": {"type": "integer", "description": "Start year filter", "nullable": True},
        "year_high": {"type": "integer", "description": "End year filter", "nullable": True},
        "sort_by": {"type": "string", "description": "Sort by: relevance or date", "nullable": True},
    }
    output_type = "string"

    def __init__(self, api_key: str, context_engine=None):
        super().__init__()
        self.api_key = api_key
        self.context_engine = context_engine

    def forward(
        self, query: str, year_low: int | None = None, year_high: int | None = None, sort_by: str | None = None
    ) -> str:
        """Search Google Scholar for academic content"""
        if not self.api_key:
            return "SerpAPI client not initialized. Please provide an API key."

        try:
            params: dict[str, Any] = {"api_key": self.api_key, "q": query}

            if year_low:
                params["as_ylo"] = year_low
            if year_high:
                params["as_yhi"] = year_high
            if sort_by == "date":
                params["scisbd"] = 1

            # Use GoogleScholarSearch for Scholar queries
            search = GoogleScholarSearch(params)
            results = search.get_dict()
            organic_results = results.get("organic_results", [])

            processed_results = []
            for result in organic_results:
                processed_results.append(
                    {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "publication_info": result.get("publication_info", {}),
                    }
                )

            return json.dumps({"scholar_results": processed_results, "total": len(processed_results)}, indent=2)

        except Exception as e:
            return f"Google Scholar search failed: {e!s}"


class MultiEngineSearchTool(Tool):
    name = "multi_engine_search"
    description = "Search across multiple search engines (Google, Bing, Yahoo, Baidu) and compare results."
    inputs = {
        "query": {"type": "string", "description": "Search query"},
        "engines": {
            "type": "string",
            "description": "Comma-separated engines (google,bing,yahoo,baidu)",
            "nullable": True,
        },
        "compare": {"type": "boolean", "description": "Whether to compare results across engines", "nullable": True},
    }
    output_type = "string"

    def __init__(self, api_key: str, context_engine=None):
        super().__init__()
        self.api_key = api_key
        self.context_engine = context_engine

    def forward(self, query: str, engines: str = "google,bing", compare: bool = True) -> str:
        """Search multiple engines and optionally compare results"""
        if not self.api_key:
            return "SerpAPI client not initialized. Please provide an API key."

        try:
            engine_list = [e.strip() for e in engines.split(",")]
            all_results: dict[str, Any] = {}

            for engine in engine_list:
                try:
                    params: dict[str, Any] = {"api_key": self.api_key, "q": query}

                    # Use appropriate search class for each engine
                    if engine == "google":
                        search = GoogleSearch(params)
                    elif engine == "bing":
                        search = BingSearch(params)
                    elif engine == "yahoo":
                        search = YahooSearch(params)
                    elif engine == "baidu":
                        search = BaiduSearch(params)
                    else:
                        all_results[engine] = f"Error: Unsupported engine {engine}"
                        continue

                    results = search.get_dict()

                    organic_results = results.get("organic_results", [])
                    processed = [
                        {
                            "position": i + 1,
                            "title": r.get("title", ""),
                            "link": r.get("link", ""),
                            "snippet": r.get("snippet", ""),
                        }
                        for i, r in enumerate(organic_results[:10])
                    ]

                    all_results[engine] = processed

                    # Store results for cross-engine comparison
                    if (
                        self.context_engine
                        and hasattr(self.context_engine, "current_task_context")
                        and self.context_engine.current_task_context
                    ):
                        if engine not in self.context_engine.current_task_context.cross_engine_results:
                            self.context_engine.current_task_context.cross_engine_results[engine] = []
                        self.context_engine.current_task_context.cross_engine_results[engine].extend(processed)

                except Exception as e:
                    all_results[engine] = f"Error: {e!s}"

            if compare and len(all_results) > 1:
                # Find common results across engines
                common_links = self._find_common_results(all_results)
                all_results["comparison"] = {
                    "common_results": common_links,
                    "engine_coverage": {
                        engine: len(results) if isinstance(results, list) else 0
                        for engine, results in all_results.items()
                        if engine != "comparison"
                    },
                }

            return json.dumps(all_results, indent=2)

        except Exception as e:
            return f"Multi-engine search failed: {e!s}"

    def _find_common_results(self, all_results: dict) -> list[dict]:
        """Find results that appear across multiple engines"""
        common_results = []
        engine_results = {k: v for k, v in all_results.items() if isinstance(v, list)}

        if len(engine_results) < 2:
            return []

        # Get first engine results as baseline
        baseline_engine = list(engine_results.keys())[0]
        baseline_results = engine_results[baseline_engine]

        for result in baseline_results:
            result_link = result.get("link", "")
            if not result_link:
                continue

            # Check if this link appears in other engines
            appearances = [baseline_engine]
            for other_engine, other_results in engine_results.items():
                if other_engine == baseline_engine:
                    continue

                for other_result in other_results:
                    if other_result.get("link", "") == result_link:
                        appearances.append(other_engine)
                        break

            if len(appearances) > 1:
                common_results.append({**result, "appears_in": appearances, "cross_engine_rank": len(appearances)})

        return sorted(common_results, key=lambda x: x["cross_engine_rank"], reverse=True)


class AgentTool(Tool):
    """Wrapper to use agents as tools for orchestration"""

    def __init__(self, agent, agent_name: str, agent_description: str):
        self.name = agent_name
        self.description = agent_description
        self.inputs = {"task": {"type": "string", "description": "Task to delegate to this specialized agent"}}
        self.output_type = "string"
        super().__init__()
        self.agent = agent

    def forward(self, task: str) -> str:
        """Execute the agent with the given task"""
        try:
            if self.agent:
                result = self.agent.run(task)
                return str(result)
            return f"Agent {self.name} is not available"
        except Exception as e:
            return f"Agent {self.name} failed: {e!s}"
