"""Memory and context primitives for multi-model orchestration."""

# ruff: noqa: B007, RUF005 - preserve the extracted implementation verbatim.

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

NON_REUSABLE_SEARCH_PARAMS = frozenset({"api_key", "q", "query"})


@dataclass
class ContextWindow:
    """Manages context window size and content prioritization"""

    max_tokens: int = 8192
    current_tokens: int = 0
    priority_content: list[str] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    compression_threshold: float = 0.8

    def add_content(self, content: str, priority: int = 1) -> bool:
        estimated_tokens = int(len(content.split()) * 1.3)

        if self.current_tokens + estimated_tokens > self.max_tokens * self.compression_threshold:
            self._compress_context()

        if priority > 3:
            self.priority_content.insert(0, content)
        else:
            self.priority_content.append(content)

        self.current_tokens += estimated_tokens
        return True

    def _compress_context(self):
        if len(self.priority_content) > 5:
            compressed_middle = f"[COMPRESSED: {len(self.priority_content[2:-2])} items summarized]"
            self.priority_content = self.priority_content[:2] + [compressed_middle] + self.priority_content[-2:]
            self.current_tokens = int(self.max_tokens * 0.6)


@dataclass
class AgentMemory:
    """Enhanced memory system with search pattern learning"""

    short_term: dict[str, Any] = field(default_factory=dict)
    long_term: dict[str, Any] = field(default_factory=dict)
    episodic: list[dict[str, Any]] = field(default_factory=list)
    semantic: dict[str, list[str]] = field(default_factory=dict)
    search_patterns: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_episodes: int = 50

    def remember_search_pattern(self, query_type: str, engine: str, success_metrics: dict[str, Any]):
        """Remember successful search patterns for different query types"""
        pattern_key = f"{query_type}_{engine}"
        if pattern_key not in self.search_patterns:
            self.search_patterns[pattern_key] = {
                "success_count": 0,
                "total_attempts": 0,
                "avg_results": 0,
                "best_params": {},
                "common_issues": [],
            }

        pattern = self.search_patterns[pattern_key]
        pattern["total_attempts"] += 1

        if success_metrics.get("success", False):
            pattern["success_count"] += 1
            pattern["avg_results"] = (pattern["avg_results"] + success_metrics.get("result_count", 0)) / 2
            if success_metrics.get("params"):
                pattern["best_params"] = {
                    key: value
                    for key, value in success_metrics["params"].items()
                    if key not in NON_REUSABLE_SEARCH_PARAMS
                }

    def get_search_recommendations(self, query_type: str, engine: str) -> dict[str, Any]:
        """Get recommendations based on past search patterns"""
        pattern_key = f"{query_type}_{engine}"
        if pattern_key in self.search_patterns:
            return self.search_patterns[pattern_key]
        return {}


@dataclass
class TaskContext:
    """Enhanced task context with search-specific tracking"""

    task_id: str
    objective: str
    current_step: int = 0
    total_steps: int = 0
    context_data: dict[str, Any] = field(default_factory=dict)
    progress_markers: list[str] = field(default_factory=list)
    search_history: list[dict[str, Any]] = field(default_factory=list)
    cross_engine_results: dict[str, list[dict]] = field(default_factory=dict)

    def log_search(self, engine: str, query: str, results_count: int, success: bool):
        """Log search attempts for analysis and optimization"""
        self.search_history.append(
            {
                "engine": engine,
                "query": query,
                "results_count": results_count,
                "success": success,
                "timestamp": datetime.now(),
                "step": self.current_step,
            }
        )

    def get_progress_summary(self) -> str:
        """Get a summary of current task progress"""
        return (
            f"Task: {self.task_id}, Step {self.current_step}/{self.total_steps}, Searches: {len(self.search_history)}"
        )


class ContextEngineeringMixin:
    """Enhanced mixin with SerpAPI context awareness"""

    def __init__(self):
        self.context_window = ContextWindow()
        self.memory = AgentMemory()
        self.current_task_context: TaskContext | None = None

    def prepare_search_context(self, query: str, search_type: str, engine: str) -> dict[str, Any]:
        """Prepare context-aware search parameters based on query and history"""
        context: dict[str, Any] = {
            "base_query": query,
            "search_type": search_type,
            "engine": engine,
            "recommendations": self.memory.get_search_recommendations(search_type, engine),
        }

        # Add relevant search history
        if self.current_task_context:
            relevant_searches = [
                s for s in self.current_task_context.search_history if s["engine"] == engine and s["success"]
            ]
            context["recent_successful_searches"] = relevant_searches[-3:]  # Last 3 successful

        return context

    def prepare_agent_context(self, task: str, agent_role: str) -> str:
        """Prepare context for agent initialization"""
        return f"""
Task: {task}
Role: {agent_role}
Memory Status: {len(self.memory.episodic)} episodes, {len(self.memory.search_patterns)} search patterns
Current Context: {self.context_window.current_tokens}/{self.context_window.max_tokens} tokens
"""

    def update_context_from_result(self, agent_name: str, task: str, result: Any, success: bool):
        """Update context based on agent results"""
        self.memory.episodic.append(
            {
                "agent": agent_name,
                "task": task,
                "result": str(result)[:500],  # Truncate for memory
                "success": success,
                "timestamp": datetime.now(),
            }
        )

        # Keep episodic memory bounded
        if len(self.memory.episodic) > self.memory.max_episodes:
            self.memory.episodic = self.memory.episodic[-self.memory.max_episodes :]

    def analyze_search_performance(self) -> dict[str, Any]:
        """Analyze search performance across engines and query types"""
        if not self.current_task_context or not self.current_task_context.search_history:
            return {"message": "No search history available"}

        analysis: dict[str, Any] = {
            "total_searches": len(self.current_task_context.search_history),
            "success_rate": 0,
            "engine_performance": {},
            "query_patterns": {},
            "recommendations": [],
        }

        successful_searches = [s for s in self.current_task_context.search_history if s["success"]]
        analysis["success_rate"] = len(successful_searches) / len(self.current_task_context.search_history)

        # Analyze by engine
        for search in self.current_task_context.search_history:
            engine = search["engine"]
            if engine not in analysis["engine_performance"]:
                analysis["engine_performance"][engine] = {"total": 0, "successful": 0, "avg_results": 0}

            perf = analysis["engine_performance"][engine]
            perf["total"] += 1
            if search["success"]:
                perf["successful"] += 1
                perf["avg_results"] = (perf["avg_results"] + search["results_count"]) / perf["successful"]

        # Calculate success rates
        for engine, perf in analysis["engine_performance"].items():
            perf["success_rate"] = perf["successful"] / perf["total"] if perf["total"] > 0 else 0

        return analysis

    def _analyze_cross_engine_results(self) -> dict[str, Any]:
        """Analyze results across different search engines"""
        if not self.current_task_context or not self.current_task_context.cross_engine_results:
            return {"message": "No cross-engine results available"}

        analysis: dict[str, Any] = {
            "engines_used": list(self.current_task_context.cross_engine_results.keys()),
            "total_unique_results": 0,
            "common_results": 0,
            "engine_specific_results": {},
        }

        # Find unique results across all engines
        all_links = set()
        for engine, results in self.current_task_context.cross_engine_results.items():
            engine_links = {r.get("link", "") for r in results if isinstance(r, dict)}
            analysis["engine_specific_results"][engine] = len(engine_links)
            all_links.update(engine_links)

        analysis["total_unique_results"] = len(all_links)

        return analysis

    def get_comprehensive_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of system state and performance"""
        return {
            "memory_state": {
                "short_term_items": len(self.memory.short_term),
                "long_term_items": len(self.memory.long_term),
                "episodic_memories": len(self.memory.episodic),
                "search_patterns": len(self.memory.search_patterns),
            },
            "context_state": {
                "current_tokens": self.context_window.current_tokens,
                "max_tokens": self.context_window.max_tokens,
                "priority_items": len(self.context_window.priority_content),
            },
            "task_state": {
                "current_task": self.current_task_context.task_id if self.current_task_context else None,
                "progress": f"{self.current_task_context.current_step}/{self.current_task_context.total_steps}"
                if self.current_task_context
                else "N/A",
                "search_history_count": len(self.current_task_context.search_history)
                if self.current_task_context
                else 0,
            },
            "performance_analysis": self.analyze_search_performance() if self.current_task_context else {},
        }
