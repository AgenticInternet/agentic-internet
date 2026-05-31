"""Tests for the context-engineering primitives in multi_model_serpapi.

These cover pure in-memory logic (no SerpAPI or model access required):
ContextWindow, AgentMemory, and TaskContext.
"""

from agentic_internet.agents.multi_model_serpapi import (
    AgentMemory,
    ContextWindow,
    TaskContext,
)


class TestContextWindow:
    def test_add_content_increments_tokens(self):
        cw = ContextWindow(max_tokens=1000)
        added = cw.add_content("one two three four")
        assert added is True
        assert cw.current_tokens > 0
        assert "one two three four" in cw.priority_content

    def test_high_priority_inserts_at_front(self):
        cw = ContextWindow(max_tokens=10_000)
        cw.add_content("low", priority=1)
        cw.add_content("high", priority=5)
        assert cw.priority_content[0] == "high"

    def test_low_priority_appends(self):
        cw = ContextWindow(max_tokens=10_000)
        cw.add_content("first", priority=1)
        cw.add_content("second", priority=1)
        assert cw.priority_content == ["first", "second"]

    def test_tokens_are_integer(self):
        cw = ContextWindow(max_tokens=10_000)
        cw.add_content("a b c d e")  # would be 5 * 1.3 = 6.5 before int()
        assert isinstance(cw.current_tokens, int)

    def test_compression_triggers_over_threshold(self):
        cw = ContextWindow(max_tokens=100, compression_threshold=0.5)
        # Push many items so the compression path runs.
        for i in range(8):
            cw.add_content(f"item number {i} with several words here")
        # Compression collapses the middle into a single summary marker.
        assert any("COMPRESSED" in item for item in cw.priority_content)


class TestAgentMemory:
    def test_remember_creates_pattern(self):
        mem = AgentMemory()
        mem.remember_search_pattern("web", "google", {"success": True, "result_count": 5})
        rec = mem.get_search_recommendations("web", "google")
        assert rec["total_attempts"] == 1
        assert rec["success_count"] == 1
        assert rec["avg_results"] > 0

    def test_remember_failure_does_not_increment_success(self):
        mem = AgentMemory()
        mem.remember_search_pattern("web", "google", {"success": False})
        rec = mem.get_search_recommendations("web", "google")
        assert rec["total_attempts"] == 1
        assert rec["success_count"] == 0

    def test_best_params_stored_on_success(self):
        mem = AgentMemory()
        mem.remember_search_pattern("web", "bing", {"success": True, "params": {"hl": "en"}})
        rec = mem.get_search_recommendations("web", "bing")
        assert rec["best_params"] == {"hl": "en"}

    def test_unknown_pattern_returns_empty(self):
        mem = AgentMemory()
        assert mem.get_search_recommendations("missing", "engine") == {}


class TestTaskContext:
    def test_log_search_appends_history(self):
        ctx = TaskContext(task_id="t1", objective="research")
        ctx.log_search("google", "ai news", 7, True)
        assert len(ctx.search_history) == 1
        entry = ctx.search_history[0]
        assert entry["engine"] == "google"
        assert entry["results_count"] == 7
        assert entry["success"] is True

    def test_progress_summary(self):
        ctx = TaskContext(task_id="t1", objective="research", current_step=2, total_steps=5)
        ctx.log_search("google", "q", 1, True)
        summary = ctx.get_progress_summary()
        assert "t1" in summary
        assert "2/5" in summary
        assert "Searches: 1" in summary
