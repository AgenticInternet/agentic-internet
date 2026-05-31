"""Tests for the search orchestrator coordination logic.

These tests use lightweight fake agents (objects exposing a ``run`` method) so
that no external model or network access is required.
"""

from agentic_internet.agents.search_orchestrator import (
    SearchAgentWrapper,
    SearchOrchestrator,
    SearchResult,
    SearchTask,
)


class FakeAgent:
    """Minimal stand-in for a smolagents agent."""

    def __init__(self, response: str = "ok", raises: Exception | None = None):
        self.response = response
        self.raises = raises
        self.prompts: list[str] = []

    def run(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.raises is not None:
            raise self.raises
        return self.response


class TestSearchAgentWrapper:
    def test_execute_success_records_result(self):
        wrapper = SearchAgentWrapper("a1", FakeAgent("found it"), specialization="news")
        task = SearchTask(query="ai", task_id="t1", agent_name="a1")

        result = wrapper.execute(task)

        assert isinstance(result, SearchResult)
        assert result.success is True
        assert result.result == "found it"
        assert result.error is None
        assert wrapper.execution_count == 1
        assert wrapper.success_count == 1

    def test_execute_failure_captures_error(self):
        wrapper = SearchAgentWrapper("a1", FakeAgent(raises=RuntimeError("boom")))
        task = SearchTask(query="ai", task_id="t1", agent_name="a1")

        result = wrapper.execute(task)

        assert result.success is False
        assert result.result is None
        assert "boom" in result.error
        assert wrapper.execution_count == 1
        assert wrapper.success_count == 0

    def test_specialized_prompt_variants(self):
        agent = FakeAgent()
        for spec, marker in [
            ("news", "recent news"),
            ("academic", "academic papers"),
            ("technical", "technical documentation"),
            ("market", "market analysis"),
            ("comprehensive", "comprehensive search"),
        ]:
            wrapper = SearchAgentWrapper("x", agent, specialization=spec)
            prompt = wrapper._create_specialized_prompt(SearchTask(query="topic", task_id="t", agent_name="x"))
            assert marker in prompt
            assert "topic" in prompt

    def test_specialized_prompt_default_passthrough(self):
        wrapper = SearchAgentWrapper("x", FakeAgent(), specialization="general")
        prompt = wrapper._create_specialized_prompt(SearchTask(query="raw query", task_id="t", agent_name="x"))
        assert prompt == "raw query"

    def test_performance_stats(self):
        wrapper = SearchAgentWrapper("a1", FakeAgent(), specialization="news")
        wrapper.execute(SearchTask(query="q", task_id="t", agent_name="a1"))
        stats = wrapper.get_performance_stats()
        assert stats["name"] == "a1"
        assert stats["specialization"] == "news"
        assert stats["executions"] == 1
        assert stats["success_rate"] == 1.0

    def test_performance_stats_no_executions(self):
        wrapper = SearchAgentWrapper("a1", FakeAgent())
        stats = wrapper.get_performance_stats()
        assert stats["success_rate"] == 0


class TestSearchOrchestrator:
    def _orchestrator(self) -> SearchOrchestrator:
        return SearchOrchestrator(verbose=False)

    def test_add_agent(self):
        orch = self._orchestrator()
        orch.add_agent("a1", FakeAgent(), specialization="news")
        assert "a1" in orch.agents
        assert orch.agents["a1"].specialization == "news"

    def test_search_without_agents_returns_error(self):
        orch = self._orchestrator()
        out = orch.search("query")
        assert "error" in out
        assert out["query"] == "query"

    def test_search_with_unknown_agent_selection(self):
        orch = self._orchestrator()
        orch.add_agent("a1", FakeAgent())
        out = orch.search("query", agents_to_use=["nope"])
        assert "error" in out

    def test_search_sequential_aggregates_results(self):
        orch = self._orchestrator()
        orch.add_agent("a1", FakeAgent("r1"), specialization="news")
        out = orch.search("topic", parallel=False)

        assert out["query"] == "topic"
        assert out["total_agents"] == 1
        assert out["successful_agents"] == 1
        assert out["failed_agents"] == 0
        assert "a1" in out["agent_results"]
        assert len(orch.execution_history) == 1

    def test_search_parallel_with_mixed_outcomes(self):
        orch = self._orchestrator()
        orch.add_agent("ok", FakeAgent("good"))
        orch.add_agent("bad", FakeAgent(raises=ValueError("nope")))

        out = orch.search("topic", parallel=True)

        assert out["total_agents"] == 2
        assert out["successful_agents"] == 1
        assert out["failed_agents"] == 1
        assert "bad" in out["failures"]

    def test_synthesis_runs_when_orchestrator_present(self):
        orch = self._orchestrator()
        orch.add_agent("a1", FakeAgent("r1"))
        orch.orchestrator_agent = FakeAgent("synthesized summary")

        out = orch.search("topic", parallel=False)
        assert out["synthesis"] == "synthesized summary"

    def test_synthesis_handles_failure_gracefully(self):
        orch = self._orchestrator()
        results = [SearchResult(task_id="t", agent_name="a1", result="x", success=True, execution_time=0.1)]
        orch.add_agent("a1", FakeAgent("x"))
        orch.orchestrator_agent = FakeAgent(raises=RuntimeError("synth fail"))

        assert orch._synthesize_results("q", results) is None

    def test_get_performance_report(self):
        orch = self._orchestrator()
        orch.add_agent("a1", FakeAgent("x"))
        orch.search("topic", parallel=False)

        report = orch.get_performance_report()
        assert report["total_executions"] == 1
        assert "a1" in report["agents"]
        assert report["agents"]["a1"]["executions"] == 1
