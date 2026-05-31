"""Tests for the BasicAgent fallback wrapper."""

from agentic_internet.agents.basic_agent import BasicAgent


class _Response:
    def __init__(self, content: str):
        self.content = content


class TestBasicAgentRun:
    def test_returns_content_attribute(self):
        agent = BasicAgent(model=lambda messages: _Response("answer text"))
        assert agent.run("question") == "answer text"

    def test_returns_plain_string_response(self):
        agent = BasicAgent(model=lambda messages: "plain answer")
        assert agent.run("question") == "plain answer"

    def test_includes_tool_descriptions_in_prompt(self):
        captured = {}

        def model(messages):
            captured["prompt"] = messages[0]["content"]
            return "ok"

        class FakeTool:
            name = "web_search"
            description = "Search the web"

        agent = BasicAgent(model=model, tools=[FakeTool()])
        agent.run("find cats")
        assert "web_search" in captured["prompt"]
        assert "find cats" in captured["prompt"]

    def test_falls_back_to_direct_call_on_type_error(self):
        calls = {"count": 0}

        def model(arg):
            calls["count"] += 1
            # First call (messages list) raises, second (string) succeeds.
            if isinstance(arg, list):
                raise TypeError("messages not supported")
            return "direct response"

        agent = BasicAgent(model=model)
        assert agent.run("q") == "direct response"
        assert calls["count"] == 2

    def test_exception_is_wrapped_in_message(self):
        def model(_):
            raise RuntimeError("model exploded")

        agent = BasicAgent(model=model)
        out = agent.run("q")
        assert out.startswith("Error executing task:")
        assert "model exploded" in out
