"""Tests for the Code Mode ToolFacade agent path."""

from typing import ClassVar
from unittest.mock import patch

import pytest
from smolagents import CodeAgent, Tool

from agentic_internet.agents.code_mode import ExecuteTool, SearchTool, ToolFacade, create_code_mode_agent


class EchoTool(Tool):
    name = "echo"
    description = "Echo text back to the caller."
    inputs: ClassVar[dict] = {"text": {"type": "string", "description": "Text to echo"}}
    output_type = "string"

    def forward(self, text: str) -> str:
        return f"echo:{text}"


class HyphenTool:
    name = "get-weather"
    description = "Get weather by city."
    inputs: ClassVar[dict] = {"city": {"type": "string", "description": "City name"}}
    output_type = "string"

    def forward(self, city: str) -> str:
        return f"weather:{city}"


class ReservedTool(Tool):
    name = "execute"
    description = "Reserved name."
    inputs: ClassVar[dict] = {}
    output_type = "string"

    def forward(self) -> str:
        return "reserved"


class DummyModel:
    pass


class TestToolFacade:
    def test_search_returns_identifier_and_escape_hatch_signatures(self):
        facade = ToolFacade([EchoTool(), HyphenTool()])

        echo = facade.search("echo")
        weather = facade.search("weather")

        assert "api.echo(text: string)" in echo
        assert 'api._tools["get-weather"](city: string)' in weather

    def test_search_is_case_insensitive_and_handles_no_matches(self):
        facade = ToolFacade([EchoTool()])
        assert "api.echo" in facade.search("ECHO")
        assert facade.search("missing") == "No matching tools found."

    def test_getattr_dispatches_to_tool_and_missing_tool_suggests_search(self):
        facade = ToolFacade([EchoTool()])
        assert facade.echo(text="hello") == "echo:hello"
        with pytest.raises(AttributeError, match="api.search"):
            _missing = facade.missing

    def test_reserved_tool_names_are_skipped(self):
        facade = ToolFacade([ReservedTool()])
        assert facade.search("reserved") == "No matching tools found."


class TestMetaTools:
    def test_search_tool_delegates_and_to_dict_restores_facade(self):
        facade = ToolFacade([EchoTool()])
        tool = SearchTool(facade=facade)

        assert "api.echo" in tool.forward("echo")
        assert isinstance(tool.to_dict(), dict)
        assert tool.facade is facade

    def test_execute_tool_runs_python_with_api_in_scope(self):
        facade = ToolFacade([EchoTool()])
        tool = ExecuteTool(facade=facade)

        assert tool.forward("api.echo(text='hello')") == "echo:hello"
        assert tool.facade is facade

    def test_execute_tool_returns_error_string_for_bad_code(self):
        tool = ExecuteTool(facade=ToolFacade([]))
        assert "Execution error" in tool.forward("missing_name")


class TestCreateCodeModeAgent:
    @patch("agentic_internet.agents.code_mode.initialize_model", return_value=DummyModel())
    def test_factory_returns_code_agent_and_attaches_facade(self, _mock_initialize_model):
        agent = create_code_mode_agent([EchoTool()], model_id="test-model", verbosity_level=0, max_steps=1)

        assert isinstance(agent, CodeAgent)
        assert isinstance(agent.facade, ToolFacade)
        assert "echo" in agent.facade._tools

    @patch("agentic_internet.agents.code_mode.initialize_model", return_value=DummyModel())
    def test_e2b_without_key_falls_back_to_local(self, _mock_initialize_model, monkeypatch):
        monkeypatch.delenv("E2B_API_KEY", raising=False)

        agent = create_code_mode_agent([EchoTool()], executor_type="e2b", verbosity_level=0, max_steps=1)

        assert isinstance(agent, CodeAgent)
