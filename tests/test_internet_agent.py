"""Tests for InternetAgent tool and execution controls."""

from unittest.mock import patch

import pytest

from agentic_internet.agents.internet_agent import InternetAgent


def test_explicit_empty_tool_list_does_not_enable_defaults():
    with (
        patch("agentic_internet.agents.internet_agent.initialize_model", return_value=object()),
        patch("agentic_internet.agents.internet_agent.ToolCallingAgent") as agent_cls,
        patch.object(InternetAgent, "_get_default_tools") as default_tools,
    ):
        agent = InternetAgent(tools=[], verbose=False)

    assert agent.tools == []
    default_tools.assert_not_called()
    assert agent_cls.call_args.kwargs["tools"] == []


@pytest.mark.parametrize(("planning_enabled", "expected_interval"), [(True, 1), (False, None)])
def test_tool_calling_agent_receives_iteration_and_planning_controls(planning_enabled, expected_interval):
    with (
        patch("agentic_internet.agents.internet_agent.initialize_model", return_value=object()),
        patch("agentic_internet.agents.internet_agent.ToolCallingAgent") as agent_cls,
    ):
        InternetAgent(
            tools=[],
            verbose=False,
            max_iterations=7,
            planning_enabled=planning_enabled,
        )

    kwargs = agent_cls.call_args.kwargs
    assert kwargs["max_steps"] == 7
    assert kwargs["planning_interval"] == expected_interval


@pytest.mark.parametrize(("planning_enabled", "expected_interval"), [(True, 1), (False, None)])
def test_code_agent_receives_iteration_and_planning_controls(planning_enabled, expected_interval):
    with (
        patch("agentic_internet.agents.internet_agent.initialize_model", return_value=object()),
        patch("agentic_internet.agents.internet_agent.CodeAgent") as agent_cls,
    ):
        InternetAgent(
            tools=[],
            verbose=False,
            agent_type="code",
            max_iterations=7,
            planning_enabled=planning_enabled,
        )

    kwargs = agent_cls.call_args.kwargs
    assert kwargs["max_steps"] == 7
    assert kwargs["planning_interval"] == expected_interval
