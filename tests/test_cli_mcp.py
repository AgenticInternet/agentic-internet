"""CLI smoke tests for MCP command routing."""

from contextlib import contextmanager
from types import SimpleNamespace

from typer.testing import CliRunner

from agentic_internet import cli

runner = CliRunner()


class FakeCodeAgent:
    """Minimal agent stub used to avoid live model calls in CLI smoke tests."""

    def __init__(self) -> None:
        self.tasks: list[str] = []

    def run(self, task: str) -> str:
        self.tasks.append(task)
        return f"handled:{task}"


def test_mcp_run_code_mode_passes_structured_output_to_mcp(monkeypatch):
    """`mcp run --agent-type code --structured-output` wires MCP tools into Code Mode."""
    mcp_calls = []
    factory_calls = []
    fake_tool = SimpleNamespace(name="lookup", description="Lookup data")
    fake_agent = FakeCodeAgent()

    @contextmanager
    def fake_mcp_tools(**kwargs):
        mcp_calls.append(kwargs)
        yield [fake_tool]

    def fake_create_code_mode_agent(**kwargs):
        factory_calls.append(kwargs)
        return fake_agent

    monkeypatch.setattr("agentic_internet.tools.mcp_integration.is_mcp_available", lambda: True)
    monkeypatch.setattr("agentic_internet.tools.mcp_integration.mcp_tools", fake_mcp_tools)
    monkeypatch.setattr("agentic_internet.agents.code_mode.create_code_mode_agent", fake_create_code_mode_agent)

    result = runner.invoke(
        cli.app,
        [
            "mcp",
            "run",
            "summarize remote data",
            "--server",
            "./fake_server.py",
            "--trust",
            "--agent-type",
            "code",
            "--structured-output",
            "--model",
            "openrouter/test-model",
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "handled:summarize remote data" in result.output
    assert mcp_calls == [
        {
            "server_path": "./fake_server.py",
            "server_url": None,
            "trust_remote_code": True,
            "structured_output": True,
        }
    ]
    assert factory_calls == [
        {
            "tools": [fake_tool],
            "model_id": "openrouter/test-model",
            "verbosity_level": 0,
        }
    ]
    assert fake_agent.tasks == ["summarize remote data"]
