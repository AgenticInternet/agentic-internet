"""CLI tests for K-LLM use-case routing."""

from typing import ClassVar

from typer.testing import CliRunner

from agentic_internet import cli

runner = CliRunner()


class FakeMultiModelSystem:
    instances: ClassVar[list["FakeMultiModelSystem"]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.setup_calls = []
        self.execute_calls = []
        FakeMultiModelSystem.instances.append(self)

    def setup_use_case_workers(self, **kwargs):
        self.setup_calls.append(kwargs)

    async def execute_multi_model_workflow(self, *args, **kwargs):
        self.execute_calls.append({"args": args, "kwargs": kwargs})
        return "fake-result"


def test_multi_command_routes_use_case_and_worker_overrides(monkeypatch):
    FakeMultiModelSystem.instances = []
    monkeypatch.setattr("agentic_internet.cli.MultiModelSerpAPISystem", FakeMultiModelSystem)

    result = runner.invoke(
        cli.app,
        [
            "multi",
            "assess repo",
            "--use-case",
            "technical_due_diligence",
            "--models",
            "qwen-coder",
            "--worker-model",
            "code_analyst=qwen-coder",
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Use case:" in result.output
    system = FakeMultiModelSystem.instances[0]
    assert system.setup_calls == [
        {
            "use_case_id": "technical_due_diligence",
            "default_model": "qwen-coder",
            "worker_model_overrides": {"code_analyst": "qwen-coder"},
        }
    ]
    assert system.execute_calls[0]["kwargs"]["use_case_id"] == "technical_due_diligence"
    assert system.execute_calls[0]["kwargs"]["orchestrator_model"] == "qwen-coder"


def test_tools_command_lists_use_cases():
    result = runner.invoke(cli.app, ["tools", "--use-cases"])
    assert result.exit_code == 0, result.output
    assert "technical_due_diligence" in result.output
    assert "market_intelligence" in result.output
