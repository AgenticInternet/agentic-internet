"""Tests for safe configuration display."""

from typer.testing import CliRunner

from agentic_internet import cli
from agentic_internet.config.settings import ModelConfig, Settings


def test_config_show_redacts_every_credential_but_preserves_public_settings(monkeypatch, tmp_path):
    sentinels = {
        "huggingface_token": "synthetic-hf-token",
        "openai_api_key": "synthetic-openai-key",
        "anthropic_api_key": "synthetic-anthropic-key",
        "openrouter_api_key": "synthetic-openrouter-key",
        "browser_use_api_key": "synthetic-browser-key",
        "exa_api_key": "synthetic-exa-key",
    }
    sensitive_settings = Settings(
        cache_dir=tmp_path,
        model=ModelConfig(name="visible-model", api_key="synthetic-model-key"),
        **sentinels,
    )
    monkeypatch.setattr(cli, "settings", sensitive_settings)

    result = CliRunner().invoke(cli.app, ["config", "--show"])

    assert result.exit_code == 0, result.output
    for sentinel in (*sentinels.values(), "synthetic-model-key"):
        assert sentinel not in result.output
    assert "<redacted>" in result.output
    assert "visible-model" in result.output
