"""Tests for configuration settings."""

import os
from unittest.mock import patch

from agentic_internet.config.settings import (
    AgentConfig,
    ModelConfig,
    Settings,
    ToolConfig,
)


class TestModelConfig:
    def test_defaults(self):
        config = ModelConfig()
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.provider == "auto"

    def test_custom_values(self):
        config = ModelConfig(temperature=0.5, max_tokens=4096)
        assert config.temperature == 0.5
        assert config.max_tokens == 4096


class TestAgentConfig:
    def test_defaults(self):
        config = AgentConfig()
        assert config.verbose is True
        assert config.max_iterations == 10
        assert config.planning_enabled is True


class TestToolConfig:
    def test_defaults(self):
        config = ToolConfig()
        assert config.web_search_enabled is True
        assert config.max_search_results == 5


class TestSettings:
    def test_get_model_provider_openrouter(self):
        s = Settings()
        provider = s.get_model_provider("openrouter/anthropic/claude-sonnet-4.5")
        assert provider == "openrouter"

    def test_get_model_provider_openai_pattern(self):
        s = Settings()
        provider = s.get_model_provider("gpt-4o")
        assert provider == "openai"

    def test_get_model_provider_anthropic_pattern(self):
        s = Settings()
        provider = s.get_model_provider("claude-3-opus")
        assert provider == "anthropic"

    def test_get_api_key_for_provider(self):
        s = Settings(openai_api_key="test-key")
        assert s.get_api_key_for_provider("openai") == "test-key"
        assert s.get_api_key_for_provider("nonexistent") is None

    def test_get_default_model_for_provider(self):
        s = Settings()
        default = s.get_default_model_for_provider("openrouter")
        assert default is not None

    def test_list_available_models_no_keys(self):
        s = Settings(
            openrouter_api_key=None,
            openai_api_key=None,
            anthropic_api_key=None,
            huggingface_token=None,
        )
        available = s.list_available_models()
        assert available == {}

    def test_list_available_models_with_key(self):
        s = Settings(openai_api_key="test")
        available = s.list_available_models()
        assert "openai" in available

    def test_get_model_info(self):
        s = Settings()
        info = s.get_model_info("gpt-4o")
        assert info["model_id"] == "gpt-4o"
        assert "provider" in info

    def test_validate_startup_no_keys(self):
        s = Settings(
            openrouter_api_key=None,
            openai_api_key=None,
            anthropic_api_key=None,
            huggingface_token=None,
            browser_use_api_key=None,
        )
        warnings = s.validate_startup()
        assert len(warnings) > 0
        assert any("No API keys" in w for w in warnings)

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_startup_with_key(self):
        s = Settings(
            openai_api_key="test-key",
            browser_use_api_key="bkey",
            openrouter_api_key=None,
            anthropic_api_key=None,
            huggingface_token=None,
        )
        warnings = s.validate_startup()
        # Should warn about missing providers but not about no keys at all
        assert not any("No API keys" in w for w in warnings)
        assert any("OPENROUTER_API_KEY" in w for w in warnings)
