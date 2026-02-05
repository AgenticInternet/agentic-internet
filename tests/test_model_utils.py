"""Tests for model utility functions."""

from unittest.mock import MagicMock, patch

import pytest

from agentic_internet.exceptions import ModelInitializationError
from agentic_internet.utils.model_utils import (
    _create_model_for_provider,
    get_model_info,
    initialize_model,
    list_available_models,
)


class TestCreateModelForProvider:
    @patch("agentic_internet.utils.model_utils.LiteLLMModel")
    def test_openrouter_adds_prefix(self, mock_litellm):
        mock_litellm.return_value = MagicMock()
        _create_model_for_provider("openrouter", "anthropic/claude-3", "key123")
        mock_litellm.assert_called_once()
        call_kwargs = mock_litellm.call_args
        assert call_kwargs.kwargs["model_id"].startswith("openrouter/")

    @patch("agentic_internet.utils.model_utils.LiteLLMModel")
    def test_openrouter_no_double_prefix(self, mock_litellm):
        mock_litellm.return_value = MagicMock()
        _create_model_for_provider("openrouter", "openrouter/anthropic/claude-3", "key123")
        call_kwargs = mock_litellm.call_args
        assert not call_kwargs.kwargs["model_id"].startswith("openrouter/openrouter/")

    @patch("agentic_internet.utils.model_utils.LiteLLMModel")
    def test_openai_provider(self, mock_litellm):
        mock_litellm.return_value = MagicMock()
        _create_model_for_provider("openai", "gpt-4", "key123")
        mock_litellm.assert_called_once()

    @patch("agentic_internet.utils.model_utils.InferenceClientModel")
    def test_huggingface_provider(self, mock_ic):
        mock_ic.return_value = MagicMock()
        _create_model_for_provider("huggingface", "meta-llama/Llama-3", "token123")
        mock_ic.assert_called_once_with(model_id="meta-llama/Llama-3", token="token123")

    def test_unknown_provider_raises(self):
        with pytest.raises(ModelInitializationError, match="Unknown provider"):
            _create_model_for_provider("unknown_provider", "model-x", "key")

    @patch("agentic_internet.utils.model_utils.LiteLLMModel")
    def test_wraps_exception(self, mock_litellm):
        mock_litellm.side_effect = RuntimeError("connection failed")
        with pytest.raises(ModelInitializationError, match="connection failed"):
            _create_model_for_provider("openai", "gpt-4", "key")


class TestInitializeModel:
    @patch("agentic_internet.utils.model_utils.settings")
    @patch("agentic_internet.utils.model_utils._create_model_for_provider")
    def test_uses_settings_defaults(self, mock_create, mock_settings):
        mock_settings.model.name = "test-model"
        mock_settings.get_model_provider.return_value = "openai"
        mock_settings.get_api_key_for_provider.return_value = "key"
        mock_create.return_value = MagicMock()

        result = initialize_model()
        assert result is not None

    @patch("agentic_internet.utils.model_utils.settings")
    @patch("agentic_internet.utils.model_utils.get_any_available_model")
    def test_fallback_on_no_provider(self, mock_fallback, mock_settings):
        mock_settings.model.name = "unknown-model"
        mock_settings.get_model_provider.return_value = None
        mock_fallback.return_value = MagicMock()

        initialize_model()
        mock_fallback.assert_called_once()

    @patch("agentic_internet.utils.model_utils.settings")
    @patch("agentic_internet.utils.model_utils.get_any_available_model")
    def test_fallback_on_no_api_key(self, mock_fallback, mock_settings):
        mock_settings.model.name = "gpt-4"
        mock_settings.get_model_provider.return_value = "openai"
        mock_settings.get_api_key_for_provider.return_value = None
        mock_fallback.return_value = MagicMock()

        initialize_model()
        mock_fallback.assert_called_once()


class TestGetModelInfo:
    def test_none_model(self):
        info = get_model_info(None)
        assert info["type"] == "None"
        assert info["model_id"] is None

    def test_model_with_id(self):
        mock_model = MagicMock()
        mock_model.model_id = "gpt-4"
        mock_model.api_key = "key"
        mock_model.temperature = 0.5
        mock_model.max_tokens = 1000

        with patch("agentic_internet.utils.model_utils.settings") as mock_settings:
            mock_settings.get_model_provider.return_value = "openai"
            info = get_model_info(mock_model)

        assert info["model_id"] == "gpt-4"
        assert info["provider"] == "openai"
        assert info["has_api_key"] is True


class TestListAvailableModels:
    @patch("agentic_internet.utils.model_utils.settings")
    def test_flattens_models(self, mock_settings):
        mock_settings.list_available_models.return_value = {
            "openai": ["gpt-4", "gpt-3.5"],
            "anthropic": ["claude-3"],
        }
        result = list_available_models()
        assert len(result) == 3
        assert all(m["ready"] is True for m in result)
        providers = {m["provider"] for m in result}
        assert "Openai" in providers
        assert "Anthropic" in providers
