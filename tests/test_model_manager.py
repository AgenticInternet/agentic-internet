"""Tests for exact model selection in the multi-model manager."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentic_internet.agents.multi_model_serpapi import ModelManager
from agentic_internet.config.settings import settings


@pytest.mark.parametrize("alias", ["claude-opus-4.8", "o3", "gpt-6.1-terra", "lfm-3-1.7b-free"])
def test_model_manager_resolves_central_alias_exactly(alias):
    expected_model_id = settings.model.openrouter_models[alias]

    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "synthetic-key"}),
        patch("agentic_internet.agents.multi_model.model_manager.LiteLLMModel") as model_cls,
    ):
        model_cls.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        manager = ModelManager()
        model = manager.get_model_for_role(alias)

    assert model.model_id == expected_model_id
    assert model.api_key == "synthetic-key"
