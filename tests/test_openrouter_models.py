"""Tests for OpenRouter model inventory helpers."""

from agentic_internet.utils.openrouter_models import (
    normalize_openrouter_model_id,
    recent_agentic_models,
    sort_models_by_created,
    summarize_openrouter_model,
    supports_parameter,
    to_litellm_openrouter_model_id,
)


def test_openrouter_id_normalization_round_trip():
    assert normalize_openrouter_model_id("openrouter/anthropic/claude-opus-4.8") == "anthropic/claude-opus-4.8"
    assert normalize_openrouter_model_id("anthropic/claude-opus-4.8") == "anthropic/claude-opus-4.8"
    assert to_litellm_openrouter_model_id("anthropic/claude-opus-4.8") == "openrouter/anthropic/claude-opus-4.8"
    assert (
        to_litellm_openrouter_model_id("openrouter/anthropic/claude-opus-4.8") == "openrouter/anthropic/claude-opus-4.8"
    )


def test_recent_agentic_models_filters_and_sorts():
    models = [
        {"id": "old/plain", "created": 1, "supported_parameters": []},
        {"id": "new/tools", "created": 3, "supported_parameters": ["tools"]},
        {"id": "middle/reasoning", "created": 2, "supported_parameters": ["include_reasoning"]},
    ]

    result = recent_agentic_models(models)

    assert [model["id"] for model in result] == ["new/tools", "middle/reasoning"]


def test_summarize_openrouter_model_extracts_agent_fields():
    model = {
        "id": "google/gemini-3.5-flash",
        "name": "Google: Gemini 3.5 Flash",
        "created": 1779193800,
        "context_length": 1048576,
        "architecture": {"modality": "text+image+file+audio+video->text"},
        "pricing": {"prompt": "0.0000015", "completion": "0.000009"},
        "supported_parameters": ["tools", "structured_outputs", "include_reasoning"],
    }

    summary = summarize_openrouter_model(model)

    assert summary["id"] == "google/gemini-3.5-flash"
    assert summary["litellm_id"] == "openrouter/google/gemini-3.5-flash"
    assert summary["tools"] is True
    assert summary["reasoning"] is True
    assert summary["structured_outputs"] is True
    assert summary["created"] == "2026-05-19"


def test_supports_parameter_and_sorting_helpers():
    assert supports_parameter({"supported_parameters": ["tools"]}, "tools") is True
    assert supports_parameter({"supported_parameters": []}, "tools") is False
    assert [m["id"] for m in sort_models_by_created([{"id": "a", "created": 1}, {"id": "b", "created": 2}])] == [
        "b",
        "a",
    ]
