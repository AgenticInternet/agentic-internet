"""Parity tests for static and central OpenRouter model catalogs."""

from agentic_internet.config.settings import ModelConfig, settings


def test_static_cli_catalog_uses_central_aliases():
    from agentic_internet.cli import STATIC_MODEL_CATALOG

    configured = settings.model.openrouter_models
    for category, entries in STATIC_MODEL_CATALOG.items():
        assert entries, f"{category} should advertise at least one model"
        for entry in entries:
            assert entry["name"] in configured, f"{entry['name']} in {category} is not centrally configured"


def test_curated_recent_aliases_are_configured():
    recent = {
        "gpt-6.1-terra",
        "gpt-6-luna",
        "claude-sonnet-5.5",
        "claude-haiku-5.5",
        "gemini-4-argon",
        "kimi-k3.1-pro",
        "qwen3.8-max-prime",
        "deepseek-v4.1-flash",
        "glm-5.3-prime",
        "mistral-large-4",
        "minimax-m3.1-flash",
        "step-5-preview",
        "command-a-plus",
        "solar-mini4",
        "perceptron-mk1.5",
        "lfm-3-1.7b",
        "lfm-3-1.7b-free",
        "nemotron-3.5-super-vl-free",
        "hy4",
        "particle-1",
        "openrouter-auto",
        "openrouter-auto-code",
    }
    assert recent <= settings.model.openrouter_models.keys()


def test_default_and_fallback_remain_in_central_catalog():
    config = ModelConfig()
    configured_ids = set(config.openrouter_models.values())
    assert config.name in configured_ids
    assert config.fallback_models["openrouter"] in configured_ids
    assert all(model_id.startswith("openrouter/") for model_id in configured_ids)
