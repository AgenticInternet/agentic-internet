"""
Utility functions for model initialization and management.
This module uses the centralized settings configuration for all model-related operations.
"""

import logging
from typing import Any, Union

from smolagents import InferenceClientModel, LiteLLMModel

from ..config.settings import settings
from ..exceptions import ModelInitializationError

logger = logging.getLogger(__name__)

ModelInstance = Union[LiteLLMModel, InferenceClientModel]


def _create_model_for_provider(
    provider: str,
    model_id: str,
    api_key: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
) -> ModelInstance:
    """
    Create a model instance for the given provider. Centralised factory to avoid duplication.

    Raises:
        ModelInitializationError: If model creation fails.
    """
    temp = temperature if temperature is not None else settings.model.temperature
    tokens = max_tokens if max_tokens is not None else settings.model.max_tokens
    tp = top_p if top_p is not None else settings.model.top_p

    try:
        if provider == "openrouter":
            if not model_id.startswith("openrouter/"):
                model_id = f"openrouter/{model_id}"
            return LiteLLMModel(
                model_id=model_id, api_key=api_key,
                temperature=temp, max_tokens=tokens, top_p=tp,
            )
        elif provider in ("openai", "anthropic"):
            return LiteLLMModel(
                model_id=model_id, api_key=api_key,
                temperature=temp, max_tokens=tokens, top_p=tp,
            )
        elif provider == "huggingface":
            return InferenceClientModel(model_id=model_id, token=api_key)
        else:
            raise ModelInitializationError(model_id, provider, f"Unknown provider: {provider}")
    except ModelInitializationError:
        raise
    except Exception as e:
        raise ModelInitializationError(model_id, provider, str(e)) from e


def initialize_model(
    model_id: str | None = None,
    verbose: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
) -> ModelInstance | None:
    """
    Initialize a model based on model_id and settings configuration.

    Returns:
        Initialized model instance or None if initialization fails.
    """
    model_id = model_id or settings.model.name
    provider = settings.get_model_provider(model_id)

    if not provider:
        logger.warning("Could not determine provider for model: %s", model_id)
        return get_any_available_model(verbose)

    api_key = settings.get_api_key_for_provider(provider)
    if not api_key:
        logger.warning("No API key found for provider: %s", provider)
        return get_any_available_model(verbose)

    try:
        return _create_model_for_provider(
            provider, model_id, api_key,
            temperature=temperature, max_tokens=max_tokens, top_p=top_p,
        )
    except ModelInitializationError as e:
        logger.warning("Failed to initialize model: %s", e)
        return get_any_available_model(verbose)


def get_any_available_model(verbose: bool = False) -> ModelInstance | None:
    """
    Try to get any available model based on settings configuration.

    Returns:
        First successfully initialized model or None.
    """
    providers = ["openrouter", "openai", "anthropic", "huggingface"]

    for provider in providers:
        api_key = settings.get_api_key_for_provider(provider)
        if not api_key:
            continue

        model_id = settings.get_default_model_for_provider(provider)
        if not model_id:
            continue

        try:
            return _create_model_for_provider(provider, model_id, api_key)
        except ModelInitializationError as e:
            logger.debug("Failed to initialize default %s model: %s", provider, e)

    return None


def get_model_info(model: Any) -> dict[str, Any]:
    """Get information about a model instance."""
    if model is None:
        return {"type": "None", "model_id": None, "provider": None}

    model_type = type(model).__name__
    model_id = getattr(model, "model_id", None) or getattr(model, "model", None)
    provider = settings.get_model_provider(model_id) if model_id else None

    return {
        "type": model_type,
        "model_id": model_id,
        "provider": provider,
        "has_api_key": bool(getattr(model, "api_key", None) or getattr(model, "token", None)),
        "temperature": getattr(model, "temperature", settings.model.temperature),
        "max_tokens": getattr(model, "max_tokens", settings.model.max_tokens),
    }


def list_available_models() -> list[dict[str, str]]:
    """List all models that can be initialized with current API keys."""
    available_by_provider = settings.list_available_models()

    return [
        {"provider": provider.capitalize(), "model_id": mid, "ready": True}
        for provider, models in available_by_provider.items()
        for mid in models
    ]
