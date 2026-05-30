"""Utilities for working with OpenRouter's model inventory."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any
from urllib.request import urlopen

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_LITELLM_PREFIX = "openrouter/"


def normalize_openrouter_model_id(model_id: str) -> str:
    """Return the OpenRouter API model ID without the LiteLLM ``openrouter/`` prefix."""
    if model_id.startswith(OPENROUTER_LITELLM_PREFIX):
        return model_id.removeprefix(OPENROUTER_LITELLM_PREFIX)
    return model_id


def to_litellm_openrouter_model_id(model_id: str) -> str:
    """Return a LiteLLM-compatible OpenRouter model ID."""
    if model_id.startswith(OPENROUTER_LITELLM_PREFIX):
        return model_id
    return f"{OPENROUTER_LITELLM_PREFIX}{model_id}"


def fetch_openrouter_models(timeout: float = 10.0) -> list[dict[str, Any]]:
    """Fetch OpenRouter's current model inventory.

    The endpoint is public. Callers should treat failures as non-fatal and fall
    back to the checked-in/static model catalog.
    """
    try:
        with urlopen(OPENROUTER_MODELS_URL, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Could not fetch OpenRouter models: %s", exc)
        return []

    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def created_date(model: dict[str, Any]) -> str:
    """Format an OpenRouter ``created`` timestamp as an ISO date."""
    created = model.get("created")
    if not isinstance(created, int):
        return "unknown"
    return datetime.fromtimestamp(created, UTC).date().isoformat()


def supports_parameter(model: dict[str, Any], parameter: str) -> bool:
    """Return whether a model advertises support for an OpenRouter parameter."""
    return parameter in (model.get("supported_parameters") or [])


def sort_models_by_created(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return models sorted newest first."""
    return sorted(models, key=lambda model: model.get("created") or 0, reverse=True)


def recent_agentic_models(models: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Return recent models that are especially useful for agent workflows."""
    agentic = [
        model
        for model in sort_models_by_created(models)
        if supports_parameter(model, "tools")
        or supports_parameter(model, "structured_outputs")
        or supports_parameter(model, "reasoning")
        or supports_parameter(model, "include_reasoning")
    ]
    return agentic[:limit]


def summarize_openrouter_model(model: dict[str, Any]) -> dict[str, Any]:
    """Extract the stable fields the CLI and docs care about."""
    architecture = model.get("architecture") or {}
    top_provider = model.get("top_provider") or {}
    pricing = model.get("pricing") or {}
    return {
        "created": created_date(model),
        "id": model.get("id", ""),
        "litellm_id": to_litellm_openrouter_model_id(str(model.get("id", ""))),
        "name": model.get("name", ""),
        "context_length": model.get("context_length") or top_provider.get("context_length"),
        "modality": architecture.get("modality", ""),
        "tools": supports_parameter(model, "tools"),
        "reasoning": supports_parameter(model, "reasoning") or supports_parameter(model, "include_reasoning"),
        "structured_outputs": supports_parameter(model, "structured_outputs"),
        "prompt_price": pricing.get("prompt"),
        "completion_price": pricing.get("completion"),
    }
