"""
Utility functions for model initialization and management.
This module uses the centralized settings configuration for all model-related operations.
"""

from typing import Optional, Any, Dict, List
from smolagents import LiteLLMModel, InferenceClientModel
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


def initialize_model(model_id: Optional[str] = None, verbose: bool = False) -> Optional[Any]:
    """
    Initialize a model based on model_id and settings configuration.
    
    This function uses the centralized settings to determine the appropriate
    provider and API key for model initialization.
    
    Args:
        model_id: The model identifier (e.g., "gpt-3.5-turbo", "openrouter/anthropic/claude-3-haiku")
        verbose: Whether to print verbose output
        
    Returns:
        Initialized model instance or None if initialization fails
    """
    model_id = model_id or settings.model.name
    
    # Determine the provider from settings
    provider = settings.get_model_provider(model_id)
    
    if not provider:
        if verbose:
            logger.warning(f"Could not determine provider for model: {model_id}")
        return get_any_available_model(verbose)
    
    # Get the API key for the provider
    api_key = settings.get_api_key_for_provider(provider)
    
    if not api_key:
        if verbose:
            logger.warning(f"No API key found for provider: {provider}")
        return get_any_available_model(verbose)
    
    # Initialize based on provider
    try:
        if provider == "openrouter":
            # Ensure OpenRouter models have the correct prefix for LiteLLM
            if not model_id.startswith("openrouter/"):
                model_id = f"openrouter/{model_id}"
            return LiteLLMModel(
                model_id=model_id,
                api_key=api_key,
                temperature=settings.model.temperature,
                max_tokens=settings.model.max_tokens,
                top_p=settings.model.top_p
            )
        elif provider in ["openai", "anthropic"]:
            return LiteLLMModel(
                model_id=model_id,
                api_key=api_key,
                temperature=settings.model.temperature,
                max_tokens=settings.model.max_tokens,
                top_p=settings.model.top_p
            )
        elif provider == "huggingface":
            return InferenceClientModel(
                model_id=model_id,
                token=api_key
            )
        else:
            if verbose:
                logger.warning(f"Unknown provider: {provider}")
            return None
    except Exception as e:
        if verbose:
            logger.warning(f"Failed to initialize {provider} model {model_id}: {e}")
        return get_any_available_model(verbose)


def get_any_available_model(verbose: bool = False) -> Optional[Any]:
    """
    Try to get any available model based on settings configuration.
    
    Uses the fallback models defined in settings for each provider.
    Priority order is determined by API key availability.
    
    Args:
        verbose: Whether to print verbose output
        
    Returns:
        First successfully initialized model or None
    """
    # Try each provider in order of preference
    providers = ["openrouter", "openai", "anthropic", "huggingface"]
    
    for provider in providers:
        api_key = settings.get_api_key_for_provider(provider)
        if not api_key:
            continue
            
        model_id = settings.get_default_model_for_provider(provider)
        if not model_id:
            continue
        
        try:
            if provider == "openrouter":
                # Ensure OpenRouter models have the correct prefix for LiteLLM
                if not model_id.startswith("openrouter/"):
                    model_id = f"openrouter/{model_id}"
                return LiteLLMModel(
                    model_id=model_id,
                    api_key=api_key,
                    temperature=settings.model.temperature,
                    max_tokens=settings.model.max_tokens,
                    top_p=settings.model.top_p
                )
            elif provider in ["openai", "anthropic"]:
                return LiteLLMModel(
                    model_id=model_id,
                    api_key=api_key,
                    temperature=settings.model.temperature,
                    max_tokens=settings.model.max_tokens,
                    top_p=settings.model.top_p
                )
            elif provider == "huggingface":
                return InferenceClientModel(
                    model_id=model_id,
                    token=api_key
                )
        except Exception as e:
            if verbose:
                logger.debug(f"Failed to initialize default {provider} model: {e}")
    
    return None


def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Get information about a model instance.
    
    Args:
        model: The model instance
        
    Returns:
        Dictionary with model information
    """
    if model is None:
        return {"type": "None", "model_id": None, "provider": None}
    
    model_type = type(model).__name__
    model_id = getattr(model, 'model_id', None) or getattr(model, 'model', None)
    
    # Use settings to get additional info
    provider = settings.get_model_provider(model_id) if model_id else None
    
    return {
        "type": model_type,
        "model_id": model_id,
        "provider": provider,
        "has_api_key": bool(getattr(model, 'api_key', None) or getattr(model, 'token', None)),
        "temperature": getattr(model, 'temperature', settings.model.temperature),
        "max_tokens": getattr(model, 'max_tokens', settings.model.max_tokens)
    }


def list_available_models() -> List[Dict[str, str]]:
    """
    List all models that can be initialized with current API keys.
    
    This function delegates to the settings configuration for the model list.
    
    Returns:
        List of available model configurations
    """
    available_by_provider = settings.list_available_models()
    
    # Flatten the structure for easier consumption
    available = []
    for provider, models in available_by_provider.items():
        for model_id in models:
            available.append({
                "provider": provider.capitalize(),
                "model_id": model_id,
                "ready": True
            })
    
    return available
