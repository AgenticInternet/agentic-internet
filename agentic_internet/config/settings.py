"""Configuration settings for the Agentic Internet application."""

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

class ModelConfig(BaseModel):
    """Configuration for LLM models."""
    name: str = Field(default="openrouter/anthropic/claude-opus-4.5", description="Default model name")
    provider: str = Field(default="auto", description="Model provider (auto, openrouter, openai, anthropic, huggingface)")
    api_key: str | None = Field(default=None, description="API key for the model provider")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=2048, description="Maximum tokens for generation")
    top_p: float = Field(default=0.95, description="Top-p sampling parameter")

    # Model provider configurations (updated Feb 2026)
    openrouter_models: dict[str, str] = Field(
        default={
            # Anthropic
            "claude-opus-4.5": "openrouter/anthropic/claude-opus-4.5",
            "claude-sonnet-4.5": "openrouter/anthropic/claude-sonnet-4.5",
            "claude-opus-4.1": "openrouter/anthropic/claude-opus-4.1",
            "claude-haiku-4.5": "openrouter/anthropic/claude-haiku-4.5",
            # OpenAI
            "gpt-5.2": "openrouter/openai/gpt-5.2",
            "gpt-5.2-chat": "openrouter/openai/gpt-5.2-chat",
            "gpt-5.2-codex": "openrouter/openai/gpt-5.2-codex",
            "gpt-5.2-pro": "openrouter/openai/gpt-5.2-pro",
            "gpt-5.1": "openrouter/openai/gpt-5.1",
            "gpt-5": "openrouter/openai/gpt-5",
            "gpt-oss-120b": "openrouter/openai/gpt-oss-120b",
            "o4-mini": "openrouter/openai/o4-mini",
            "o3": "openrouter/openai/o3",
            "o3-mini": "openrouter/openai/o3-mini",
            # Google
            "gemini-3-flash": "openrouter/google/gemini-3-flash-preview",
            "gemini-3-pro": "openrouter/google/gemini-3-pro-preview",
            "gemini-2.5-pro": "openrouter/google/gemini-2.5-pro",
            "gemini-2.5-flash": "openrouter/google/gemini-2.5-flash",
            # DeepSeek
            "deepseek-v3.2": "openrouter/deepseek/deepseek-v3.2",
            "deepseek-v3.2-speciale": "openrouter/deepseek/deepseek-v3.2-speciale",
            "deepseek-r1": "openrouter/deepseek/deepseek-r1-0528",
            # xAI
            "grok-4.1-fast": "openrouter/x-ai/grok-4.1-fast",
            "grok-4": "openrouter/x-ai/grok-4",
            "grok-4-fast": "openrouter/x-ai/grok-4-fast",
            "grok-code-fast": "openrouter/x-ai/grok-code-fast-1",
            # Mistral
            "devstral-2": "openrouter/mistralai/devstral-2512",
            "mistral-large": "openrouter/mistralai/mistral-large-2512",
            "mistral-medium-3.1": "openrouter/mistralai/mistral-medium-3.1",
            "mistral-small": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
            "magistral-medium": "openrouter/mistralai/magistral-medium-2506",
            # Qwen
            "qwen3-coder": "openrouter/qwen/qwen3-coder",
            "qwen3-coder-plus": "openrouter/qwen/qwen3-coder-plus",
            "qwen3-coder-flash": "openrouter/qwen/qwen3-coder-flash",
            "qwen3-coder-next": "openrouter/qwen/qwen3-coder-next",
            "qwen3-235b": "openrouter/qwen/qwen3-235b-a22b",
            "qwen3-max": "openrouter/qwen/qwen3-max",
            # Meta
            "llama-4-maverick": "openrouter/meta-llama/llama-4-maverick",
            "llama-4-scout": "openrouter/meta-llama/llama-4-scout",
            # MoonshotAI
            "kimi-k2.5": "openrouter/moonshotai/kimi-k2.5",
            "kimi-k2": "openrouter/moonshotai/kimi-k2",
            # Perplexity
            "sonar-pro": "openrouter/perplexity/sonar-pro",
            "sonar": "openrouter/perplexity/sonar",
            "sonar-reasoning-pro": "openrouter/perplexity/sonar-reasoning-pro",
            "sonar-deep-research": "openrouter/perplexity/sonar-deep-research",
            # MiniMax
            "minimax-m2.1": "openrouter/minimax/minimax-m2.1",
            # Xiaomi
            "mimo-v2-flash": "openrouter/xiaomi/mimo-v2-flash",
            # Alibaba
            "tongyi-deepsearch": "openrouter/alibaba/tongyi-deepresearch-30b-a3b",
        },
        description="Available OpenRouter models"
    )

    openai_models: dict[str, str] = Field(
        default={
            "gpt-5.2": "gpt-5.2",
            "gpt-5.2-chat": "gpt-5.2-chat",
            "gpt-5.1": "gpt-5.1",
            "gpt-5": "gpt-5",
            "gpt-4.1": "gpt-4.1",
            "gpt-4o": "gpt-4o",
            "o4-mini": "o4-mini",
            "o3": "o3",
            "o3-mini": "o3-mini",
        },
        description="Available OpenAI models"
    )

    anthropic_models: dict[str, str] = Field(
        default={
            "claude-opus-4.5": "claude-opus-4.5",
            "claude-sonnet-4.5": "claude-sonnet-4.5",
            "claude-opus-4.1": "claude-opus-4.1",
            "claude-haiku-4.5": "claude-haiku-4.5",
        },
        description="Available Anthropic models"
    )

    huggingface_models: dict[str, str] = Field(
        default={
            "llama-4-scout": "meta-llama/llama-4-scout",
            "llama-4-maverick": "meta-llama/llama-4-maverick",
            "qwen3-235b": "qwen/qwen3-235b-a22b",
        },
        description="Available HuggingFace models"
    )

    # Fallback model preferences by provider
    fallback_models: dict[str, str] = Field(
        default={
            "openrouter": "openrouter/deepseek/deepseek-v3.2",
            "openai": "gpt-5.2",
            "anthropic": "claude-sonnet-4.5",
            "huggingface": "meta-llama/llama-4-scout",
        },
        description="Fallback models for each provider"
    )

    class Config:
        protected_namespaces = ()

class AgentConfig(BaseModel):
    """Configuration for agent behavior."""
    verbose: bool = Field(default=True, description="Enable verbose output")
    max_iterations: int = Field(default=10, description="Maximum iterations for agent")
    memory_enabled: bool = Field(default=True, description="Enable agent memory")
    tool_choice: str = Field(default="auto", description="Tool selection strategy")
    planning_enabled: bool = Field(default=True, description="Enable multi-step planning")

    class Config:
        protected_namespaces = ()

class ToolConfig(BaseModel):
    """Configuration for tools."""
    web_search_enabled: bool = Field(default=True, description="Enable web search tool")
    code_execution_enabled: bool = Field(default=True, description="Enable code execution")
    browser_enabled: bool = Field(default=True, description="Enable browser automation")
    file_operations_enabled: bool = Field(default=True, description="Enable file operations")
    max_search_results: int = Field(default=5, description="Maximum search results")

    class Config:
        protected_namespaces = ()

class Settings(BaseModel):
    """Main settings for the application."""
    # API Keys from environment
    huggingface_token: str | None = Field(
        default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN"),
        description="HuggingFace API token"
    )
    openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key"
    )
    anthropic_api_key: str | None = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"),
        description="Anthropic API key"
    )
    openrouter_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY"),
        description="OpenRouter API key"
    )
    browser_use_api_key: str | None = Field(
        default_factory=lambda: os.getenv("BROWSER_USE_API_KEY"),
        description="Browser Use Cloud API key"
    )

    # Model configuration
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Agent configuration
    agent: AgentConfig = Field(default_factory=AgentConfig)

    # Tool configuration
    tools: ToolConfig = Field(default_factory=ToolConfig)

    # Application settings
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "agentic_internet",
        description="Cache directory for models and data"
    )
    log_level: str = Field(default="INFO", description="Logging level")

    class Config:
        protected_namespaces = ()

    def model_post_init(self, __context):
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model_provider(self, model_id: str | None = None) -> str | None:
        """Determine the provider for a given model ID."""
        model_id = model_id or self.model.name

        # First check if the model exists in our configured model lists
        # This gives us explicit control over provider mapping
        for _key, full_id in self.model.openrouter_models.items():
            if model_id == full_id or model_id.endswith(full_id):
                return "openrouter"

        for _key, full_id in self.model.openai_models.items():
            if model_id == full_id:
                return "openai"

        for _key, full_id in self.model.anthropic_models.items():
            if model_id == full_id:
                return "anthropic"

        for _key, full_id in self.model.huggingface_models.items():
            if model_id == full_id:
                return "huggingface"

        # Fallback to pattern matching for models not in our lists
        # Check if it's explicitly an OpenRouter model
        if "openrouter/" in model_id.lower():
            return "openrouter"
        # Check if it matches OpenAI patterns
        elif any(name in model_id.lower() for name in ["gpt-3.5", "gpt-4", "gpt-5", "gpt", "o3", "o4", "o1"]):
            return "openai"
        # Check if it matches Anthropic patterns
        elif "claude" in model_id.lower():
            return "anthropic"
        # Check for common model providers that typically use OpenRouter
        elif any(provider in model_id.lower() for provider in [
            "deepseek/", "perplexity/", "x-ai/", "alibaba/", "mistral/",
            "qwen/", "moonshotai/", "minimax/", "xiaomi/", "google/",
        ]):
            if self.openrouter_api_key:
                return "openrouter"
        # Check if it matches HuggingFace patterns
        elif "/" in model_id and any(org in model_id.lower() for org in ["microsoft/", "meta-llama/", "huggingface/"]):
            return "huggingface"

        # Auto-detect based on available API keys
        if self.model.provider == "auto":
            # For unknown models with slashes, prefer OpenRouter if available
            if "/" in model_id and self.openrouter_api_key:
                return "openrouter"
            elif self.openai_api_key:
                return "openai"
            elif self.anthropic_api_key:
                return "anthropic"
            elif self.huggingface_token:
                return "huggingface"

        return self.model.provider if self.model.provider != "auto" else None

    def get_api_key_for_provider(self, provider: str) -> str | None:
        """Get the API key for a specific provider."""
        provider_keys = {
            "openrouter": self.openrouter_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "huggingface": self.huggingface_token
        }
        return provider_keys.get(provider)

    def get_default_model_for_provider(self, provider: str) -> str | None:
        """Get the default/fallback model for a provider."""
        return self.model.fallback_models.get(provider)

    def list_available_models(self) -> dict[str, list]:
        """List all available models based on configured API keys."""
        available = {}

        if self.openrouter_api_key:
            available["openrouter"] = list(self.model.openrouter_models.values())

        if self.openai_api_key:
            available["openai"] = list(self.model.openai_models.values())

        if self.anthropic_api_key:
            available["anthropic"] = list(self.model.anthropic_models.values())

        if self.huggingface_token:
            available["huggingface"] = list(self.model.huggingface_models.values())

        return available

    def get_model_info(self, model_id: str | None = None) -> dict[str, Any]:
        """Get information about a model."""
        model_id = model_id or self.model.name
        provider = self.get_model_provider(model_id)

        return {
            "model_id": model_id,
            "provider": provider,
            "has_api_key": bool(self.get_api_key_for_provider(provider)) if provider else False,
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
            "top_p": self.model.top_p,
        }

    def validate_startup(self) -> list[str]:
        """Validate configuration at startup and return warnings for missing keys."""
        warnings: list[str] = []

        provider_key_map = {
            "openrouter": ("OPENROUTER_API_KEY", self.openrouter_api_key),
            "openai": ("OPENAI_API_KEY", self.openai_api_key),
            "anthropic": ("ANTHROPIC_API_KEY", self.anthropic_api_key),
            "huggingface": ("HUGGINGFACE_TOKEN", self.huggingface_token),
        }

        has_any_key = False
        for provider, (env_var, value) in provider_key_map.items():
            if value:
                has_any_key = True
            else:
                warnings.append(f"{env_var} not set -- {provider} models will be unavailable")

        if not has_any_key:
            warnings.insert(0, "No API keys configured. At least one provider key is required.")

        if not self.browser_use_api_key:
            warnings.append("BROWSER_USE_API_KEY not set -- browser automation will be unavailable")

        for msg in warnings:
            logger.warning(msg)

        return warnings


# Global settings instance
settings = Settings()
