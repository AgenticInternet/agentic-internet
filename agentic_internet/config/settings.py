"""Configuration settings for the Agentic Internet application."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelConfig(BaseModel):
    """Configuration for LLM models."""
    name: str = Field(default="openrouter/anthropic/claude-sonnet-4", description="Default model name")
    provider: str = Field(default="auto", description="Model provider (auto, openrouter, openai, anthropic, huggingface)")
    api_key: Optional[str] = Field(default=None, description="API key for the model provider")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=2048, description="Maximum tokens for generation")
    top_p: float = Field(default=0.95, description="Top-p sampling parameter")
    
    # Model provider configurations
    openrouter_models: Dict[str, str] = Field(
        default={
            "claude-sonnet-4": "openrouter/anthropic/claude-sonnet-4",
            "deepseek-r1": "openrouter/deepseek/deepseek-r1-0528",
            "deepseek-chat-v3": "openrouter/deepseek/deepseek-chat-v3.1",
            "mistral-small": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
            "sonar": "openrouter/perplexity/sonar",
            "sonar-reasoning": "openrouter/perplexity/sonar-reasoning",
            "o3-mini": "openrouter/openai/o3-mini",
            "mistral-large": "openrouter/mistralai/mistral-large-2411",
            "gemini-2.5-flash": "openrouter/google/gemini-2.5-flash",
            "gpt-oss": "openrouter/openai/gpt-oss-120b",
            "gemini-2.5-pro": "openrouter/google/gemini-2.5-pro",
            "gpt-5-chat": "openrouter/openai/gpt-5-chat",
            "qwen3-coder": "openrouter/qwen/qwen3-coder",
            "qwen3-235b": "openrouter/qwen/qwen3-235b-a22b",
            "llama-4-scout": "openrouter/meta-llama/llama-4-scout:nitro",
            "grok-3-mini": "openrouter/x-ai/grok-3-mini-beta",
            "gpt-5": "openrouter/openai/gpt-5",
            "kimi-k2": "openrouter/moonshotai/kimi-k2"
        },
        description="Available OpenRouter models"
    )
    
    openai_models: Dict[str, str] = Field(
        default={
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-5": "gpt-5"
        },
        description="Available OpenAI models"
    )
    
    anthropic_models: Dict[str, str] = Field(
        default={
            "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku": "claude-3-5-haiku-20241022"
        },
        description="Available Anthropic models"
    )
    
    huggingface_models: Dict[str, str] = Field(
        default={
            "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
            "llama-4-scout": "openrouter/meta-llama/llama-4-scout",
        },
        description="Available HuggingFace models"
    )
    
    # Fallback model preferences by provider
    fallback_models: Dict[str, str] = Field(
        default={
            "openrouter": "openrouter/anthropic/claude-sonnet-4",
            "openai": "gpt-5",
            "anthropic": "claude-3-7-sonnet",
            "huggingface": "llama-4-scout"
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
    huggingface_token: Optional[str] = Field(
        default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN"),
        description="HuggingFace API token"
    )
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key"
    )
    anthropic_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"),
        description="Anthropic API key"
    )
    openrouter_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY"),
        description="OpenRouter API key"
    )
    browser_use_api_key: Optional[str] = Field(
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
    
    def get_model_provider(self, model_id: Optional[str] = None) -> Optional[str]:
        """Determine the provider for a given model ID."""
        model_id = model_id or self.model.name
        
        # Check if it's explicitly an OpenRouter model
        if "openrouter/" in model_id.lower():
            return "openrouter"
        # Check if it matches OpenAI patterns
        elif any(name in model_id.lower() for name in ["gpt-3.5", "gpt-4", "gpt", "o3"]):
            return "openai"
        # Check if it matches Anthropic patterns  
        elif "claude" in model_id.lower():
            return "anthropic"
        # Check if it matches HuggingFace patterns
        elif any(org in model_id.lower() for org in ["microsoft/", "meta-llama/", "qwen/", "mistralai/", "google/"]):
            return "huggingface"
        
        # Auto-detect based on available API keys
        if self.model.provider == "auto":
            if self.openrouter_api_key:
                return "openrouter"
            elif self.openai_api_key:
                return "openai"
            elif self.anthropic_api_key:
                return "anthropic"
            elif self.huggingface_token:
                return "huggingface"
        
        return self.model.provider if self.model.provider != "auto" else None
    
    def get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get the API key for a specific provider."""
        provider_keys = {
            "openrouter": self.openrouter_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "huggingface": self.huggingface_token
        }
        return provider_keys.get(provider)
    
    def get_default_model_for_provider(self, provider: str) -> Optional[str]:
        """Get the default/fallback model for a provider."""
        return self.model.fallback_models.get(provider)
    
    def list_available_models(self) -> Dict[str, list]:
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
    
    def get_model_info(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a model."""
        model_id = model_id or self.model.name
        provider = self.get_model_provider(model_id)
        
        return {
            "model_id": model_id,
            "provider": provider,
            "has_api_key": bool(self.get_api_key_for_provider(provider)) if provider else False,
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
            "top_p": self.model.top_p
        }

# Global settings instance
settings = Settings()
