"""Model construction and role-based selection for multi-model orchestration."""

import logging
import os
from typing import Any

from smolagents import LiteLLMModel

from agentic_internet.config.settings import settings

logger = logging.getLogger(__name__)


# Multi-Model Configuration for Specialized Tasks
# Multi-Model Configuration for Specialized Tasks
class ModelManager:
    """Manages multiple LiteLLM models with strategic assignment"""

    def __init__(self):
        # Initialize comprehensive model suite
        api_key = os.environ.get("OPENROUTER_API_KEY")
        self.api_key = api_key
        if not api_key:
            logger.warning("OPENROUTER_API_KEY not found. Model initialization will be skipped.")
            self.models_available = False
            return

        self.models_available = True

        try:
            self.claude = LiteLLMModel(
                model_id="openrouter/anthropic/claude-opus-4.5", api_key=api_key, temperature=0.7
            )
        except Exception as e:
            logger.warning("Failed to initialize Claude model: %s", e)
            self.claude = None

        # Initialize all other models with error handling (updated November 2026)
        models_config = [
            ("deepseek", "openrouter/deepseek/deepseek-r1-0528", 0.6),
            ("deepseek_chat", "openrouter/deepseek/deepseek-v3.2", 0.6),
            ("deepseek_speciale", "openrouter/deepseek/deepseek-v4.1-flash-1017", 0.6),
            ("mistral_small", "openrouter/mistralai/mistral-small-3.2-24b-instruct", 0.8),
            ("sonar", "openrouter/perplexity/sonar-pro", 0.8),
            ("sonar_reasoning", "openrouter/perplexity/sonar-reasoning-pro", None),
            ("openai_o4", "openrouter/openai/gpt-6-luna", None),
            ("mistral_large", "openrouter/mistralai/mistral-large-2512", 0.8),
            ("gemini", "openrouter/google/gemini-3-flash-preview", 0.7),
            ("gpt_oss", "openrouter/openai/gpt-oss-120b", None),
            ("gemini_pro", "openrouter/google/gemini-4-argon", 0.6),
            ("chatgpt", "openrouter/openai/gpt-5.2-chat", 0.5),
            ("qwen3_coder", "openrouter/qwen/qwen3-coder", 0.6),
            ("qwen_big", "openrouter/qwen/qwen3.8-max-prime", 0.6),
            ("llama4", "openrouter/meta-llama/llama-4-maverick", 0.6),
            ("xai", "openrouter/x-ai/grok-4.3", 0.8),
            ("gpt5", "openrouter/openai/gpt-5.2", 0.6),
            ("kimi", "openrouter/moonshotai/kimi-k2.5", 0.6),
            ("devstral", "openrouter/mistralai/devstral-2512", 0.6),
            ("mimo", "openrouter/liquid/lfm-3-1.7b", 0.7),
            ("minimax", "openrouter/minimax/minimax-m3.1-flash", 0.6),
        ]

        for attr_name, model_id, temp in models_config:
            try:
                if temp is not None:
                    model = LiteLLMModel(model_id=model_id, api_key=api_key, temperature=temp)
                else:
                    model = LiteLLMModel(model_id=model_id, api_key=api_key)
                setattr(self, attr_name, model)
            except Exception as e:
                logger.warning("Failed to initialize %s model: %s", attr_name, e)
                setattr(self, attr_name, None)

    def get_model_for_role(self, role: str) -> LiteLLMModel | None:
        """Strategic model assignment based on agent role and capabilities"""
        if not self.models_available:
            return None

        # Resolve explicit aliases centrally so a requested model is never replaced by an older fallback.
        configured_model_id = settings.model.openrouter_models.get(role.lower())
        if configured_model_id:
            try:
                return LiteLLMModel(
                    model_id=configured_model_id,
                    api_key=self.api_key,
                    temperature=settings.model.temperature,
                    max_tokens=settings.model.max_tokens,
                    top_p=settings.model.top_p,
                )
            except Exception as e:
                logger.warning("Failed to initialize configured model %s: %s", role, e)
                return None

        # First check if the role corresponds to a direct model name
        # This allows users to specify models like 'gpt-5', 'claude-4', etc.
        model_name_mappings = {
            "gpt-5.2": "gpt5",
            "gpt-5": "gpt5",
            "claude-opus-4.5": "claude",
            "claude": "claude",
            "deepseek": "deepseek",
            "deepseek-v3.2": "deepseek_chat",
            "deepseek-speciale": "deepseek_speciale",
            "gemini": "gemini",
            "gemini-3-flash": "gemini",
            "gemini-3-pro": "gemini_pro",
            "gemini-pro": "gemini_pro",
            "sonar": "sonar",
            "sonar-pro": "sonar",
            "sonar-reasoning": "sonar_reasoning",
            "mistral": "mistral_large",
            "mistral-small": "mistral_small",
            "mistral-large": "mistral_large",
            "devstral": "devstral",
            "qwen": "qwen_big",
            "qwen-coder": "qwen3_coder",
            "xai": "xai",
            "grok": "xai",
            "grok-4.1": "xai",
            "kimi": "kimi",
            "kimi-k2.5": "kimi",
            "chatgpt": "chatgpt",
            "o4-mini": "openai_o4",
            "o3": "openai_o4",
            "mimo": "mimo",
            "minimax": "minimax",
            "llama-4": "llama4",
        }

        # Check if the role is a direct model specification
        if role.lower() in model_name_mappings:
            model_attr = model_name_mappings[role.lower()]
            model = getattr(self, model_attr, None)
            if model is not None:
                logger.info("Using specified model: %s", role)
                return model
            else:
                logger.warning("Requested model %s is not available", role)

        # Fall back to role-based assignments (updated Feb 2026)
        model_assignments = {
            # Orchestrator: Needs excellent reasoning and coordination
            "orchestrator": getattr(self, "claude", None),  # Claude Opus 4.5
            # Search Researcher: Needs broad knowledge and search optimization
            "search_researcher": getattr(self, "sonar", None),  # Perplexity Sonar Pro
            # E-commerce Analyst: Needs structured analysis and price comparison
            "ecommerce_analyst": getattr(self, "gemini_pro", None),  # Gemini 3 Pro
            # Local Business Analyst: Needs location awareness and business intelligence
            "local_business_analyst": getattr(self, "mistral_large", None),  # Mistral Large 2512
            # Academic Researcher: Needs research methodology and citation analysis
            "academic_researcher": getattr(self, "sonar_reasoning", None),  # Sonar Reasoning Pro
            # Data Analyst: Needs coding and mathematical capabilities
            "data_analyst": getattr(self, "qwen3_coder", None),  # Qwen3 Coder
            # Browser Navigator: Needs web interaction understanding
            "browser_navigator": getattr(self, "gpt5", None),  # GPT-5.2
            # Competitive Intelligence: Needs synthesis and strategic analysis
            "competitive_analyst": getattr(self, "gpt5", None),  # GPT-5.2
            # Market Research Synthesizer: Needs comprehensive analysis
            "market_synthesizer": getattr(self, "qwen_big", None),  # Qwen3 235B
            # Creative/Content Tasks: Needs creative capabilities
            "content_creator": getattr(self, "xai", None),  # Grok 4.1 Fast
            # Fast Response Tasks: Needs speed
            "quick_responder": getattr(self, "gemini", None),  # Gemini 3 Flash
            # Reasoning Tasks: Needs deep logical thinking
            "deep_reasoner": getattr(self, "openai_o4", None),  # O4 Mini
            # Conversational Tasks: Needs natural dialogue
            "conversationalist": getattr(self, "chatgpt", None),  # GPT-5.2 Chat
            # Long Context Tasks: Needs extensive memory
            "long_context": getattr(self, "kimi", None),  # Kimi K2.5
            # Agentic Coding: Needs strong code understanding
            "agentic_coder": getattr(self, "devstral", None),  # Devstral 2
            # Cost-effective: Good performance at low cost
            "cost_effective": getattr(self, "mimo", None),  # MiMo-V2-Flash
        }

        model = model_assignments.get(role, getattr(self, "claude", None))  # Default to Claude

        # If the requested model is None, try to find any available model
        if model is None:
            for attr_name in ["claude", "deepseek_chat", "gemini", "mistral_small"]:
                fallback_model = getattr(self, attr_name, None)
                if fallback_model is not None:
                    logger.warning("Using fallback model %s for role %s", attr_name, role)
                    return fallback_model

        return model

    def get_model_capabilities(self) -> dict[str, dict[str, Any]]:
        """Get detailed capabilities of each model for optimization"""
        return {
            "claude": {"strengths": ["reasoning", "planning", "analysis"], "best_for": "orchestration"},
            "deepseek": {"strengths": ["reasoning", "math", "logic"], "best_for": "complex_analysis"},
            "deepseek_chat": {"strengths": ["conversation", "reasoning"], "best_for": "interactive_tasks"},
            "deepseek_speciale": {"strengths": ["reasoning", "specialized"], "best_for": "expert_tasks"},
            "mistral_small": {"strengths": ["efficiency", "speed"], "best_for": "quick_tasks"},
            "sonar": {"strengths": ["search", "web_knowledge"], "best_for": "research_tasks"},
            "sonar_reasoning": {"strengths": ["research", "citations"], "best_for": "academic_research"},
            "openai_o4": {"strengths": ["reasoning", "problem_solving"], "best_for": "complex_reasoning"},
            "mistral_large": {"strengths": ["analysis", "business"], "best_for": "business_intelligence"},
            "gemini": {"strengths": ["speed", "multimodal"], "best_for": "fast_response"},
            "gpt_oss": {"strengths": ["general", "balanced"], "best_for": "general_tasks"},
            "gemini_pro": {"strengths": ["analysis", "structured_data"], "best_for": "data_analysis"},
            "chatgpt": {"strengths": ["conversation", "creativity"], "best_for": "chat_tasks"},
            "qwen3_coder": {"strengths": ["coding", "technical"], "best_for": "programming_analysis"},
            "qwen_big": {"strengths": ["context", "synthesis"], "best_for": "comprehensive_analysis"},
            "llama4": {"strengths": ["navigation", "exploration"], "best_for": "web_navigation"},
            "xai": {"strengths": ["creativity", "innovation", "speed"], "best_for": "creative_tasks"},
            "gpt5": {"strengths": ["advanced_reasoning", "strategy"], "best_for": "strategic_analysis"},
            "kimi": {"strengths": ["long_context", "memory"], "best_for": "extended_analysis"},
            "devstral": {"strengths": ["coding", "agentic"], "best_for": "agentic_coding"},
            "mimo": {"strengths": ["efficiency", "cost"], "best_for": "cost_effective_tasks"},
            "minimax": {"strengths": ["reasoning", "balanced"], "best_for": "general_reasoning"},
        }
