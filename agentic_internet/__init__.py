"""Agentic Internet - Advanced AI agent for intelligent internet interactions."""

import logging

from .agents.internet_agent import InternetAgent, ResearchAgent
from .agents.search_orchestrator import SearchOrchestrator, create_search_orchestrator
from .agents.specialized_agents import (
    BrowserAutomationAgent,
    ContentCreationAgent,
    DataAnalysisAgent,
    MarketResearchAgent,
    TechnicalSupportAgent,
)
from .config.settings import settings
from .exceptions import (
    AgenticInternetError,
    APIKeyMissingError,
    BrowserAutomationError,
    CodeExecutionError,
    ConfigurationError,
    MCPError,
    ModelInitializationError,
    ProviderNotFoundError,
    SearchError,
    ToolExecutionError,
    UnsafeCodeError,
)

__version__ = "0.1.0"

# Configure package-level logging (NullHandler so library users control output)
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import multi-model system if dependencies are available
try:
    from .agents.multi_model_serpapi import (
        GoogleMapsLocalTool,
        GoogleScholarTool,
        GoogleSearchTool,
        GoogleShoppingTool,
        ModelManager,
        MultiEngineSearchTool,
        MultiModelSerpAPISystem,
    )
    MULTI_MODEL_AVAILABLE = True
except ImportError:
    MULTI_MODEL_AVAILABLE = False
    MultiModelSerpAPISystem = None  # type: ignore[assignment,misc]
    ModelManager = None  # type: ignore[assignment,misc]
    GoogleSearchTool = None  # type: ignore[assignment,misc]
    GoogleShoppingTool = None  # type: ignore[assignment,misc]
    GoogleMapsLocalTool = None  # type: ignore[assignment,misc]
    GoogleScholarTool = None  # type: ignore[assignment,misc]
    MultiEngineSearchTool = None  # type: ignore[assignment,misc]

__all__ = [
    "MULTI_MODEL_AVAILABLE",
    "APIKeyMissingError",
    "AgenticInternetError",
    "BrowserAutomationAgent",
    "BrowserAutomationError",
    "CodeExecutionError",
    "ConfigurationError",
    "ContentCreationAgent",
    "DataAnalysisAgent",
    "GoogleMapsLocalTool",
    "GoogleScholarTool",
    "GoogleSearchTool",
    "GoogleShoppingTool",
    "InternetAgent",
    "MCPError",
    "MarketResearchAgent",
    "ModelInitializationError",
    "ModelManager",
    "MultiEngineSearchTool",
    "MultiModelSerpAPISystem",
    "ProviderNotFoundError",
    "ResearchAgent",
    "SearchError",
    "SearchOrchestrator",
    "TechnicalSupportAgent",
    "ToolExecutionError",
    "UnsafeCodeError",
    "__version__",
    "create_search_orchestrator",
    "settings",
]
