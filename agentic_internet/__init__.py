"""Agentic Internet - Advanced AI agent for intelligent internet interactions."""

__version__ = "0.1.0"

from .agents.internet_agent import InternetAgent, ResearchAgent
from .agents.specialized_agents import (
    BrowserAutomationAgent,
    DataAnalysisAgent,
    ContentCreationAgent,
    MarketResearchAgent,
    TechnicalSupportAgent
)
from .agents.search_orchestrator import SearchOrchestrator, create_search_orchestrator
from .config.settings import settings

# Import multi-model system if dependencies are available
try:
    from .agents.multi_model_serpapi import (
        MultiModelSerpAPISystem,
        ModelManager,
        GoogleSearchTool,
        GoogleShoppingTool,
        GoogleMapsLocalTool,
        GoogleScholarTool,
        MultiEngineSearchTool
    )
    MULTI_MODEL_AVAILABLE = True
except ImportError:
    MULTI_MODEL_AVAILABLE = False
    MultiModelSerpAPISystem = None
    ModelManager = None
    GoogleSearchTool = None
    GoogleShoppingTool = None
    GoogleMapsLocalTool = None
    GoogleScholarTool = None
    MultiEngineSearchTool = None

__all__ = [
    "InternetAgent",
    "ResearchAgent",
    "BrowserAutomationAgent",
    "DataAnalysisAgent",
    "ContentCreationAgent",
    "MarketResearchAgent",
    "TechnicalSupportAgent",
    "SearchOrchestrator",
    "create_search_orchestrator",
    "settings",
    "MultiModelSerpAPISystem",
    "ModelManager",
    "GoogleSearchTool",
    "GoogleShoppingTool",
    "GoogleMapsLocalTool",
    "GoogleScholarTool",
    "MultiEngineSearchTool",
    "MULTI_MODEL_AVAILABLE",
    "__version__"
]
