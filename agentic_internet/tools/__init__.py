"""Tools module for AgenticInternet."""

from .web_search import WebSearchTool, WebScraperTool, NewsSearchTool
from .code_execution import PythonExecutorTool, DataAnalysisTool
from .browser_use import BrowserUseTool, AsyncBrowserUseTool, StructuredBrowserUseTool

# MCP integration (optional - only available if mcp package is installed)
try:
    from .mcp_integration import (
        MCPToolIntegration,
        MCPServerConfig,
        MCPServerManager,
        mcp_tools,
        load_mcp_config_from_env,
        is_mcp_available,
        MCP_AVAILABLE,
    )
except ImportError:
    # MCP packages not installed
    MCP_AVAILABLE = False
    MCPToolIntegration = None
    MCPServerConfig = None
    MCPServerManager = None
    mcp_tools = None
    load_mcp_config_from_env = None
    is_mcp_available = lambda: False

__all__ = [
    # Web tools
    "WebSearchTool",
    "WebScraperTool",
    "NewsSearchTool",
    # Code tools
    "PythonExecutorTool",
    "DataAnalysisTool",
    # Browser tools
    "BrowserUseTool",
    "AsyncBrowserUseTool",
    "StructuredBrowserUseTool",
    # MCP tools
    "MCPToolIntegration",
    "MCPServerConfig",
    "MCPServerManager",
    "mcp_tools",
    "load_mcp_config_from_env",
    "is_mcp_available",
    "MCP_AVAILABLE",
]