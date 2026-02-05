"""Tools module for AgenticInternet."""

from .browser_use import AsyncBrowserUseTool, BrowserUseTool, StructuredBrowserUseTool
from .code_execution import DataAnalysisTool, PythonExecutorTool
from .web_search import NewsSearchTool, WebScraperTool, WebSearchTool

# MCP integration (optional - only available if mcp package is installed)
try:
    from .mcp_integration import (
        MCP_AVAILABLE,
        MCPServerConfig,
        MCPServerManager,
        MCPToolIntegration,
        is_mcp_available,
        load_mcp_config_from_env,
        mcp_tools,
    )
except ImportError:
    # MCP packages not installed
    MCP_AVAILABLE = False
    MCPToolIntegration = None
    MCPServerConfig = None
    MCPServerManager = None
    mcp_tools = None
    load_mcp_config_from_env = None
    def is_mcp_available() -> bool:
        return False

__all__ = [
    "MCP_AVAILABLE",
    "AsyncBrowserUseTool",
    # Browser tools
    "BrowserUseTool",
    "DataAnalysisTool",
    "MCPServerConfig",
    "MCPServerManager",
    # MCP tools
    "MCPToolIntegration",
    "NewsSearchTool",
    # Code tools
    "PythonExecutorTool",
    "StructuredBrowserUseTool",
    "WebScraperTool",
    # Web tools
    "WebSearchTool",
    "is_mcp_available",
    "load_mcp_config_from_env",
    "mcp_tools",
]
