"""
MCP (Model Context Protocol) Integration for Agentic Internet.

This module provides integration with MCP servers via stdio and streamable-http transports,
allowing the agent to use external MCP tools as part of its toolkit.

Based on smolagents ToolCollection.from_mcp() which uses a context manager pattern.
See: https://huggingface.co/docs/smolagents/reference/tools#smolagents.ToolCollection.from_mcp
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Union
from urllib.parse import urlparse, urlunparse

from ..exceptions import MCPNotAvailableError

logger = logging.getLogger(__name__)

# Conditional imports for MCP support
MCP_AVAILABLE = False
try:
    from mcp import StdioServerParameters
    from smolagents import ToolCollection
    MCP_AVAILABLE = True
except ImportError:
    ToolCollection = None
    StdioServerParameters = None
    logger.debug("MCP packages not installed. Install with: pip install smolagents mcp")


def check_mcp_available() -> None:
    """Check if MCP packages are available and raise informative error if not."""
    if not MCP_AVAILABLE:
        raise MCPNotAvailableError()

def _normalize_http_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url

    path = parsed.path or "/"
    if path == "/":
        logger.warning("HTTP MCP URL missing path; defaulting to /mcp/ endpoint")
        path = "/mcp/"
        return urlunparse(parsed._replace(path=path))

    return url


class MCPToolIntegration:
    """
    Integration layer for MCP (Model Context Protocol) servers using smolagents.

    Supports both stdio and streamable-http transports for connecting to MCP servers.
    Uses smolagents' native ToolCollection.from_mcp() method with context manager.

    Example:
        >>> with MCPToolIntegration.connect(
        ...     server_config="./server.py",
        ...     trust_remote_code=True
        ... ) as tools:
        ...     agent = InternetAgent(tools=[*tools])
        ...     result = agent.run("Use MCP tools")
    """

    def __init__(
        self,
        server_config: Union[str, dict[str, Any]],
        transport_type: str = "stdio",
        env: dict[str, str] | None = None,
        trust_remote_code: bool = False,
        structured_output: bool | None = None,
    ):
        """
        Initialize MCP tool integration.

        Args:
            server_config: Either a file path to MCP server script (for stdio)
                          or a URL string (for http), or a dict with detailed config
            transport_type: Type of transport ("stdio" or "streamable-http")
            env: Environment variables to pass to stdio servers
            trust_remote_code: Whether to trust execution of code from MCP tools.
                              MUST be True to use MCP tools.
            structured_output: Whether to enable structured output features (optional)
        """
        check_mcp_available()

        self.server_config = server_config
        self.transport_type = transport_type
        self.env = env or {}
        self.trust_remote_code = trust_remote_code
        self.structured_output = structured_output
        self._context_manager = None
        self._tools: list[Any] = []

    def _create_server_parameters(self) -> Union['StdioServerParameters', dict[str, Any]]:
        """
        Create server parameters based on transport type.

        Returns:
            StdioServerParameters for stdio or dict for HTTP transport
        """
        if self.transport_type == "stdio":
            if isinstance(self.server_config, dict):
                server_path = self.server_config.get("path")
                command = self.server_config.get("command", "python")
            else:
                server_path = str(self.server_config)
                command = "python"

            if not server_path:
                raise ValueError("Server path required for stdio transport")

            # Resolve to absolute path
            server_path = str(Path(server_path).resolve())

            # Merge system environment with custom env
            merged_env = {**os.environ, **self.env}

            # Create StdioServerParameters for stdio transport
            return StdioServerParameters(
                command=command,
                args=[server_path],
                env=merged_env,
            )

        elif self.transport_type in ("streamable-http", "http"):
            url = self.server_config.get("url") if isinstance(self.server_config, dict) else str(self.server_config)

            if not url:
                raise ValueError("URL required for http transport")

            url = _normalize_http_url(url)

            # Return dict format for HTTP transport
            return {
                "url": url,
                "transport": "streamable-http"
            }

        else:
            raise ValueError(
                f"Unsupported transport type: {self.transport_type}. "
                "Use 'stdio' or 'streamable-http'"
            )

    def get_server_parameters(self) -> Union['StdioServerParameters', dict[str, Any]]:
        """Get the server parameters for this integration."""
        return self._create_server_parameters()

    @classmethod
    @contextmanager
    def connect(
        cls,
        server_config: Union[str, dict[str, Any]],
        transport_type: str = "stdio",
        env: dict[str, str] | None = None,
        trust_remote_code: bool = False,
        structured_output: bool | None = None,
    ):
        """
        Context manager to connect to MCP server and get tools.

        This is the recommended way to use MCP tools as it properly handles
        connection lifecycle.

        Args:
            server_config: Server path (stdio) or URL (http)
            transport_type: "stdio" or "streamable-http"
            env: Environment variables for stdio servers
            trust_remote_code: Must be True to execute MCP tools
            structured_output: Enable structured output features

        Yields:
            List of MCP tools ready for use with agents

        Example:
            >>> with MCPToolIntegration.connect(
            ...     "./my_server.py",
            ...     trust_remote_code=True
            ... ) as tools:
            ...     agent = InternetAgent(tools=tools)
            ...     result = agent.run("task")
        """
        check_mcp_available()

        integration = cls(
            server_config=server_config,
            transport_type=transport_type,
            env=env,
            trust_remote_code=trust_remote_code,
            structured_output=structured_output,
        )

        server_params = integration._create_server_parameters()

        # Build kwargs for from_mcp
        mcp_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        if structured_output is not None:
            mcp_kwargs["structured_output"] = structured_output

        # Use the context manager from smolagents
        with ToolCollection.from_mcp(server_params, **mcp_kwargs) as tool_collection:
            yield tool_collection.tools

    def get_tools_list(self) -> list[Any]:
        """Get the list of loaded tools (after entering context)."""
        return self._tools


class MCPServerConfig:
    """Configuration for an MCP server."""

    def __init__(
        self,
        name: str,
        server_config: Union[str, dict[str, Any]],
        transport_type: str = "stdio",
        env: dict[str, str] | None = None,
        trust_remote_code: bool = False,
    ):
        self.name = name
        self.server_config = server_config
        self.transport_type = transport_type
        self.env = env or {}
        self.trust_remote_code = trust_remote_code

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "server_config": self.server_config,
            "transport_type": self.transport_type,
            "env": self.env,
            "trust_remote_code": self.trust_remote_code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'MCPServerConfig':
        """Create config from dictionary."""
        return cls(
            name=data.get("name", "unnamed"),
            server_config=data.get("server_config", data.get("path") or data.get("url")),
            transport_type=data.get("transport_type", "stdio"),
            env=data.get("env", {}),
            trust_remote_code=data.get("trust_remote_code", False),
        )


class MCPServerManager:
    """
    Manages multiple MCP server configurations.

    Note: Due to the context manager pattern required by smolagents,
    this class stores configurations rather than active connections.
    Use the context manager methods to connect to servers.

    Example:
        >>> manager = MCPServerManager()
        >>> manager.add_server("myserver", "./server.py", trust_remote_code=True)
        >>> with manager.connect_all() as tools:
        ...     agent = InternetAgent(tools=tools)
        ...     result = agent.run("task")
    """

    def __init__(self):
        self.servers: dict[str, MCPServerConfig] = {}

    def add_server(
        self,
        name: str,
        server_config: Union[str, dict[str, Any]],
        transport_type: str = "stdio",
        env: dict[str, str] | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        """
        Add an MCP server configuration.

        Args:
            name: Unique identifier for this server
            server_config: Server configuration (path or URL)
            transport_type: Type of transport ("stdio" or "streamable-http")
            env: Environment variables for the server
            trust_remote_code: Whether to trust remote code execution
        """
        config = MCPServerConfig(
            name=name,
            server_config=server_config,
            transport_type=transport_type,
            env=env,
            trust_remote_code=trust_remote_code,
        )
        self.servers[name] = config

    def remove_server(self, name: str) -> bool:
        """Remove a server configuration by name."""
        if name in self.servers:
            del self.servers[name]
            return True
        return False

    def list_servers(self) -> list[str]:
        """List all configured server names."""
        return list(self.servers.keys())

    def get_server(self, name: str) -> MCPServerConfig | None:
        """Get a server configuration by name."""
        return self.servers.get(name)

    @contextmanager
    def connect(self, server_name: str):
        """
        Connect to a single MCP server by name.

        Args:
            server_name: Name of the server to connect to

        Yields:
            List of tools from the server
        """
        config = self.servers.get(server_name)
        if not config:
            raise ValueError(f"Server '{server_name}' not found")

        with MCPToolIntegration.connect(
            server_config=config.server_config,
            transport_type=config.transport_type,
            env=config.env,
            trust_remote_code=config.trust_remote_code,
        ) as tools:
            yield tools

    @contextmanager
    def connect_all(self):
        """
        Connect to all configured MCP servers and yield combined tools.

        Note: This connects to servers sequentially due to the context manager
        pattern. For production use with many servers, consider connecting
        to servers individually.

        Yields:
            Combined list of tools from all servers
        """
        check_mcp_available()

        if not self.servers:
            yield []
            return

        # For a single server, use simple approach
        if len(self.servers) == 1:
            server_name = next(iter(self.servers.keys()))
            with self.connect(server_name) as tools:
                yield tools
            return

        # For multiple servers, we need to nest context managers
        # This is a limitation of the synchronous context manager pattern
        all_tools = []
        server_list = list(self.servers.values())

        # Connect to first server
        with MCPToolIntegration.connect(
            server_config=server_list[0].server_config,
            transport_type=server_list[0].transport_type,
            env=server_list[0].env,
            trust_remote_code=server_list[0].trust_remote_code,
        ) as tools:
            all_tools.extend(tools)

            # Recursively connect to remaining servers if any
            if len(server_list) > 1:
                # For simplicity, only support 2 servers in nested mode
                # For more servers, recommend using connect() individually
                if len(server_list) == 2:
                    with MCPToolIntegration.connect(
                        server_config=server_list[1].server_config,
                        transport_type=server_list[1].transport_type,
                        env=server_list[1].env,
                        trust_remote_code=server_list[1].trust_remote_code,
                    ) as tools2:
                        all_tools.extend(tools2)
                        yield all_tools
                else:
                    logger.warning(
                        "connect_all() with >2 servers not fully supported. "
                        "Consider connecting to servers individually."
                    )
                    yield all_tools
            else:
                yield all_tools


# Convenience context manager for simple use cases
@contextmanager
def mcp_tools(
    server_path: str | None = None,
    server_url: str | None = None,
    env: dict[str, str] | None = None,
    trust_remote_code: bool = False,
    structured_output: bool | None = None,
):
    """
    Context manager to quickly get MCP tools from a single server.

    This is the recommended way to use MCP tools as it properly handles
    the connection lifecycle.

    Args:
        server_path: Path to local MCP server script (for stdio)
        server_url: URL of remote MCP server (for streamable-http)
        env: Environment variables (for stdio servers)
        trust_remote_code: Whether to trust execution of code from MCP tools.
                          MUST be True to use MCP tools.
        structured_output: Whether to enable structured output features

    Yields:
        List of MCP tools ready for use with agents

    Example:
        >>> with mcp_tools(server_path="./my_mcp_server.py", trust_remote_code=True) as tools:
        ...     agent = InternetAgent(tools=tools)
        ...     result = agent.run("Use MCP tools")
    """
    check_mcp_available()

    if server_path:
        transport_type = "stdio"
        server_config = server_path
    elif server_url:
        transport_type = "streamable-http"
        server_config = server_url
    else:
        raise ValueError("Either server_path or server_url must be provided")

    with MCPToolIntegration.connect(
        server_config=server_config,
        transport_type=transport_type,
        env=env,
        trust_remote_code=trust_remote_code,
        structured_output=structured_output,
    ) as tools:
        yield tools


def load_mcp_config_from_env() -> list[MCPServerConfig]:
    """
    Load MCP server configurations from environment variables.

    Expected format:
        MCP_SERVER_1_TYPE=stdio
        MCP_SERVER_1_PATH=/path/to/server.py
        MCP_SERVER_1_TRUST=true
        MCP_SERVER_1_ENV_API_KEY=secret

        MCP_SERVER_2_TYPE=http
        MCP_SERVER_2_URL=https://api.example.com/mcp
        MCP_SERVER_2_TRUST=true

    Returns:
        List of MCPServerConfig objects
    """
    configs = []
    i = 1

    while True:
        prefix = f"MCP_SERVER_{i}_"
        server_type = os.getenv(f"{prefix}TYPE")

        if not server_type:
            break

        name = os.getenv(f"{prefix}NAME", f"server_{i}")
        trust_remote_code = os.getenv(f"{prefix}TRUST", "false").lower() in ("true", "1", "yes")

        if server_type == "stdio":
            server_path = os.getenv(f"{prefix}PATH")
            if not server_path:
                logger.warning(f"MCP_SERVER_{i}: Missing PATH for stdio server")
                i += 1
                continue

            # Collect environment variables
            env = {}
            for key, value in os.environ.items():
                if key.startswith(f"{prefix}ENV_"):
                    env_key = key.replace(f"{prefix}ENV_", "")
                    env[env_key] = value

            config = MCPServerConfig(
                name=name,
                server_config=server_path,
                transport_type="stdio",
                env=env,
                trust_remote_code=trust_remote_code,
            )

        elif server_type in ("http", "streamable-http"):
            server_url = os.getenv(f"{prefix}URL")
            if not server_url:
                logger.warning(f"MCP_SERVER_{i}: Missing URL for http server")
                i += 1
                continue

            config = MCPServerConfig(
                name=name,
                server_config=server_url,
                transport_type="streamable-http",
                env={},
                trust_remote_code=trust_remote_code,
            )
        else:
            logger.warning(f"MCP_SERVER_{i}: Unknown type '{server_type}'")
            i += 1
            continue

        configs.append(config)
        i += 1

    return configs


def is_mcp_available() -> bool:
    """Check if MCP packages are available."""
    return MCP_AVAILABLE


# Module exports
__all__ = [
    "MCP_AVAILABLE",
    "MCPServerConfig",
    "MCPServerManager",
    "MCPToolIntegration",
    "check_mcp_available",
    "is_mcp_available",
    "load_mcp_config_from_env",
    "mcp_tools",
]
