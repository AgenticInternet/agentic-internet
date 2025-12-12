# MCP Integration for AgenticInternet

This guide explains how to integrate MCP (Model Context Protocol) servers with the AgenticInternet framework, enabling your agents to use external MCP tools.

## Overview

The MCP integration allows AgenticInternet to connect to MCP servers and use their tools seamlessly. This is powered by [smolagents](https://huggingface.co/docs/smolagents) native MCP support using a **context manager pattern**.

### Supported Transports

- **stdio**: Local MCP servers running as subprocesses
- **streamable-http**: Remote MCP servers accessible via HTTP

## Installation

```bash
# Basic installation
pip install smolagents mcp

# Or with uv
uv pip install smolagents mcp

# For creating FastMCP servers
pip install fastmcp
```

## Quick Start

### Using stdio Transport (Local Server)

```python
from agentic_internet.tools.mcp_integration import mcp_tools
from agentic_internet.agents.internet_agent import InternetAgent

# Use context manager to connect to MCP server
with mcp_tools(
    server_path="./my_mcp_server.py",
    trust_remote_code=True,  # Required to execute MCP tools
) as tools:
    # Create agent with MCP tools
    agent = InternetAgent(
        tools=list(tools),  # Convert to list
        verbose=True
    )
    
    # Use the agent with MCP tools
    result = agent.run("Use the MCP tools to accomplish this task")
    print(result)

# Connection is automatically closed when exiting the context
```

### Using Streamable HTTP Transport (Remote Server)

```python
from agentic_internet.tools.mcp_integration import mcp_tools
from agentic_internet.agents.internet_agent import InternetAgent

# Connect to remote MCP server
with mcp_tools(
    server_url="http://localhost:8000/mcp/",
    trust_remote_code=True,
) as tools:
    agent = InternetAgent(
        tools=list(tools),
        verbose=True
    )
    
    result = agent.run("Search for information using MCP tools")
    print(result)
```

## CLI Usage

The AgenticInternet CLI includes MCP commands:

```bash
# Show MCP integration status
uv run python -m agentic_internet.cli mcp info

# List configured MCP servers (from environment)
uv run python -m agentic_internet.cli mcp list

# Test connection to an MCP server
uv run python -m agentic_internet.cli mcp test --server ./my_server.py

# Run a task with MCP tools
uv run python -m agentic_internet.cli mcp run \
    "Calculate sum of 1 and 2" \
    --server ./my_server.py \
    --trust
```

## Advanced Usage

### Using MCPToolIntegration Class

For more control, use the `MCPToolIntegration.connect()` class method:

```python
from agentic_internet.tools.mcp_integration import MCPToolIntegration

# Using the class method context manager
with MCPToolIntegration.connect(
    server_config="./server.py",
    transport_type="stdio",
    env={
        "API_KEY": "your-api-key",
        "LOG_LEVEL": "INFO"
    },
    trust_remote_code=True,
) as tools:
    # Access tools
    for tool in tools:
        print(f"Tool: {tool.name} - {tool.description}")
```

### Managing Multiple MCP Servers

```python
from agentic_internet.tools.mcp_integration import MCPServerManager
from agentic_internet.agents.internet_agent import InternetAgent

# Create manager
manager = MCPServerManager()

# Add server configurations
manager.add_server(
    name="local_tools",
    server_config="./local_server.py",
    transport_type="stdio",
    trust_remote_code=True,
)

manager.add_server(
    name="remote_tools",
    server_config="http://api.example.com/mcp",
    transport_type="streamable-http",
    trust_remote_code=True,
)

# Connect to a specific server
with manager.connect("local_tools") as tools:
    agent = InternetAgent(tools=list(tools), verbose=True)
    result = agent.run("Use local MCP tools")
    print(result)

# Or connect to all servers (limited to 2 servers)
with manager.connect_all() as all_tools:
    agent = InternetAgent(tools=list(all_tools), verbose=True)
    result = agent.run("Use tools from multiple servers")
```

### Environment-Based Configuration

Configure MCP servers using environment variables:

```bash
# .env file
MCP_SERVER_1_TYPE=stdio
MCP_SERVER_1_PATH=/path/to/server.py
MCP_SERVER_1_NAME=my-server
MCP_SERVER_1_TRUST=true
MCP_SERVER_1_ENV_API_KEY=secret_key

MCP_SERVER_2_TYPE=http
MCP_SERVER_2_URL=http://localhost:8000/mcp
MCP_SERVER_2_NAME=remote-server
MCP_SERVER_2_TRUST=true
```

```python
from agentic_internet.tools.mcp_integration import (
    load_mcp_config_from_env,
    MCPServerManager,
)

# Load configurations from environment
configs = load_mcp_config_from_env()

# Create manager and add servers from config
manager = MCPServerManager()
for config in configs:
    manager.add_server(
        name=config.name,
        server_config=config.server_config,
        transport_type=config.transport_type,
        env=config.env,
        trust_remote_code=config.trust_remote_code,
    )

print(f"Loaded {len(configs)} server configurations")
print(f"Servers: {manager.list_servers()}")
```

## Creating MCP Servers

### Example FastMCP Server

See `agentic_internet/examples/example_mcp_server.py` for a complete example.

```python
from fastmcp import FastMCP

mcp = FastMCP(name="My MCP Server")

@mcp.tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@mcp.tool
def get_info(query: str) -> dict:
    """Get information based on query."""
    return {
        "query": query,
        "result": f"Information about {query}"
    }

if __name__ == "__main__":
    mcp.run()
```

### Running the Server

```bash
# For stdio (used by AgenticInternet)
python example_mcp_server.py

# For HTTP (testing/debugging)
fastmcp run example_mcp_server.py --transport http --port 8000
```

## Integration with InternetAgent

### Adding MCP Tools to Existing Agents

```python
from agentic_internet.agents.internet_agent import InternetAgent
from agentic_internet.tools.mcp_integration import mcp_tools

# Use context manager pattern
with mcp_tools(
    server_path="./my_server.py",
    trust_remote_code=True
) as tools:
    # Create agent with MCP tools
    agent = InternetAgent(
        tools=list(tools),  # MCP tools
        verbose=True,
        agent_type="tool_calling"  # or "code" for code execution
    )
    
    # Run task
    result = agent.run("Search the web and use MCP tools to analyze results")
    print(result)

# Connection is automatically closed when exiting the context
```

### Combining with Built-in Tools

```python
from agentic_internet.agents.internet_agent import InternetAgent
from agentic_internet.tools.mcp_integration import mcp_tools
from agentic_internet.tools import get_default_tools

# Get built-in tools
builtin_tools = get_default_tools()

# Add MCP tools
with mcp_tools(
    server_path="./server.py",
    trust_remote_code=True
) as mcp_tool_list:
    # Combine built-in and MCP tools
    all_tools = builtin_tools + list(mcp_tool_list)
    
    agent = InternetAgent(tools=all_tools, verbose=True)
    result = agent.run("Use both web search and MCP tools")
    print(result)
```

## Security Considerations

### trust_remote_code Parameter

**IMPORTANT**: Always set `trust_remote_code=True` when using MCP tools, but only with servers you trust:

```python
from agentic_internet.tools.mcp_integration import mcp_tools

# ✅ GOOD: Trust code from your own servers
with mcp_tools(
    server_path="./my_trusted_server.py",
    trust_remote_code=True  # Required for execution
) as tools:
    # Use tools safely
    pass

# ❌ BAD: Don't use untrusted MCP servers
with mcp_tools(
    server_url="http://untrusted-server.com/mcp",
    trust_remote_code=True  # Dangerous!
) as tools:
    pass
```

### Best Practices

1. **Verify server code** before running it
2. **Use environment variables** for sensitive data (API keys, tokens)
3. **Run in isolated environments** when testing new servers
4. **Validate server responses** before using them
5. **Monitor resource usage** of MCP servers

## Troubleshooting

### Common Issues

#### "Required packages not installed"

```bash
pip install smolagents mcp
```

#### "Server path required for stdio transport"

Ensure you provide a valid path to your MCP server:

```python
from agentic_internet.tools.mcp_integration import mcp_tools

with mcp_tools(
    server_path="/absolute/path/to/server.py",  # Use absolute path
    trust_remote_code=True
) as tools:
    pass
```

#### "Connection refused" (HTTP transport)

Make sure your HTTP server is running:

```bash
# Terminal 1: Start the server
fastmcp run server.py --transport http --port 8000

# Terminal 2: Run your agent
python your_agent.py
```

#### "trust_remote_code must be True"

MCP tool execution requires explicit trust:

```python
from agentic_internet.tools.mcp_integration import mcp_tools

with mcp_tools(
    server_path="./server.py",
    trust_remote_code=True  # Must be True
) as tools:
    pass
```

### Debug Mode

Enable verbose logging to troubleshoot issues:

```python
import logging
from agentic_internet.tools.mcp_integration import mcp_tools
from agentic_internet.agents.internet_agent import InternetAgent

logging.basicConfig(level=logging.DEBUG)

with mcp_tools(server_path="./server.py", trust_remote_code=True) as tools:
    agent = InternetAgent(
        tools=list(tools),
        verbose=True  # Enable verbose output
    )
```

## Examples

See the `agentic_internet/examples/` directory for complete examples:

- `example_mcp_server.py`: Sample MCP server with various tools
- `example_mcp_usage.py`: Comprehensive usage examples

Run the examples:

```bash
# Run usage examples
python -m agentic_internet.examples.example_mcp_usage

# Run the example server (for testing)
python -m agentic_internet.examples.example_mcp_server
```

## API Reference

### mcp_tools()

Convenience context manager to quickly connect to an MCP server and get tools.

**Parameters:**
- `server_path` (str, optional): Path to local MCP server script (for stdio transport)
- `server_url` (str, optional): URL of remote MCP server (for HTTP transport)
- `env` (dict, optional): Environment variables for stdio servers
- `trust_remote_code` (bool): Whether to trust code execution (default: False)

**Yields:** List of tools from the MCP server

**Example:**
```python
with mcp_tools(server_path="./server.py", trust_remote_code=True) as tools:
    # tools is a list of Tool objects
    agent = InternetAgent(tools=list(tools))
```

### MCPToolIntegration

Main class for MCP integration with more configuration options.

**Class Methods:**
- `connect(server_config, transport_type, env, trust_remote_code)`: Context manager to connect and get tools

**Parameters for connect():**
- `server_config` (str): Server path (stdio) or URL (HTTP)
- `transport_type` (str): "stdio" or "streamable-http"
- `env` (dict, optional): Environment variables
- `trust_remote_code` (bool): Trust code execution (default: False)

**Example:**
```python
with MCPToolIntegration.connect(
    server_config="./server.py",
    transport_type="stdio",
    trust_remote_code=True
) as tools:
    # Use tools
    pass
```

### MCPServerManager

Manages multiple MCP server configurations.

**Methods:**
- `add_server(name, server_config, transport_type, env, trust_remote_code)`: Register a server
- `list_servers()`: List registered server names
- `connect(server_name)`: Context manager to connect to a specific server
- `connect_all()`: Context manager to connect to all servers (max 2)

**Example:**
```python
manager = MCPServerManager()
manager.add_server("my-server", "./server.py", "stdio", trust_remote_code=True)

with manager.connect("my-server") as tools:
    # Use tools from "my-server"
    pass
```

### MCPServerConfig

Dataclass for server configuration (returned by `load_mcp_config_from_env()`).

**Attributes:**
- `name` (str): Server name
- `server_config` (str): Server path or URL
- `transport_type` (str): "stdio" or "streamable-http"
- `env` (dict): Environment variables
- `trust_remote_code` (bool): Whether to trust remote code

### load_mcp_config_from_env()

Load MCP server configurations from environment variables.

**Returns:** List of `MCPServerConfig` objects

### is_mcp_available()

Check if MCP packages are installed.

**Returns:** `bool` - True if smolagents and mcp packages are available

## Resources

- [smolagents Documentation](https://huggingface.co/docs/smolagents)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://gofastmcp.com/)
- [AgenticInternet GitHub](https://github.com/AgenticInternet/agentic-internet)

## Contributing

Contributions are welcome! Please see the main README for contribution guidelines.

## License

MIT License - see LICENSE file for details.
