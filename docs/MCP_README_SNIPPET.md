# MCP Integration Feature

Add this section to the main README.md to highlight the new MCP integration capability.

---

## 🔌 MCP Integration

AgenticInternet now supports integration with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers, allowing your agents to use external tools seamlessly.

### Quick Example

```python
import asyncio
from agentic_internet.tools.mcp_integration import create_mcp_tools
from agentic_internet.agents.internet_agent import InternetAgent

async def main():
    # Load tools from an MCP server
    mcp_tools = await create_mcp_tools(
        server_path="./my_mcp_server.py",
        trust_remote_code=True
    )
    
    # Create agent with MCP tools
    agent = InternetAgent(
        tools=[*mcp_tools.tools],
        verbose=True
    )
    
    result = agent.run("Use MCP tools to complete this task")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Features

- ✅ **stdio transport**: Connect to local MCP servers running as subprocesses
- ✅ **streamable-http transport**: Connect to remote MCP servers via HTTP
- ✅ **Multiple server management**: Coordinate tools from multiple MCP servers
- ✅ **Environment-based configuration**: Load server configs from env variables
- ✅ **Native smolagents integration**: Uses smolagents' built-in `ToolCollection.from_mcp()`

### Supported Transports

| Transport | Use Case | Example |
|-----------|----------|---------|
| **stdio** | Local MCP servers | Development, testing, trusted servers |
| **streamable-http** | Remote MCP servers | Production, cloud-hosted services |

### Documentation

For comprehensive MCP integration documentation, see [docs/MCP_INTEGRATION.md](docs/MCP_INTEGRATION.md).

### CLI Command (Coming Soon)

```bash
# Add MCP server to your agent
uv run python -m agentic_internet.cli mcp add my-server --path ./server.py

# List MCP servers
uv run python -m agentic_internet.cli mcp list

# Run agent with MCP tools
uv run python -m agentic_internet.cli run "Search and analyze" --with-mcp my-server
```

---

## Installation Note

To use MCP integration, install the MCP SDK:

```bash
pip install mcp

# Or with uv
uv pip install mcp
```
