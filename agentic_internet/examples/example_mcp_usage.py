"""
Example usage of MCP integration with AgenticInternet.

This script demonstrates how to use MCP servers with the agentic-internet framework,
supporting both stdio and HTTP transports.

Usage:
    python -m agentic_internet.examples.example_mcp_usage
    
Requirements:
    pip install smolagents mcp fastmcp
"""

import os
from pathlib import Path


def example_stdio_usage():
    """Example: Using MCP tools via stdio transport (local server)."""
    print("\n" + "="*60)
    print("Example 1: STDIO Transport (Local MCP Server)")
    print("="*60 + "\n")
    
    try:
        from agentic_internet.tools.mcp_integration import mcp_tools, is_mcp_available
        
        if not is_mcp_available():
            print("MCP packages not installed. Install with: pip install smolagents mcp")
            return
        
        # Path to the example MCP server
        server_path = Path(__file__).parent / "example_mcp_server.py"
        
        if not server_path.exists():
            print(f"Example server not found at: {server_path}")
            return
        
        print(f"Connecting to MCP server: {server_path}")
        
        # Use context manager to connect and get tools
        with mcp_tools(
            server_path=str(server_path),
            trust_remote_code=True,  # Required to execute MCP tools
        ) as tools:
            print(f"\nLoaded {len(tools)} tools from MCP server:")
            for tool in tools:
                desc = getattr(tool, 'description', 'No description')[:60]
                print(f"  - {tool.name}: {desc}...")
        
        print("\n[Connection closed automatically]")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "-"*60 + "\n")


def example_http_usage():
    """Example: Using MCP tools via HTTP transport (remote server)."""
    print("\n" + "="*60)
    print("Example 2: HTTP Transport (Remote MCP Server)")
    print("="*60 + "\n")
    
    try:
        from agentic_internet.tools.mcp_integration import mcp_tools, is_mcp_available
        
        if not is_mcp_available():
            print("MCP packages not installed.")
            return
        
        # For this example, you need a running MCP server accessible via HTTP
        # Start one with: fastmcp run example_mcp_server.py --transport http --port 8000
        
        server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/")
        
        print(f"Connecting to remote MCP server: {server_url}")
        print("(Start a server with: fastmcp run example_mcp_server.py --transport http --port 8000)")
        
        # Use context manager to connect
        with mcp_tools(
            server_url=server_url,
            trust_remote_code=True,
        ) as tools:
            print(f"\nLoaded {len(tools)} tools from MCP server:")
            for tool in tools:
                print(f"  - {tool.name}")
            
    except Exception as e:
        print(f"\nNote: HTTP example requires a running MCP server.")
        print(f"Start one with: fastmcp run example_mcp_server.py --transport http --port 8000")
        print(f"Error: {e}")
    
    print("\n" + "-"*60 + "\n")


def example_with_agent():
    """Example: Using MCP tools with an InternetAgent."""
    print("\n" + "="*60)
    print("Example 3: Integration with InternetAgent")
    print("="*60 + "\n")
    
    try:
        from agentic_internet.tools.mcp_integration import mcp_tools, is_mcp_available
        from agentic_internet.agents.internet_agent import InternetAgent
        
        if not is_mcp_available():
            print("MCP packages not installed.")
            return
        
        server_path = Path(__file__).parent / "example_mcp_server.py"
        
        if not server_path.exists():
            print(f"Example server not found at: {server_path}")
            return
        
        print("Creating agent with MCP tools...")
        print(f"Server: {server_path}")
        
        # Connect to MCP server and create agent
        with mcp_tools(
            server_path=str(server_path),
            trust_remote_code=True,
        ) as tools:
            print(f"\nLoaded {len(tools)} MCP tools")
            
            # Create agent with MCP tools
            # Note: This creates a new agent with ONLY MCP tools
            # In practice, you might want to combine with default tools
            agent = InternetAgent(
                tools=list(tools),
                verbose=True,
            )
            
            print("\nAgent ready with MCP tools!")
            print("Tools available to agent:")
            for tool in agent.tools:
                print(f"  - {tool.name}")
            
            # Example task (uncomment to run):
            # result = agent.run("Calculate the sum of 42 and 58")
            # print(f"\nResult: {result}")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure agentic_internet is properly installed")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "-"*60 + "\n")


def example_server_manager():
    """Example: Using MCPServerManager to manage multiple servers."""
    print("\n" + "="*60)
    print("Example 4: Managing Multiple MCP Servers")
    print("="*60 + "\n")
    
    try:
        from agentic_internet.tools.mcp_integration import MCPServerManager, is_mcp_available
        
        if not is_mcp_available():
            print("MCP packages not installed.")
            return
        
        server_path = Path(__file__).parent / "example_mcp_server.py"
        
        # Create manager and add servers
        manager = MCPServerManager()
        
        manager.add_server(
            name="example_server",
            server_config=str(server_path),
            transport_type="stdio",
            trust_remote_code=True,
        )
        
        # Add another server (example - would need actual server)
        # manager.add_server(
        #     name="remote_server",
        #     server_config="http://localhost:8000/mcp/",
        #     transport_type="streamable-http",
        #     trust_remote_code=True,
        # )
        
        print(f"Configured servers: {manager.list_servers()}")
        
        # Connect to a specific server
        with manager.connect("example_server") as tools:
            print(f"\nLoaded {len(tools)} tools from example_server:")
            for tool in tools:
                print(f"  - {tool.name}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "-"*60 + "\n")


def example_env_config():
    """Example: Loading MCP configuration from environment variables."""
    print("\n" + "="*60)
    print("Example 5: Environment-Based Configuration")
    print("="*60 + "\n")
    
    try:
        from agentic_internet.tools.mcp_integration import load_mcp_config_from_env, is_mcp_available
        
        if not is_mcp_available():
            print("MCP packages not installed.")
            return
        
        print("Environment variable format:")
        print("  MCP_SERVER_1_TYPE=stdio")
        print("  MCP_SERVER_1_PATH=/path/to/server.py")
        print("  MCP_SERVER_1_NAME=my-server")
        print("  MCP_SERVER_1_TRUST=true")
        print("  MCP_SERVER_1_ENV_API_KEY=secret")
        print()
        print("  MCP_SERVER_2_TYPE=http")
        print("  MCP_SERVER_2_URL=https://api.example.com/mcp")
        print("  MCP_SERVER_2_TRUST=true")
        print()
        
        # Load configuration from environment
        configs = load_mcp_config_from_env()
        
        if configs:
            print(f"Found {len(configs)} MCP server configurations:")
            for config in configs:
                print(f"\n  Name: {config.name}")
                print(f"  Type: {config.transport_type}")
                print(f"  Config: {config.server_config}")
                print(f"  Trust: {config.trust_remote_code}")
        else:
            print("No MCP servers configured in environment.")
            print("Set environment variables as shown above to configure servers.")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "-"*60 + "\n")


def example_cli_usage():
    """Show CLI usage examples."""
    print("\n" + "="*60)
    print("Example 6: CLI Usage")
    print("="*60 + "\n")
    
    print("MCP CLI commands:")
    print()
    print("  # Show MCP integration status")
    print("  uv run python -m agentic_internet.cli mcp info")
    print()
    print("  # List configured MCP servers")
    print("  uv run python -m agentic_internet.cli mcp list")
    print()
    print("  # Test connection to an MCP server")
    print("  uv run python -m agentic_internet.cli mcp test --server ./server.py")
    print()
    print("  # Run a task with MCP tools")
    print("  uv run python -m agentic_internet.cli mcp run \\")
    print("      'Calculate sum of 1 and 2' \\")
    print("      --server ./server.py \\")
    print("      --trust")
    print()
    
    print("-"*60 + "\n")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("AgenticInternet MCP Integration Examples")
    print("="*60)
    
    # Check MCP availability first
    try:
        from agentic_internet.tools.mcp_integration import is_mcp_available
        if not is_mcp_available():
            print("\n[WARNING] MCP packages not installed.")
            print("Install with: pip install smolagents mcp")
            print("For creating servers: pip install fastmcp")
            print("\nShowing conceptual examples only...\n")
    except ImportError:
        print("\n[WARNING] Could not import MCP integration.")
        print("Make sure agentic_internet is properly installed.\n")
    
    # Run examples
    example_stdio_usage()
    example_http_usage()
    example_with_agent()
    example_server_manager()
    example_env_config()
    example_cli_usage()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
