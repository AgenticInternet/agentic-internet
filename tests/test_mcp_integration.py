"""
Tests for MCP integration with AgenticInternet.

These tests verify the MCP integration module works correctly.
Note: Some tests require the 'mcp' and 'smolagents' packages to be installed.
"""

import os

import pytest


class TestMCPAvailability:
    """Test MCP availability detection."""

    def test_is_mcp_available_function(self):
        """Test is_mcp_available function exists and returns bool."""
        from agentic_internet.tools.mcp_integration import is_mcp_available

        result = is_mcp_available()
        assert isinstance(result, bool)

    def test_mcp_available_constant(self):
        """Test MCP_AVAILABLE constant is defined."""
        from agentic_internet.tools.mcp_integration import MCP_AVAILABLE

        assert isinstance(MCP_AVAILABLE, bool)


class TestMCPImports:
    """Test that MCP integration modules can be imported."""

    def test_imports_from_module(self):
        """Test importing main classes from module."""
        from agentic_internet.tools.mcp_integration import (
            MCPServerConfig,
            MCPServerManager,
            MCPToolIntegration,
            is_mcp_available,
            load_mcp_config_from_env,
            mcp_tools,
        )

        # These should all be importable even if MCP is not installed
        assert MCPToolIntegration is not None or not is_mcp_available()
        assert MCPServerConfig is not None or not is_mcp_available()
        assert MCPServerManager is not None or not is_mcp_available()
        assert mcp_tools is not None or not is_mcp_available()
        assert load_mcp_config_from_env is not None

    def test_imports_from_tools_init(self):
        """Test importing from tools __init__.py."""
        from agentic_internet.tools import (
            MCP_AVAILABLE,
            is_mcp_available,
        )

        assert isinstance(MCP_AVAILABLE, bool)
        assert callable(is_mcp_available)


class TestMCPServerConfig:
    """Test MCPServerConfig class."""

    @pytest.fixture
    def skip_if_no_mcp(self):
        """Skip test if MCP is not available."""
        from agentic_internet.tools.mcp_integration import is_mcp_available
        if not is_mcp_available():
            pytest.skip("MCP packages not installed")

    def test_server_config_creation(self, skip_if_no_mcp):
        """Test creating MCPServerConfig."""
        from agentic_internet.tools.mcp_integration import MCPServerConfig

        config = MCPServerConfig(
            name="test_server",
            server_config="./server.py",
            transport_type="stdio",
            trust_remote_code=True,
        )

        assert config.name == "test_server"
        assert config.server_config == "./server.py"
        assert config.transport_type == "stdio"
        assert config.trust_remote_code is True

    def test_server_config_to_dict(self, skip_if_no_mcp):
        """Test MCPServerConfig.to_dict()."""
        from agentic_internet.tools.mcp_integration import MCPServerConfig

        config = MCPServerConfig(
            name="test",
            server_config="./test.py",
            transport_type="stdio",
        )

        data = config.to_dict()
        assert data["name"] == "test"
        assert data["server_config"] == "./test.py"
        assert data["transport_type"] == "stdio"

    def test_server_config_from_dict(self, skip_if_no_mcp):
        """Test MCPServerConfig.from_dict()."""
        from agentic_internet.tools.mcp_integration import MCPServerConfig

        data = {
            "name": "test",
            "server_config": "./test.py",
            "transport_type": "stdio",
            "trust_remote_code": True,
        }

        config = MCPServerConfig.from_dict(data)
        assert config.name == "test"
        assert config.server_config == "./test.py"
        assert config.trust_remote_code is True


class TestMCPServerManager:
    """Test MCPServerManager class."""

    @pytest.fixture
    def skip_if_no_mcp(self):
        """Skip test if MCP is not available."""
        from agentic_internet.tools.mcp_integration import is_mcp_available
        if not is_mcp_available():
            pytest.skip("MCP packages not installed")

    def test_manager_creation(self, skip_if_no_mcp):
        """Test MCPServerManager can be created."""
        from agentic_internet.tools.mcp_integration import MCPServerManager

        manager = MCPServerManager()
        assert manager is not None
        assert len(manager.servers) == 0

    def test_manager_add_server(self, skip_if_no_mcp):
        """Test adding servers to MCPServerManager."""
        from agentic_internet.tools.mcp_integration import MCPServerManager

        manager = MCPServerManager()
        manager.add_server(
            name="test_server",
            server_config="./test_server.py",
            transport_type="stdio",
            trust_remote_code=True,
        )

        assert len(manager.servers) == 1
        assert "test_server" in manager.servers
        assert manager.servers["test_server"].trust_remote_code is True

    def test_manager_list_servers(self, skip_if_no_mcp):
        """Test listing servers."""
        from agentic_internet.tools.mcp_integration import MCPServerManager

        manager = MCPServerManager()
        manager.add_server("server1", "./s1.py")
        manager.add_server("server2", "./s2.py")

        servers = manager.list_servers()
        assert len(servers) == 2
        assert "server1" in servers
        assert "server2" in servers

    def test_manager_remove_server(self, skip_if_no_mcp):
        """Test removing servers."""
        from agentic_internet.tools.mcp_integration import MCPServerManager

        manager = MCPServerManager()
        manager.add_server("test", "./test.py")

        result = manager.remove_server("test")
        assert result is True
        assert len(manager.servers) == 0

        # Removing non-existent server returns False
        result = manager.remove_server("nonexistent")
        assert result is False


class TestMCPToolIntegration:
    """Test MCPToolIntegration class."""

    @pytest.fixture
    def skip_if_no_mcp(self):
        """Skip test if MCP is not available."""
        from agentic_internet.tools.mcp_integration import is_mcp_available
        if not is_mcp_available():
            pytest.skip("MCP packages not installed")

    def test_integration_stdio_config(self, skip_if_no_mcp):
        """Test MCPToolIntegration with stdio configuration."""
        from agentic_internet.tools.mcp_integration import MCPToolIntegration

        integration = MCPToolIntegration(
            server_config="./test_server.py",
            transport_type="stdio",
            env={"TEST_VAR": "value"},
            trust_remote_code=True,
        )

        assert integration.server_config == "./test_server.py"
        assert integration.transport_type == "stdio"
        assert integration.env["TEST_VAR"] == "value"
        assert integration.trust_remote_code is True

    def test_integration_http_config(self, skip_if_no_mcp):
        """Test MCPToolIntegration with HTTP configuration."""
        from agentic_internet.tools.mcp_integration import MCPToolIntegration

        integration = MCPToolIntegration(
            server_config="http://localhost:8000/mcp",
            transport_type="streamable-http",
            trust_remote_code=True,
        )

        assert integration.server_config == "http://localhost:8000/mcp"
        assert integration.transport_type == "streamable-http"

    def test_integration_get_server_parameters_stdio(self, skip_if_no_mcp):
        """Test getting server parameters for stdio transport."""
        from agentic_internet.tools.mcp_integration import MCPToolIntegration

        integration = MCPToolIntegration(
            server_config="./test.py",
            transport_type="stdio",
            trust_remote_code=True,
        )

        params = integration.get_server_parameters()
        # Should be StdioServerParameters
        assert hasattr(params, 'command')
        assert hasattr(params, 'args')

    def test_integration_get_server_parameters_http(self, skip_if_no_mcp):
        """Test getting server parameters for HTTP transport."""
        from agentic_internet.tools.mcp_integration import MCPToolIntegration

        integration = MCPToolIntegration(
            server_config="http://localhost:8000/mcp",
            transport_type="streamable-http",
        )

        params = integration.get_server_parameters()
        # Should be dict for HTTP
        assert isinstance(params, dict)
        assert params["url"] == "http://localhost:8000/mcp"
        assert params["transport"] == "streamable-http"


class TestEnvironmentConfig:
    """Test environment-based configuration loading."""

    def test_load_empty_env_config(self):
        """Test loading config when no env vars are set."""
        from agentic_internet.tools.mcp_integration import load_mcp_config_from_env

        # Temporarily clear any MCP env vars
        original_env = {}
        for key in list(os.environ.keys()):
            if key.startswith("MCP_SERVER_"):
                original_env[key] = os.environ.pop(key)

        try:
            configs = load_mcp_config_from_env()
            assert isinstance(configs, list)
            assert len(configs) == 0
        finally:
            # Restore original env
            os.environ.update(original_env)

    @pytest.fixture
    def skip_if_no_mcp(self):
        """Skip test if MCP is not available."""
        from agentic_internet.tools.mcp_integration import is_mcp_available
        if not is_mcp_available():
            pytest.skip("MCP packages not installed")

    def test_load_stdio_config_from_env(self, skip_if_no_mcp):
        """Test loading stdio server config from env."""
        from agentic_internet.tools.mcp_integration import load_mcp_config_from_env

        # Set test env vars
        os.environ["MCP_SERVER_1_TYPE"] = "stdio"
        os.environ["MCP_SERVER_1_PATH"] = "/test/server.py"
        os.environ["MCP_SERVER_1_NAME"] = "test-server"
        os.environ["MCP_SERVER_1_TRUST"] = "true"

        try:
            configs = load_mcp_config_from_env()
            assert len(configs) >= 1

            config = configs[0]
            assert config.name == "test-server"
            assert config.server_config == "/test/server.py"
            assert config.transport_type == "stdio"
            assert config.trust_remote_code is True
        finally:
            # Clean up
            for key in ["MCP_SERVER_1_TYPE", "MCP_SERVER_1_PATH",
                       "MCP_SERVER_1_NAME", "MCP_SERVER_1_TRUST"]:
                os.environ.pop(key, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
