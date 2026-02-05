"""Tests for custom exception classes."""

from agentic_internet.exceptions import (
    AgenticInternetError,
    APIKeyMissingError,
    BrowserAutomationError,
    CodeExecutionError,
    ConfigurationError,
    MCPError,
    MCPNotAvailableError,
    ModelInitializationError,
    ProviderNotFoundError,
    SearchError,
    ToolExecutionError,
    UnsafeCodeError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        exceptions = [
            ModelInitializationError("m"),
            ProviderNotFoundError("m"),
            APIKeyMissingError("p"),
            SearchError("e", "q"),
            ToolExecutionError("t"),
            CodeExecutionError("r"),
            UnsafeCodeError("p"),
            BrowserAutomationError("c"),
            ConfigurationError("msg"),
            MCPError("msg"),
            MCPNotAvailableError(),
        ]
        for exc in exceptions:
            assert isinstance(exc, AgenticInternetError)
            assert isinstance(exc, Exception)

    def test_model_init_error_attributes(self):
        exc = ModelInitializationError("gpt-4", provider="openai", cause="timeout")
        assert exc.model_id == "gpt-4"
        assert exc.provider == "openai"
        assert exc.cause == "timeout"
        assert "gpt-4" in str(exc)
        assert "openai" in str(exc)

    def test_search_error_attributes(self):
        exc = SearchError("google", "test query", cause="rate limited")
        assert exc.engine == "google"
        assert exc.query == "test query"
        assert "rate limited" in str(exc)

    def test_unsafe_code_error(self):
        exc = UnsafeCodeError("__import__")
        assert isinstance(exc, CodeExecutionError)
        assert isinstance(exc, ToolExecutionError)
        assert "__import__" in str(exc)

    def test_mcp_not_available(self):
        exc = MCPNotAvailableError()
        assert isinstance(exc, MCPError)
        assert "not installed" in str(exc)
