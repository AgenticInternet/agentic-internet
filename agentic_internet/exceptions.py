"""Custom exceptions for the Agentic Internet application."""


class AgenticInternetError(Exception):
    """Base exception for all agentic-internet errors."""


class ModelInitializationError(AgenticInternetError):
    """Raised when a model fails to initialize."""

    def __init__(self, model_id: str, provider: str | None = None, cause: str | None = None):
        self.model_id = model_id
        self.provider = provider
        self.cause = cause
        detail = f"Failed to initialize model '{model_id}'"
        if provider:
            detail += f" (provider: {provider})"
        if cause:
            detail += f": {cause}"
        super().__init__(detail)


class ProviderNotFoundError(AgenticInternetError):
    """Raised when no provider can be determined for a model."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Could not determine provider for model: {model_id}")


class APIKeyMissingError(AgenticInternetError):
    """Raised when a required API key is not configured."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(
            f"No API key found for provider '{provider}'. "
            f"Set the appropriate environment variable in your .env file."
        )


class SearchError(AgenticInternetError):
    """Raised when a search operation fails."""

    def __init__(self, engine: str, query: str, cause: str | None = None):
        self.engine = engine
        self.query = query
        self.cause = cause
        detail = f"Search failed on {engine} for query '{query}'"
        if cause:
            detail += f": {cause}"
        super().__init__(detail)


class ToolExecutionError(AgenticInternetError):
    """Raised when a tool fails to execute."""

    def __init__(self, tool_name: str, cause: str | None = None):
        self.tool_name = tool_name
        self.cause = cause
        detail = f"Tool '{tool_name}' execution failed"
        if cause:
            detail += f": {cause}"
        super().__init__(detail)


class CodeExecutionError(ToolExecutionError):
    """Raised when code execution is blocked or fails."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(tool_name="python_executor", cause=reason)


class UnsafeCodeError(CodeExecutionError):
    """Raised when submitted code contains unsafe operations."""

    def __init__(self, pattern: str):
        self.pattern = pattern
        super().__init__(reason=f"Code contains potentially unsafe operation: {pattern}")


class BrowserAutomationError(ToolExecutionError):
    """Raised when browser automation fails."""

    def __init__(self, cause: str):
        super().__init__(tool_name="browser_use", cause=cause)


class ConfigurationError(AgenticInternetError):
    """Raised for configuration-related issues."""


class MCPError(AgenticInternetError):
    """Raised for MCP integration errors."""

    def __init__(self, message: str):
        super().__init__(f"MCP error: {message}")


class MCPNotAvailableError(MCPError):
    """Raised when MCP packages are not installed."""

    def __init__(self) -> None:
        super().__init__(
            "MCP packages not installed. Install with: pip install smolagents mcp"
        )
