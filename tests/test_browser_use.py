"""Tests for browser use tools."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_internet.tools.browser_use import (
    AsyncBrowserUseTool,
    BrowserUseTool,
    StructuredBrowserUseTool,
)


class TestBrowserUseTool:
    def test_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            tool = BrowserUseTool(api_key=None)
            result = tool.forward("test task")
            assert "not available" in result.lower() or "not configured" in result.lower()

    @patch("agentic_internet.tools.browser_use.HAS_BROWSER_USE", True)
    @patch("agentic_internet.tools.browser_use.BrowserUse")
    def test_successful_task(self, mock_browser_cls):
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.result = "Task completed successfully"
        mock_client.runs.create.return_value = SimpleNamespace(id="run-1")
        mock_client.runs.wait_for_completion.return_value = mock_result
        mock_browser_cls.return_value = mock_client

        tool = BrowserUseTool(api_key="test-key")
        result = tool.forward("navigate to example.com")
        assert "Task completed successfully" in result

    @patch("agentic_internet.tools.browser_use.HAS_BROWSER_USE", True)
    @patch("agentic_internet.tools.browser_use.BrowserUse")
    def test_no_output(self, mock_browser_cls):
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.result = None
        mock_result.error = None
        mock_result.status.value = "completed"
        mock_client.runs.create.return_value = SimpleNamespace(id="run-1")
        mock_client.runs.wait_for_completion.return_value = mock_result
        mock_browser_cls.return_value = mock_client

        tool = BrowserUseTool(api_key="test-key")
        result = tool.forward("task")
        assert "completed" in result.lower()

    @patch("agentic_internet.tools.browser_use.HAS_BROWSER_USE", True)
    @patch("agentic_internet.tools.browser_use.BrowserUse")
    def test_exception_handling(self, mock_browser_cls):
        mock_client = MagicMock()
        mock_client.runs.create.side_effect = RuntimeError("connection failed")
        mock_browser_cls.return_value = mock_client

        tool = BrowserUseTool(api_key="test-key")
        result = tool.forward("task")
        assert "failed" in result.lower()


class TestAsyncBrowserUseTool:
    def test_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            tool = AsyncBrowserUseTool(api_key=None)
            result = tool.forward("test")
            assert "not available" in result.lower() or "not configured" in result.lower()

    @patch("agentic_internet.tools.browser_use.HAS_BROWSER_USE", True)
    @patch("agentic_internet.tools.browser_use.AsyncBrowserUse")
    def test_successful_task(self, mock_browser_cls):
        mock_client = MagicMock()
        mock_client.runs.create = AsyncMock(return_value=SimpleNamespace(id="run-1"))
        mock_client.runs.wait_for_completion = AsyncMock(
            return_value=SimpleNamespace(result="Async task completed", error=None)
        )
        mock_browser_cls.return_value = mock_client

        tool = AsyncBrowserUseTool(api_key="test-key")
        assert tool.forward("navigate to example.com") == "Async task completed"

    @patch("agentic_internet.tools.browser_use.HAS_BROWSER_USE", True)
    @patch("agentic_internet.tools.browser_use.AsyncBrowserUse")
    def test_stream_uses_v4_run_lifecycle(self, mock_browser_cls):
        mock_client = MagicMock()
        mock_client.runs.create = AsyncMock(return_value=SimpleNamespace(id="run-1"))
        mock_client.runs.wait_for_completion = AsyncMock(
            return_value=SimpleNamespace(result="Stream task completed", error=None)
        )
        mock_browser_cls.return_value = mock_client

        tool = AsyncBrowserUseTool(api_key="test-key")
        assert tool.forward("navigate to example.com", stream=True) == "Stream task completed"


class TestStructuredBrowserUseTool:
    def test_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            tool = StructuredBrowserUseTool(api_key=None)
            result = tool.forward("test")
            assert "not available" in result.lower() or "not configured" in result.lower()
