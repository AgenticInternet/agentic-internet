"""Browser Use tool for advanced web automation and scraping."""

import asyncio
import logging
import os

from pydantic import BaseModel
from smolagents import Tool

logger = logging.getLogger(__name__)

try:
    from browser_use_sdk import AsyncBrowserUse, BrowserUse
    HAS_BROWSER_USE = True
except ImportError:
    HAS_BROWSER_USE = False
    BrowserUse = None  # type: ignore[assignment,misc]
    AsyncBrowserUse = None  # type: ignore[assignment,misc]


class BrowserUseTool(Tool):
    """
    Tool for web automation using Browser Use Cloud API.
    Enables agents to interact with websites, fill forms, and extract structured data.
    """

    name = "browser_use"
    description = """Advanced browser automation tool for interacting with websites, filling forms,
    clicking buttons, and extracting structured data. Use this when you need to interact with
    dynamic websites or perform complex web automation tasks."""

    inputs = {
        "task": {
            "type": "string",
            "description": "Description of the task to perform in the browser"
        },
        "structured_output": {
            "type": "boolean",
            "description": "Whether to return structured output (optional)",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None):
        """Initialize the Browser Use tool with API key."""
        super().__init__()
        self.api_key = api_key or os.environ.get("BROWSER_USE_API_KEY")

        if not self.api_key or not HAS_BROWSER_USE:
            if not HAS_BROWSER_USE:
                logger.warning("browser-use-sdk not installed. Install with: pip install browser-use-sdk")
            else:
                logger.warning("Browser Use API key not found. Tool will have limited functionality.")
            self.client = None
        else:
            self.client = BrowserUse(api_key=self.api_key)

    def forward(self, task: str, structured_output: bool = False) -> str:
        """Execute a browser automation task."""
        if not self.client:
            return "Browser Use not available. Check API key and browser-use-sdk installation."

        try:
            result = self.client.tasks.run(task=task)
            if result.done_output:
                return result.done_output
            return f"Task completed but no output was returned. Status: {getattr(result, 'status', 'unknown')}"
        except Exception as e:
            logger.error("Browser automation failed: %s", e)
            return f"Browser automation failed: {e}"


class AsyncBrowserUseTool(Tool):
    """
    Async version of Browser Use tool for web automation.
    """

    name = "async_browser_use"
    description = """Async browser automation tool for high-performance web interactions.
    Use this for complex scraping tasks that require parallel execution or streaming updates."""

    inputs = {
        "task": {
            "type": "string",
            "description": "Description of the task to perform in the browser"
        },
        "stream": {
            "type": "boolean",
            "description": "Whether to stream updates (optional)",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None):
        """Initialize the Async Browser Use tool."""
        super().__init__()
        self.api_key = api_key or os.environ.get("BROWSER_USE_API_KEY")

        if not self.api_key or not HAS_BROWSER_USE:
            if not HAS_BROWSER_USE:
                logger.warning("browser-use-sdk not installed.")
            else:
                logger.warning("Browser Use API key not found. Tool will have limited functionality.")
            self.client = None
        else:
            self.client = AsyncBrowserUse(api_key=self.api_key)

    def forward(self, task: str, stream: bool = False) -> str:
        """Execute an async browser automation task."""
        if not self.client:
            return "Browser Use not available. Check API key and browser-use-sdk installation."

        try:
            coro = self._run_with_stream(task) if stream else self._run_simple(task)
            return asyncio.run(coro)
        except Exception as e:
            logger.error("Async browser automation failed: %s", e)
            return f"Async browser automation failed: {e}"

    async def _run_simple(self, task: str) -> str:
        """Run a simple async task."""
        result = await self.client.tasks.run(task=task)
        if result.done_output:
            return result.done_output
        return f"Task completed. Status: {getattr(result, 'status', 'completed')}"

    async def _run_with_stream(self, task: str) -> str:
        """Run a task with streaming updates."""
        # Create the task
        created_task = await self.client.tasks.create(task=task)

        updates = []
        # Stream updates
        async for update in self.client.tasks.stream(created_task.id):
            if len(update.steps) > 0:
                last_step = update.steps[-1]
                updates.append(f"Step: {last_step.url if hasattr(last_step, 'url') else 'processing'} - {last_step.next_goal if hasattr(last_step, 'next_goal') else 'working'}")

            if update.status == "finished":
                if update.done_output:
                    return update.done_output
                else:
                    return "Task completed.\nSteps performed:\n" + "\n".join(updates)

        return "Task stream ended without completion."


class StructuredBrowserUseTool(Tool):
    """
    Browser Use tool with structured output support using Pydantic models.
    """

    name = "structured_browser_use"
    description = """Browser automation with structured data extraction.
    Use this when you need to extract specific structured information from websites."""

    inputs = {
        "task": {
            "type": "string",
            "description": "Description of the data extraction task"
        },
        "schema": {
            "type": "string",
            "description": "JSON schema describing the expected output structure",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None):
        """Initialize the Structured Browser Use tool."""
        super().__init__()
        self.api_key = api_key or os.environ.get("BROWSER_USE_API_KEY")

        if not self.api_key or not HAS_BROWSER_USE:
            if not HAS_BROWSER_USE:
                logger.warning("browser-use-sdk not installed.")
            else:
                logger.warning("Browser Use API key not found. Tool will have limited functionality.")
            self.client = None
        else:
            self.client = AsyncBrowserUse(api_key=self.api_key)

    def forward(self, task: str, schema: str | None = None) -> str:
        """Execute browser automation with structured output."""
        if not self.client:
            return "Browser Use not available. Check API key and browser-use-sdk installation."

        try:
            return asyncio.run(self._extract_structured_data(task, schema))
        except Exception as e:
            logger.error("Structured browser extraction failed: %s", e)
            return f"Structured browser extraction failed: {e}"

    async def _extract_structured_data(self, task: str, schema: str | None = None) -> str:
        """Extract structured data from web pages."""
        # For now, we'll use the standard run method
        # In a real implementation, you'd parse the schema and use it
        result = await self.client.tasks.run(task=task)

        if result.done_output:
            return result.done_output
        else:
            return "No structured data extracted."


# Example Pydantic models for common extraction tasks
class WebArticle(BaseModel):
    """Model for web article extraction."""
    title: str
    author: str | None = None
    date: str | None = None
    content: str
    url: str


class ProductInfo(BaseModel):
    """Model for e-commerce product extraction."""
    name: str
    price: str
    availability: str | None = None
    rating: float | None = None
    reviews_count: int | None = None
    description: str | None = None
    image_url: str | None = None


class ContactInfo(BaseModel):
    """Model for contact information extraction."""
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    address: str | None = None
    website: str | None = None


class SearchResults(BaseModel):
    """Model for search results extraction."""
    results: list[dict[str, str]]
    total_results: int | None = None
    next_page_url: str | None = None
