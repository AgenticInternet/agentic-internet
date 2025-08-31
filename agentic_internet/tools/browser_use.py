"""Browser Use tool for advanced web automation and scraping."""

import os
import asyncio
from typing import Optional, Dict, Any, List
from smolagents import Tool
import logging
from pydantic import BaseModel
from browser_use_sdk import BrowserUse, AsyncBrowserUse

logger = logging.getLogger(__name__)


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
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Browser Use tool with API key."""
        super().__init__()
        self.api_key = api_key or os.environ.get("BROWSER_USE_API_KEY")
        
        if not self.api_key:
            logger.warning("Browser Use API key not found. Tool will have limited functionality.")
            self.client = None
        else:
            self.client = BrowserUse(api_key=self.api_key)
    
    def forward(self, task: str, structured_output: bool = False) -> str:
        """Execute a browser automation task."""
        if not self.client:
            return "Browser Use API key not configured. Please set BROWSER_USE_API_KEY environment variable."
        
        try:
            # Run the task
            result = self.client.tasks.run(task=task)
            
            if result.done_output:
                return result.done_output
            else:
                return f"Task completed but no output was returned. Status: {result.status if hasattr(result, 'status') else 'unknown'}"
                
        except Exception as e:
            error_msg = f"Browser automation failed: {str(e)}"
            logger.error(error_msg)
            return error_msg


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
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Async Browser Use tool."""
        super().__init__()
        self.api_key = api_key or os.environ.get("BROWSER_USE_API_KEY")
        
        if not self.api_key:
            logger.warning("Browser Use API key not found. Tool will have limited functionality.")
            self.client = None
        else:
            self.client = AsyncBrowserUse(api_key=self.api_key)
    
    def forward(self, task: str, stream: bool = False) -> str:
        """Execute an async browser automation task."""
        if not self.client:
            return "Browser Use API key not configured. Please set BROWSER_USE_API_KEY environment variable."
        
        try:
            # Run async task in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                if stream:
                    result = loop.run_until_complete(self._run_with_stream(task))
                else:
                    result = loop.run_until_complete(self._run_simple(task))
                return result
            finally:
                loop.close()
                
        except Exception as e:
            error_msg = f"Async browser automation failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _run_simple(self, task: str) -> str:
        """Run a simple async task."""
        result = await self.client.tasks.run(task=task)
        
        if result.done_output:
            return result.done_output
        else:
            return f"Task completed. Status: {result.status if hasattr(result, 'status') else 'completed'}"
    
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
                    return f"Task completed.\nSteps performed:\n" + "\n".join(updates)
        
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
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Structured Browser Use tool."""
        super().__init__()
        self.api_key = api_key or os.environ.get("BROWSER_USE_API_KEY")
        
        if not self.api_key:
            logger.warning("Browser Use API key not found. Tool will have limited functionality.")
            self.client = None
        else:
            self.client = AsyncBrowserUse(api_key=self.api_key)
    
    def forward(self, task: str, schema: Optional[str] = None) -> str:
        """Execute browser automation with structured output."""
        if not self.client:
            return "Browser Use API key not configured. Please set BROWSER_USE_API_KEY environment variable."
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(self._extract_structured_data(task, schema))
                return result
            finally:
                loop.close()
                
        except Exception as e:
            error_msg = f"Structured browser extraction failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _extract_structured_data(self, task: str, schema: Optional[str] = None) -> str:
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
    author: Optional[str] = None
    date: Optional[str] = None
    content: str
    url: str


class ProductInfo(BaseModel):
    """Model for e-commerce product extraction."""
    name: str
    price: str
    availability: Optional[str] = None
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    description: Optional[str] = None
    image_url: Optional[str] = None


class ContactInfo(BaseModel):
    """Model for contact information extraction."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    website: Optional[str] = None


class SearchResults(BaseModel):
    """Model for search results extraction."""
    results: List[Dict[str, str]]
    total_results: Optional[int] = None
    next_page_url: Optional[str] = None
