"""
Basic Agent wrapper for models that don't support tool calling or code execution
"""

from __future__ import annotations

from typing import Any

from smolagents import Tool


class BasicAgent:
    """
    A simple agent that directly queries the model without tools or code execution.
    Used as a fallback for models that don't support advanced features.
    """

    def __init__(self, model: Any, tools: list[Tool] | None = None, prompt_templates: Any = None, **kwargs: Any):
        self.model = model
        self.tools = tools or []
        self.prompt_templates = prompt_templates

    def run(self, task: str) -> str:
        """
        Execute a task by directly querying the model.
        For models that don't support tools, we'll describe available tools in the prompt.
        """
        try:
            # Build a prompt that includes tool descriptions if available
            tool_descriptions = ""
            if self.tools:
                tool_descriptions = "\n\nAvailable search capabilities:\n"
                for tool in self.tools:
                    tool_descriptions += f"- {tool.name}: {tool.description}\n"

            # Create the full prompt
            full_prompt = f"""
You are a research assistant with access to various search tools.
{tool_descriptions}

Please provide a comprehensive answer to the following query. If the query would benefit from web search,
indicate what searches would be helpful, but provide the best answer you can based on your knowledge.

Query: {task}

Response:"""

            # Get response from model - handle both direct calls and message-based calls
            try:
                # Try calling with messages format (for LiteLLM models)
                messages = [{"role": "user", "content": full_prompt}]
                response = self.model(messages)
            except (TypeError, AttributeError):
                # Fallback to direct call if messages format doesn't work
                response = self.model(full_prompt)

            # Extract the actual response text
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)

        except Exception as e:
            return f"Error executing task: {e!s}"
