"""Main internet agent implementation using smolagents."""

import logging
from typing import Any, Literal

import pandas as pd
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from smolagents import (
    CodeAgent,
    Tool,
    ToolCallingAgent,
    load_tool,
)

from ..config.settings import settings
from ..exceptions import ModelInitializationError
from ..tools.browser_use import AsyncBrowserUseTool, BrowserUseTool, StructuredBrowserUseTool
from ..tools.code_execution import DataAnalysisTool, PythonExecutorTool
from ..tools.web_search import NewsSearchTool, WebScraperTool, WebSearchTool
from ..utils.model_utils import initialize_model

console = Console()
logger = logging.getLogger(__name__)


class InternetAgent:
    """
    Advanced internet agent capable of web search, scraping, and intelligent reasoning.
    """

    def __init__(
        self,
        model_id: str | None = None,
        tools: list[Tool] | None = None,
        verbose: bool = True,
        max_iterations: int = 10,
        planning_enabled: bool = True,
        agent_type: Literal["tool_calling", "code"] = "tool_calling",
        additional_authorized_imports: list[str] | None = None
    ):
        """
        Initialize the Internet Agent.

        Args:
            model_id: The model to use for the agent
            tools: List of tools to make available to the agent
            verbose: Whether to print verbose output
            max_iterations: Maximum number of iterations for the agent
            planning_enabled: Whether to enable multi-step planning
            agent_type: Type of agent to create ("tool_calling" or "code")
            additional_authorized_imports: Additional imports for CodeAgent
        """
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.planning_enabled = planning_enabled
        self.agent_type = agent_type
        self.additional_authorized_imports = additional_authorized_imports or []

        # Initialize model
        self.model = self._initialize_model(model_id)

        # Initialize tools
        self.tools = tools or self._get_default_tools()

        # Create the appropriate agent type
        self.agent = self._create_agent()

        if self.verbose:
            console.print(Panel.fit(
                f"[bold green]Internet Agent Initialized[/bold green]\n"
                f"Model: {model_id or settings.model.name}\n"
                f"Agent Type: {self.agent_type}\n"
                f"Tools: {[tool.name for tool in self.tools]}\n"
                f"Max Iterations: {self.max_iterations}",
                title="Agent Ready"
            ))

    def _initialize_model(self, model_id: str | None = None) -> Any:
        """Initialize the LLM model using centralized model initialization."""
        model = initialize_model(model_id, verbose=self.verbose)
        if not model:
            raise ModelInitializationError(
                model_id=model_id or settings.model.name,
                cause="No model could be initialized. Check your API keys.",
            )
        return model

    def _create_agent(self):
        """Create the appropriate agent based on agent_type."""
        if self.agent_type == "code":
            # CodeAgent can execute Python code and use tools
            default_imports = [
                "pandas", "numpy", "json", "csv", "re",
                "datetime", "time", "requests", "urllib",
                "math", "statistics", "collections"
            ]
            return CodeAgent(
                tools=self.tools,
                model=self.model,
                max_steps=self.max_iterations,
                additional_authorized_imports=default_imports + self.additional_authorized_imports
            )
        else:
            # Default to ToolCallingAgent
            return ToolCallingAgent(
                tools=self.tools,
                model=self.model
            )

    def _get_default_tools(self) -> list[Tool]:
        """Get the default set of tools for the agent."""
        tools = []

        # Add web search tools
        if settings.tools.web_search_enabled:
            tools.extend([
                WebSearchTool(),
                WebScraperTool(),
                NewsSearchTool()
            ])

        # Add Browser Use tools if API key is available
        if settings.tools.browser_enabled and settings.browser_use_api_key:
            tools.extend([
                BrowserUseTool(api_key=settings.browser_use_api_key),
                AsyncBrowserUseTool(api_key=settings.browser_use_api_key),
                StructuredBrowserUseTool(api_key=settings.browser_use_api_key)
            ])

        # Add code execution tools
        if settings.tools.code_execution_enabled:
            tools.extend([
                PythonExecutorTool(),
                DataAnalysisTool()
            ])

        # Try to load built-in smolagents tools
        try:
            # Calculator tool
            tools.append(load_tool("calculator"))
        except Exception as e:
            logger.debug(f"Could not load calculator tool: {e}")

        return tools

    def run(self, task: str, *, show_result: bool = True, **kwargs) -> str:
        """
        Run the agent on a specific task.

        Args:
            task: The task/query to execute
            **kwargs: Additional arguments to pass to the agent

        Returns:
            The agent's response
        """
        if self.verbose:
            console.print(Panel.fit(
                f"[bold blue]Task:[/bold blue] {task}",
                title="Executing"
            ))

        try:
            # Run the agent
            result = self.agent.run(task, **kwargs)

            if self.verbose and show_result:
                console.print(Panel.fit(
                    Markdown(str(result)),
                    title="[bold green]Result[/bold green]"
                ))

            return result

        except Exception as e:
            logger.error("Error executing task: %s", e, exc_info=True)
            error_msg = f"Error executing task: {e}"
            if self.verbose:
                console.print(f"[bold red]{error_msg}[/bold red]")
            return error_msg

    def chat(self) -> None:
        """
        Start an interactive chat session with the agent.
        """
        console.print(Panel.fit(
            "[bold cyan]Interactive Chat Mode[/bold cyan]\n"
            "Type 'exit' or 'quit' to end the session.\n"
            "Type 'help' for available commands.",
            title="Chat Started"
        ))

        chat_history = []

        while True:
            try:
                # Get user input
                user_input = console.input("\n[bold yellow]You:[/bold yellow] ")

                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    console.print("[bold cyan]Goodbye![/bold cyan]")
                    break

                # Check for help command
                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                # Check for tool list command
                if user_input.lower() == 'tools':
                    self._show_tools()
                    continue

                # Process the input
                response = self.run(user_input)
                chat_history.append({"user": user_input, "agent": response})

            except KeyboardInterrupt:
                console.print("\n[bold cyan]Chat interrupted. Goodbye![/bold cyan]")
                break
            except Exception as e:
                console.print(f"[bold red]Error: {e!s}[/bold red]")

    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
        **Available Commands:**
        - `exit`, `quit`, `bye` - End the chat session
        - `help` - Show this help message
        - `tools` - List available tools

        **Example Tasks:**
        - "Search for the latest news about AI"
        - "What's the weather in San Francisco?"
        - "Scrape the content from https://example.com"
        - "Calculate the compound interest for $1000 at 5% for 10 years"
        - "Analyze this data: [1,2,3,4,5]"
        """
        console.print(Panel(Markdown(help_text), title="Help"))

    def _show_tools(self) -> None:
        """Show available tools."""
        tools_info = []
        for tool in self.tools:
            tools_info.append(f"**{tool.name}**: {tool.description[:100]}...")

        tools_text = "\n".join(tools_info)
        console.print(Panel(Markdown(tools_text), title="Available Tools"))


class ResearchAgent(InternetAgent):
    """
    Specialized agent for conducting research tasks.
    """

    def __init__(self, **kwargs):
        """Initialize the research agent with research-specific tools."""
        super().__init__(**kwargs)
        self.research_history = []

    def research(self, topic: str, depth: str = "moderate") -> dict[str, Any]:
        """
        Conduct research on a specific topic.

        Args:
            topic: The research topic
            depth: Research depth - "quick", "moderate", or "deep"

        Returns:
            Research findings dictionary
        """
        depth_prompts = {
            "quick": f"Quickly search for basic information about {topic}. Provide a brief summary.",
            "moderate": f"Research {topic}. Search for information, find recent news, and provide a comprehensive summary with sources.",
            "deep": f"Conduct deep research on {topic}. Search multiple sources, analyze different perspectives, check recent developments, and provide a detailed analysis with citations."
        }

        prompt = depth_prompts.get(depth, depth_prompts["moderate"])

        if self.verbose:
            console.print(f"[bold blue]Researching:[/bold blue] {topic} (Depth: {depth})")

        result = self.run(prompt, show_result=False)

        research_entry = {
            "topic": topic,
            "depth": depth,
            "findings": result,
            "timestamp": pd.Timestamp.now().isoformat()
        }

        self.research_history.append(research_entry)

        return research_entry

    def get_research_history(self) -> list[dict[str, Any]]:
        """Get the research history."""
        return self.research_history
