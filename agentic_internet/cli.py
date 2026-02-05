"""Command-line interface for the Agentic Internet application."""

import asyncio
import json
import logging
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .agents.internet_agent import InternetAgent, ResearchAgent
from .agents.multi_model_serpapi import MultiModelSerpAPISystem
from .config.settings import settings

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="agentic-internet",
    help="Advanced AI agent for intelligent internet interactions",
    add_completion=False
)
console = Console()


@app.command()
def chat(
    model: str | None = typer.Option(
        None,
        "--model", "-m",
        help="Model to use (e.g., claude-opus-4.5, gpt-5.2, gemini-3-flash, grok-4.1-fast, deepseek-v3.2, qwen3-coder, devstral-2, sonar-pro)"
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet", "-v/-q",
        help="Enable verbose output"
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations", "-i",
        help="Maximum iterations for the agent"
    )
):
    """Start an interactive chat session with the agent."""
    try:
        agent = InternetAgent(
            model_id=model,
            verbose=verbose,
            max_iterations=max_iterations
        )
        agent.chat()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e!s}")
        raise typer.Exit(1) from e


@app.command()
def run(
    task: str = typer.Argument(..., help="The task to execute"),
    model: str | None = typer.Option(
        None,
        "--model", "-m",
        help="Model to use (e.g., claude-opus-4.5, gpt-5.2, gemini-3-flash, grok-4.1-fast, deepseek-v3.2, qwen3-coder, devstral-2, sonar-pro)"
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet", "-v/-q",
        help="Enable verbose output"
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations", "-i",
        help="Maximum iterations for the agent"
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Save output to file"
    )
):
    """Run a single task with the agent."""
    try:
        agent = InternetAgent(
            model_id=model,
            verbose=verbose,
            max_iterations=max_iterations
        )

        result = agent.run(task)

        if output:
            output.write_text(str(result))
            console.print(f"[green]Output saved to {output}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e!s}")
        raise typer.Exit(1) from e


@app.command()
def research(
    topic: str = typer.Argument(..., help="The topic to research"),
    depth: str = typer.Option(
        "moderate",
        "--depth", "-d",
        help="Research depth: quick, moderate, or deep"
    ),
    model: str | None = typer.Option(
        None,
        "--model", "-m",
        help="Model to use (e.g., claude-opus-4.5, gpt-5.2, gemini-3-flash, grok-4.1-fast, deepseek-v3.2, qwen3-coder, devstral-2, sonar-pro)"
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Save research results to file"
    ),
    format: str = typer.Option(
        "markdown",
        "--format", "-f",
        help="Output format: markdown or json"
    )
):
    """Conduct research on a specific topic."""
    if depth not in ["quick", "moderate", "deep"]:
        console.print("[bold red]Error:[/bold red] Depth must be 'quick', 'moderate', or 'deep'")
        raise typer.Exit(1)

    try:
        agent = ResearchAgent(model_id=model)
        result = agent.research(topic, depth)

        if output:
            if format == "json":
                output.write_text(json.dumps(result, indent=2))
            else:
                # Create markdown output
                md_content = f"# Research: {result['topic']}\n\n"
                md_content += f"**Depth:** {result['depth']}\n"
                md_content += f"**Timestamp:** {result['timestamp']}\n\n"
                md_content += f"## Findings\n\n{result['findings']}"
                output.write_text(md_content)

            console.print(f"[green]Research saved to {output}[/green]")
        else:
            console.print(Panel(
                Markdown(result['findings']),
                title=f"Research: {topic}"
            ))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e!s}")
        raise typer.Exit(1) from e


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show", "-s",
        help="Show current configuration"
    ),
    set_key: str | None = typer.Option(
        None,
        "--set", "-k",
        help="Set a configuration value (format: key=value)"
    )
):
    """Manage configuration settings."""
    if show:
        # Display current configuration
        config_dict = settings.model_dump()
        # Convert Path objects to strings for JSON serialization
        if 'cache_dir' in config_dict and hasattr(config_dict['cache_dir'], '__fspath__'):
            config_dict['cache_dir'] = str(config_dict['cache_dir'])
        console.print(Panel(
            json.dumps(config_dict, indent=2, default=str),
            title="Current Configuration"
        ))

    elif set_key:
        console.print(
            "[yellow]Runtime configuration changes are not supported.[/yellow]\n"
            "Edit your .env file directly or set environment variables instead.\n"
            "See .env.example for available options."
        )

    else:
        console.print("Use --show to display configuration or --set to modify it")


@app.command()
def multi(
    task: str = typer.Argument(..., help="The task to execute with multi-model system"),
    models: list[str] | None = typer.Option(
        None,
        "--models", "-m",
        help="Models to use (can specify multiple)"
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet", "-v/-q",
        help="Enable verbose output"
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Save output to file"
    ),
    use_news: bool = typer.Option(
        False,
        "--news", "-n",
        help="Include news research models"
    ),
    max_workers: int = typer.Option(
        3,
        "--workers", "-w",
        help="Maximum number of concurrent workers"
    )
):
    """Run a task using the multi-model orchestration system."""
    try:
        # Initialize the multi-model system
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Initializing multi-model system...", total=None)

            # Set environment variables if needed
            if not os.getenv("SERPAPI_API_KEY"):
                console.print("[yellow]Warning: SERPAPI_API_KEY not set. SerpAPI tools may not work.[/yellow]")

            system = MultiModelSerpAPISystem(
                serpapi_key=os.getenv("SERPAPI_API_KEY"),
                context_window_size=16384
            )
            # Setup workers with the specified model if provided
            default_model = models[0] if models else None
            system.setup_multi_model_workers(default_model=default_model)

        # Run the task
        console.print(f"\n[bold cyan]Executing task:[/bold cyan] {task}")
        if models:
            console.print(f"[bold green]Using model:[/bold green] {models[0]}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Processing with multiple models...", total=None)

            # Run asynchronously
            orchestrator_model = models[0] if models else "claude-opus-4.5"
            result = asyncio.run(
                system.execute_multi_model_workflow(task, timeout=600, orchestrator_model=orchestrator_model)
            )

        # Display results
        console.print("\n[bold green]Results:[/bold green]")

        if isinstance(result, dict):
            for model, response in result.items():
                console.print(Panel(
                    str(response),
                    title=f"Model: {model}",
                    border_style="cyan"
                ))
        else:
            console.print(result)

        # Save output if requested
        if output:
            output_data = {
                "task": task,
                "models": models,
                "results": result
            }
            output.write_text(json.dumps(output_data, indent=2, default=str))
            console.print(f"\n[green]Output saved to {output}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e!s}")
        raise typer.Exit(1) from e


@app.command()
def orchestrate(
    task: str = typer.Argument(..., help="Complex task requiring orchestration"),
    coordinator: str = typer.Option(
        "claude-opus-4.5",
        "--coordinator", "-c",
        help="Model to use as coordinator"
    ),
    workers: list[str] | None = typer.Option(
        None,
        "--workers", "-w",
        help="Worker models (can specify multiple)"
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet", "-v/-q",
        help="Enable verbose output"
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Save output to file"
    )
):
    """Execute complex tasks using model orchestration."""
    try:
        console.print(Panel(
            f"[bold cyan]Task:[/bold cyan] {task}\n"
            f"[bold cyan]Coordinator:[/bold cyan] {coordinator}\n"
            f"[bold cyan]Workers:[/bold cyan] {', '.join(workers) if workers else 'Default'}",
            title="Orchestration Setup"
        ))

        # Initialize the multi-model system
        system = MultiModelSerpAPISystem(
            serpapi_key=os.getenv("SERPAPI_API_KEY"),
            context_window_size=16384
        )
        system.setup_multi_model_workers()

        # Execute orchestrated task
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Orchestrating task execution...", total=None)

            result = asyncio.run(
                system.execute_multi_model_workflow(task, timeout=600, orchestrator_model=coordinator)
            )

        # Display orchestrated results
        console.print("\n[bold green]Orchestration Results:[/bold green]")
        console.print(result)

        if output:
            output_data = {
                "task": task,
                "coordinator": coordinator,
                "workers": workers,
                "result": result
            }
            output.write_text(json.dumps(output_data, indent=2, default=str))
            console.print(f"\n[green]Results saved to {output}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e!s}")
        raise typer.Exit(1) from e


@app.command()
def tools(
    list_tools: bool = typer.Option(
        True,
        "--list", "-l",
        help="List available tools"
    ),
    multi_model: bool = typer.Option(
        False,
        "--multi", "-m",
        help="Show multi-model tools"
    )
):
    """Manage and display available tools."""
    if list_tools:
        if multi_model:
            # Show multi-model tools
            console.print(Panel.fit(
                "[bold cyan]Multi-Model SerpAPI Tools[/bold cyan]",
                title="Tools"
            ))


            tools_info = [
                ("GoogleSearchTool", "Search the web using Google"),
                ("GoogleShoppingTool", "Search for products and prices"),
                ("GoogleMapsLocalTool", "Find local businesses and places"),
                ("GoogleScholarTool", "Search academic papers and citations"),
                ("MultiEngineSearchTool", "Search across multiple search engines")
            ]

            for name, desc in tools_info:
                console.print(f"\n[bold]{name}[/bold]")
                console.print(f"  {desc}")
        else:
            # Show regular tools
            agent = InternetAgent(verbose=False)
            console.print(Panel.fit(
                "[bold cyan]Available Tools[/bold cyan]",
                title="Tools"
            ))
            for tool in agent.tools:
                console.print(f"\n[bold]{tool.name}[/bold]")
                console.print(f"  {tool.description}")


@app.command()
def news(
    query: str = typer.Argument(..., help="News topic or query to search"),
    sources: list[str] | None = typer.Option(
        None,
        "--sources", "-s",
        help="Specific news sources to search"
    ),
    timeframe: str = typer.Option(
        "24h",
        "--time", "-t",
        help="Time frame: 1h, 24h, 7d, 30d"
    ),
    limit: int = typer.Option(
        10,
        "--limit", "-l",
        help="Maximum number of results"
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Save results to file"
    ),
    format: str = typer.Option(
        "markdown",
        "--format", "-f",
        help="Output format: markdown, json, or text"
    )
):
    """Search and analyze news articles."""
    try:
        console.print(f"[bold cyan]Searching news for:[/bold cyan] {query}")

        # Initialize multi-model system for news
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Searching news sources...", total=None)

            system = MultiModelSerpAPISystem(
                serpapi_key=os.getenv("SERPAPI_API_KEY"),
                context_window_size=16384
            )
            system.setup_multi_model_workers()

            # Construct news search query
            news_query = f"Search for recent news about {query}"
            if timeframe:
                news_query += f" from the last {timeframe}"
            if sources:
                news_query += f" from sources: {', '.join(sources)}"
            news_query += f" (limit to {limit} results)"

            # Execute search
            result = asyncio.run(
                system.execute_multi_model_workflow(news_query, timeout=300)
            )

        # Format and display results
        console.print("\n[bold green]News Results:[/bold green]\n")

        if format == "json" and output:
            output.write_text(json.dumps(result, indent=2, default=str))
            console.print(f"[green]Results saved to {output}[/green]")
        elif format == "markdown":
            md_content = f"# News Search: {query}\n\n"
            md_content += f"**Timeframe:** {timeframe}\n"
            if sources:
                md_content += f"**Sources:** {', '.join(sources)}\n"
            md_content += "\n## Results\n\n"

            if isinstance(result, dict):
                for model, response in result.items():
                    md_content += f"### Analysis by {model}\n\n"
                    md_content += str(response) + "\n\n"
            else:
                md_content += str(result)

            if output:
                output.write_text(md_content)
                console.print(f"[green]Results saved to {output}[/green]")
            else:
                console.print(Markdown(md_content))
        else:
            console.print(result)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e!s}")
        raise typer.Exit(1) from e


@app.command()
def models(
    list_all: bool = typer.Option(
        True,
        "--list", "-l",
        help="List all available models"
    ),
    category: str | None = typer.Option(
        None,
        "--category", "-c",
        help="Filter by category: general, news, research, code"
    ),
    details: bool = typer.Option(
        False,
        "--details", "-d",
        help="Show detailed information about models"
    )
):
    """List and manage available AI models."""

    # Define model categories with default configuration (updated Feb 2026)
    model_catalog = {
        "general": [
            {"name": "claude-opus-4.5", "provider": "openrouter", "desc": "Anthropic Claude Opus 4.5 - top reasoning & planning", "default": True},
            {"name": "gpt-5.2", "provider": "openrouter", "desc": "OpenAI GPT-5.2 - latest flagship model"},
            {"name": "gemini-3-pro", "provider": "openrouter", "desc": "Google Gemini 3 Pro - multimodal flagship"},
            {"name": "gemini-3-flash", "provider": "openrouter", "desc": "Google Gemini 3 Flash - fast multimodal"},
            {"name": "deepseek-v3.2", "provider": "openrouter", "desc": "DeepSeek V3.2 - strong open-weight model"},
            {"name": "grok-4.1-fast", "provider": "openrouter", "desc": "xAI Grok 4.1 Fast - creative & innovative"},
            {"name": "llama-4-maverick", "provider": "openrouter", "desc": "Meta Llama 4 Maverick - open source flagship"},
            {"name": "kimi-k2.5", "provider": "openrouter", "desc": "Moonshot Kimi K2.5 - long context specialist"},
            {"name": "minimax-m2.1", "provider": "openrouter", "desc": "MiniMax M2.1 - balanced reasoning"},
            {"name": "sonar-pro", "provider": "openrouter", "desc": "Perplexity Sonar Pro - search-optimized"},
        ],
        "code": [
            {"name": "qwen3-coder", "provider": "openrouter", "desc": "Qwen3 Coder - top coding model"},
            {"name": "qwen3-coder-plus", "provider": "openrouter", "desc": "Qwen3 Coder Plus - enhanced coding"},
            {"name": "devstral-2", "provider": "openrouter", "desc": "Mistral Devstral 2 - agentic coding"},
            {"name": "gpt-5.2-codex", "provider": "openrouter", "desc": "OpenAI GPT-5.2 Codex - code generation"},
            {"name": "deepseek-v3.2-speciale", "provider": "openrouter", "desc": "DeepSeek V3.2 Speciale - specialized tasks"},
            {"name": "grok-code-fast", "provider": "openrouter", "desc": "xAI Grok Code Fast - rapid code generation"},
            {"name": "mimo-v2-flash", "provider": "openrouter", "desc": "Xiaomi MiMo V2 Flash - cost-effective coding"},
        ],
        "research": [
            {"name": "deepseek-r1", "provider": "openrouter", "desc": "DeepSeek R1 - advanced reasoning chain-of-thought"},
            {"name": "sonar-deep-research", "provider": "openrouter", "desc": "Perplexity Sonar Deep Research"},
            {"name": "sonar-reasoning-pro", "provider": "openrouter", "desc": "Perplexity Sonar Reasoning Pro"},
            {"name": "o4-mini", "provider": "openrouter", "desc": "OpenAI O4 Mini - efficient reasoning"},
            {"name": "o3", "provider": "openrouter", "desc": "OpenAI O3 - deep reasoning"},
            {"name": "gemini-3-pro", "provider": "openrouter", "desc": "Google Gemini 3 Pro - multimodal analysis"},
            {"name": "tongyi-deepsearch", "provider": "openrouter", "desc": "Alibaba Tongyi DeepSearch"},
        ],
        "news": [
            {"name": "sonar-pro", "provider": "openrouter", "desc": "Perplexity Sonar Pro - real-time web search"},
            {"name": "sonar", "provider": "openrouter", "desc": "Perplexity Sonar - fast web search"},
            {"name": "gemini-3-flash", "provider": "openrouter", "desc": "Google Gemini 3 Flash - fast multimodal"},
            {"name": "llama-4-scout", "provider": "openrouter", "desc": "Meta Llama 4 Scout - open source"},
            {"name": "mistral-small", "provider": "openrouter", "desc": "Mistral Small 3.2 - efficient & fast"},
            {"name": "mistral-large", "provider": "openrouter", "desc": "Mistral Large 2512 - business analysis"},
        ],
        "science": [
            {"name": "deepseek-r1", "provider": "openrouter", "desc": "DeepSeek R1 - advanced reasoning"},
            {"name": "o4-mini", "provider": "openrouter", "desc": "OpenAI O4 Mini - scientific reasoning"},
            {"name": "gemini-3-pro", "provider": "openrouter", "desc": "Google Gemini 3 Pro - multimodal analysis"},
            {"name": "qwen3-235b", "provider": "openrouter", "desc": "Qwen3 235B - massive context synthesis"},
            {"name": "magistral-medium", "provider": "openrouter", "desc": "Mistral Magistral Medium - specialized"},
        ],
    }

    if list_all:
        # Create a table for models
        table = Table(title="Available AI Models", show_header=True, header_style="bold cyan")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Provider", style="magenta")
        table.add_column("Category", style="green")
        table.add_column("Default", style="yellow")
        if details:
            table.add_column("Description", style="white")

        # Filter by category if specified
        categories_to_show = [category] if category else model_catalog.keys()

        for cat in categories_to_show:
            if cat in model_catalog:
                for model in model_catalog[cat]:
                    default_marker = "✓" if model.get("default", False) else ""
                    if details:
                        table.add_row(
                            model["name"],
                            model["provider"],
                            cat,
                            default_marker,
                            model["desc"]
                        )
                    else:
                        table.add_row(
                            model["name"],
                            model["provider"],
                            cat,
                            default_marker
                        )

        console.print(table)

        # Show default model info
        console.print("\n[bold green]Default Model Configuration:[/bold green]")
        console.print("• [cyan]claude-opus-4.5[/cyan] - Used as default across all categories")
        console.print("• Provides excellent balance of capability, speed, and reliability")
        console.print("• Strong performance in reasoning, coding, and analysis tasks")

        # Show configuration tip
        console.print("\n[dim]Tip: Use --model or -m flag with any command to specify a different model[/dim]")
        console.print("[dim]Example: agentic-internet chat --model gpt-5.2[/dim]")


# MCP subcommand group
mcp_app = typer.Typer(
    name="mcp",
    help="Manage MCP (Model Context Protocol) servers",
    add_completion=False
)
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("list")
def mcp_list():
    """List available MCP servers configured in environment."""
    try:
        from .tools.mcp_integration import (
            is_mcp_available,
            load_mcp_config_from_env,
        )

        if not is_mcp_available():
            console.print("[yellow]MCP packages not installed.[/yellow]")
            console.print("Install with: [cyan]pip install smolagents mcp[/cyan]")
            raise typer.Exit(1)

        # Load configs from environment
        configs = load_mcp_config_from_env()

        if not configs:
            console.print("[yellow]No MCP servers configured in environment.[/yellow]")
            console.print("\n[dim]Configure servers using environment variables:[/dim]")
            console.print("  MCP_SERVER_1_TYPE=stdio")
            console.print("  MCP_SERVER_1_PATH=/path/to/server.py")
            console.print("  MCP_SERVER_1_NAME=my-server")
            console.print("  MCP_SERVER_1_TRUST=true")
            return

        # Display configured servers
        table = Table(title="Configured MCP Servers", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Config", style="white")
        table.add_column("Trust", style="yellow")

        for config in configs:
            trust_marker = "✓" if config.trust_remote_code else "✗"
            table.add_row(
                config.name,
                config.transport_type,
                str(config.server_config)[:50],
                trust_marker
            )

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e!s}")
        raise typer.Exit(1) from e


@mcp_app.command("info")
def mcp_info():
    """Show MCP integration status and information."""
    try:
        from .tools.mcp_integration import is_mcp_available

        console.print(Panel.fit(
            "[bold cyan]MCP Integration Status[/bold cyan]",
            title="MCP Info"
        ))

        if is_mcp_available():
            console.print("[green]✓ MCP packages installed[/green]")
            console.print("  - smolagents: [green]available[/green]")
            console.print("  - mcp: [green]available[/green]")
        else:
            console.print("[red]✗ MCP packages not installed[/red]")
            console.print("\n[dim]Install with:[/dim]")
            console.print("  [cyan]pip install smolagents mcp[/cyan]")
            console.print("  [cyan]pip install fastmcp[/cyan] (for creating servers)")
            return

        console.print("\n[bold]Supported Transports:[/bold]")
        console.print("  • [cyan]stdio[/cyan] - Local servers via subprocess")
        console.print("  • [cyan]streamable-http[/cyan] - Remote servers via HTTP")

        console.print("\n[bold]Environment Variables:[/bold]")
        console.print("  MCP_SERVER_N_TYPE     - Server type (stdio/http)")
        console.print("  MCP_SERVER_N_PATH     - Path for stdio servers")
        console.print("  MCP_SERVER_N_URL      - URL for http servers")
        console.print("  MCP_SERVER_N_NAME     - Server name")
        console.print("  MCP_SERVER_N_TRUST    - Trust remote code (true/false)")
        console.print("  MCP_SERVER_N_ENV_*    - Environment variables for server")

        console.print("\n[bold]Documentation:[/bold]")
        console.print("  See docs/MCP_INTEGRATION.md for detailed usage")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e!s}")
        raise typer.Exit(1) from e


@mcp_app.command("run")
def mcp_run(
    task: str = typer.Argument(..., help="The task to execute with MCP tools"),
    server_path: str | None = typer.Option(
        None,
        "--server", "-s",
        help="Path to MCP server script (for stdio transport)"
    ),
    server_url: str | None = typer.Option(
        None,
        "--url", "-u",
        help="URL of MCP server (for http transport)"
    ),
    trust: bool = typer.Option(
        False,
        "--trust/--no-trust", "-t/-T",
        help="Trust remote code execution (required for MCP tools)"
    ),
    model: str | None = typer.Option(
        None,
        "--model", "-m",
        help="Model to use for the agent"
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet", "-v/-q",
        help="Enable verbose output"
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Save output to file"
    ),
):
    """Run a task using MCP tools from a server."""
    try:
        from .tools.mcp_integration import is_mcp_available, mcp_tools

        if not is_mcp_available():
            console.print("[red]MCP packages not installed.[/red]")
            console.print("Install with: [cyan]pip install smolagents mcp[/cyan]")
            raise typer.Exit(1)

        if not server_path and not server_url:
            console.print("[red]Error: Must provide --server or --url[/red]")
            raise typer.Exit(1)

        if not trust:
            console.print("[yellow]Warning: MCP tools require --trust flag to execute.[/yellow]")
            console.print("Add [cyan]--trust[/cyan] or [cyan]-t[/cyan] to enable tool execution.")
            raise typer.Exit(1)

        console.print(Panel.fit(
            f"[bold cyan]Task:[/bold cyan] {task}\n"
            f"[bold cyan]Server:[/bold cyan] {server_path or server_url}\n"
            f"[bold cyan]Trust:[/bold cyan] {'Enabled' if trust else 'Disabled'}",
            title="MCP Run"
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Connecting to MCP server...", total=None)

            # Connect to MCP server and run task
            with mcp_tools(
                server_path=server_path,
                server_url=server_url,
                trust_remote_code=trust,
            ) as tools:
                progress.add_task(f"Loaded {len(tools)} MCP tools", total=None)

                # Create agent with MCP tools
                agent = InternetAgent(
                    model_id=model,
                    tools=list(tools),
                    verbose=verbose,
                )

                progress.add_task("Executing task...", total=None)
                result = agent.run(task)

        if output:
            output.write_text(str(result))
            console.print(f"\n[green]Output saved to {output}[/green]")
        else:
            console.print("\n[bold green]Result:[/bold green]")
            console.print(result)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e!s}")
        raise typer.Exit(1) from e


@mcp_app.command("test")
def mcp_test(
    server_path: str | None = typer.Option(
        None,
        "--server", "-s",
        help="Path to MCP server script to test"
    ),
    server_url: str | None = typer.Option(
        None,
        "--url", "-u",
        help="URL of MCP server to test"
    ),
):
    """Test connection to an MCP server and list available tools."""
    try:
        from .tools.mcp_integration import is_mcp_available, mcp_tools

        if not is_mcp_available():
            console.print("[red]MCP packages not installed.[/red]")
            console.print("Install with: [cyan]pip install smolagents mcp[/cyan]")
            raise typer.Exit(1)

        if not server_path and not server_url:
            console.print("[red]Error: Must provide --server or --url[/red]")
            raise typer.Exit(1)

        server_display = server_path or server_url
        console.print(f"[cyan]Testing connection to:[/cyan] {server_display}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Connecting to MCP server...", total=None)

            try:
                with mcp_tools(
                    server_path=server_path,
                    server_url=server_url,
                    trust_remote_code=True,  # Required to load tools
                ) as tools:
                    console.print("\n[green]✓ Connected successfully![/green]")
                    console.print(f"[green]✓ Found {len(tools)} tools[/green]\n")

                    if tools:
                        table = Table(
                            title="Available MCP Tools",
                            show_header=True,
                            header_style="bold cyan"
                        )
                        table.add_column("Tool Name", style="cyan")
                        table.add_column("Description", style="white")

                        for tool in tools:
                            desc = getattr(tool, 'description', 'No description')[:80]
                            table.add_row(tool.name, desc)

                        console.print(table)
                    else:
                        console.print("[yellow]No tools found on server[/yellow]")

            except Exception as conn_error:
                console.print(f"\n[red]✗ Connection failed:[/red] {conn_error!s}")
                raise typer.Exit(1) from conn_error

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e!s}")
        raise typer.Exit(1) from e


@app.command()
def version():
    """Display version information."""
    from . import __version__
    console.print(f"[bold cyan]Agentic Internet[/bold cyan] version {__version__}")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
