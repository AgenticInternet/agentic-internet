"""Command-line interface for the Agentic Internet application."""

import typer
import asyncio
import os
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
import json

from .agents.internet_agent import InternetAgent, ResearchAgent
from .agents.multi_model_serpapi import MultiModelSerpAPISystem
from .config.settings import settings

app = typer.Typer(
    name="agentic-internet",
    help="Advanced AI agent for intelligent internet interactions",
    add_completion=False
)
console = Console()


@app.command()
def chat(
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Model to use (e.g., claude-4, gpt-5-chat, gemini-2.5-flash)"
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
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def run(
    task: str = typer.Argument(..., help="The task to execute"),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Model to use"
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
    output: Optional[Path] = typer.Option(
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
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def research(
    topic: str = typer.Argument(..., help="The topic to research"),
    depth: str = typer.Option(
        "moderate",
        "--depth", "-d",
        help="Research depth: quick, moderate, or deep"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Model to use"
    ),
    output: Optional[Path] = typer.Option(
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
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show", "-s",
        help="Show current configuration"
    ),
    set_key: Optional[str] = typer.Option(
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
        # Parse and set configuration
        try:
            key, value = set_key.split("=", 1)
            # This is a simplified example - in production you'd want proper configuration management
            console.print(f"[yellow]Configuration management not fully implemented[/yellow]")
            console.print(f"Would set: {key} = {value}")
        except ValueError:
            console.print("[bold red]Error:[/bold red] Invalid format. Use key=value")
            raise typer.Exit(1)
    
    else:
        console.print("Use --show to display configuration or --set to modify it")


@app.command()
def multi(
    task: str = typer.Argument(..., help="The task to execute with multi-model system"),
    models: Optional[List[str]] = typer.Option(
        None,
        "--models", "-m",
        help="Models to use (can specify multiple)"
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet", "-v/-q",
        help="Enable verbose output"
    ),
    output: Optional[Path] = typer.Option(
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
            if not os.getenv("SERP_API_KEY"):
                console.print("[yellow]Warning: SERP_API_KEY not set. SerpAPI tools may not work.[/yellow]")
            
            system = MultiModelSerpAPISystem(
                serpapi_key=os.getenv("SERPAPI_API_KEY"),
                context_window_size=16384
            )
            system.setup_multi_model_workers()
        
        # Run the task
        console.print(f"\n[bold cyan]Executing task:[/bold cyan] {task}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Processing with multiple models...", total=None)
            
            # Run asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                system.execute_multi_model_workflow(task, timeout=600)
            )
            loop.close()
        
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
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def orchestrate(
    task: str = typer.Argument(..., help="Complex task requiring orchestration"),
    coordinator: str = typer.Option(
        "claude-4",
        "--coordinator", "-c",
        help="Model to use as coordinator"
    ),
    workers: Optional[List[str]] = typer.Option(
        None,
        "--workers", "-w",
        help="Worker models (can specify multiple)"
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet", "-v/-q",
        help="Enable verbose output"
    ),
    output: Optional[Path] = typer.Option(
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
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                system.execute_multi_model_workflow(task, timeout=600, orchestrator_model=coordinator)
            )
            loop.close()
        
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
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


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
            
            from .agents.multi_model_serpapi import (
                GoogleSearchTool, GoogleShoppingTool, 
                GoogleMapsLocalTool, GoogleScholarTool,
                MultiEngineSearchTool
            )
            
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
    sources: Optional[List[str]] = typer.Option(
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
    output: Optional[Path] = typer.Option(
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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                system.execute_multi_model_workflow(news_query, timeout=300)
            )
            loop.close()
        
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
            md_content += f"\n## Results\n\n"
            
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
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def models(
    list_all: bool = typer.Option(
        True,
        "--list", "-l",
        help="List all available models"
    ),
    category: Optional[str] = typer.Option(
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

    # deepseek/deepseek-chat-v3.1
    # 
    
    # Define model categories with default configuration
    model_catalog = {
        "general": [
            {"name": "claude-4", "provider": "openrouter", "desc": "Latest Claude 4 - most capable model", "default": True},
            {"name": "gpt-5-chat", "provider": "openrouter", "desc": "GPT-5 Gemini 2.5 Flash - multimodal capabilities"},
            {"name": "gemini-2.5-flash", "provider": "openrouter", "desc": "Gemini 2.5 Flash - multimodal capabilities"},
            {"name": "meta-llama/Llama-4-scout", "provider": "openrouter", "desc": "Open source LLM"},
            {"name": "mistral-small-3.2-24b-instruct", "provider": "openrouter", "desc": "Mistral Small - multimodal capabilities"},
            {"name": "sonar", "provider": "openrouter", "desc": "Sonar - multimodal capabilities"},
            {"name": "sonar-reasoning", "provider": "openrouter", "desc": "Sonar Reasoning - multimodal capabilities"},
            {"name": "o3-mini", "provider": "openrouter", "desc": "O3 Mini - multimodal capabilities"},
            {"name": "mistral-large-2411", "provider": "openrouter", "desc": "Mistral Large - multimodal capabilities"},
            {"name": "deepseek-r1-0528", "provider": "openrouter", "desc": "Advanced reasoning model"},
            {"name": "deepseek-chat-v3.1", "provider": "openrouter", "desc": "Advanced reasoning model"},
            {"name": "qwen/qwen3-235b-a22b", "provider": "openrouter", "desc": "Multilingual capabilities"},
            {"name": "gpt-oss-120b", "provider": "openrouter", "desc": "Opensource OpenAI Model"},
            {"name": "gemini-2.5-pro", "provider": "openrouter", "desc": "Gemini 2.5 Pro - multimodal capabilities"},
            {"name": "gpt-5", "provider": "openrouter", "desc": "GPT-5 - multimodal capabilities"},
            {"name": "kimi-k2", "provider": "openrouter", "desc": "Kimi K2 - multimodal capabilities"},
        ],
        "code": [
            {"name": "qwen3-coder", "provider": "openrouter", "desc": "Multilingual capabilities"},
            {"name": "deepseek-chat-v3.1", "provider": "openrouter", "desc": "Advanced reasoning model"},
            {"name": "gpt-oss-120b", "provider": "openrouter", "desc": "Opensource OpenAI Model"},
            {"name": "gemini-2.5-pro", "provider": "openrouter", "desc": "Gemini 2.5 Pro - multimodal capabilities"},
            {"name": "gpt-5", "provider": "openrouter", "desc": "GPT-5 - multimodal capabilities"},
        ],
        "research": [
            {"name": "deepseek-r1-0528", "provider": "openrouter", "desc": "Advanced reasoning model"},
            {"name": "deepseek-chat-v3.1", "provider": "openrouter", "desc": "Advanced reasoning model"},
            {"name": "gpt-oss-120b", "provider": "openrouter", "desc": "Opensource OpenAI Model"},
            {"name": "gemini-2.5-pro", "provider": "openrouter", "desc": "Gemini 2.5 Pro - multimodal capabilities"},
            {"name": "gpt-5", "provider": "openrouter", "desc": "GPT-5 - multimodal capabilities"},
        ],
        "news": [
            {"name": "gemini-2.5-flash", "provider": "openrouter", "desc": "Gemini 2.5 Flash - multimodal capabilities"},
            {"name": "meta-llama/Llama-4-scout", "provider": "openrouter", "desc": "Open source LLM"},
            {"name": "mistral-small-3.2-24b-instruct", "provider": "openrouter", "desc": "Mistral Small - multimodal capabilities"},
            {"name": "sonar", "provider": "openrouter", "desc": "Sonar - multimodal capabilities"},
            {"name": "sonar-reasoning", "provider": "openrouter", "desc": "Sonar Reasoning - multimodal capabilities"},
            {"name": "mistral-large-2411", "provider": "openrouter", "desc": "Mistral Large - multimodal capabilities"},
        ],
        "science": [
            {"name": "gemini-2.5-flash", "provider": "openrouter", "desc": "Gemini 2.5 Flash - multimodal capabilities"},
            {"name": "deepseek-chat-v3.1", "provider": "openrouter", "desc": "Advanced reasoning model"},
            {"name": "deepseek-r1-0528", "provider": "openrouter", "desc": "Advanced reasoning model"},
            {"name": "gpt-oss-120b", "provider": "openrouter", "desc": "Opensource OpenAI Model"},
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
        console.print("• [cyan]claude-4[/cyan] - Used as default across all categories")
        console.print("• Provides excellent balance of capability, speed, and reliability")
        console.print("• Strong performance in reasoning, coding, and analysis tasks")
        
        # Show configuration tip
        console.print("\n[dim]Tip: Use --model or -m flag with any command to specify a different model[/dim]")
        console.print("[dim]Example: agentic-internet chat --model gpt-5-chat[/dim]")


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
