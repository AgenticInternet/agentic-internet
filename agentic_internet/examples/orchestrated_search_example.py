"""
Example demonstrating orchestrated search with multiple agents.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Load environment variables
load_dotenv()

# Import our components
from agentic_internet.agents.internet_agent import InternetAgent
from agentic_internet.agents.search_orchestrator import SearchOrchestrator, create_search_orchestrator
from agentic_internet.tools.web_search import NewsSearchTool, WebScraperTool, WebSearchTool

console = Console()


def demo_basic_orchestrator():
    """Demonstrate basic orchestrator functionality."""
    console.print(Panel.fit(
        "[bold cyan]Basic Orchestrator Demo[/bold cyan]\n"
        "Setting up a search orchestrator with multiple agents",
        title="Demo 1"
    ))

    # Create tools for the agents
    tools = [
        WebSearchTool(),
        WebScraperTool(),
        NewsSearchTool()
    ]

    # Create the orchestrator
    orchestrator = create_search_orchestrator(tools, verbose=True)

    # Perform a search
    query = "Latest developments in AI agents and multi-agent systems"
    console.print(f"\n[bold blue]Searching for:[/bold blue] {query}\n")

    results = orchestrator.search(query, parallel=True)

    # Display results
    console.print("\n[bold green]Search Results:[/bold green]")

    if results.get("synthesis"):
        console.print(Panel(
            Markdown(results["synthesis"]),
            title="Synthesized Results"
        ))

    if "agent_results" in results:
        for agent_name, agent_data in results["agent_results"].items():
            console.print(f"\n[bold]{agent_name}[/bold] ({agent_data.get('specialization', 'general')}):")
            console.print(agent_data.get('result', 'No results')[:500] + "...")

    # Show performance report
    perf_report = orchestrator.get_performance_report()
    console.print("\n[bold cyan]Performance Report:[/bold cyan]")
    console.print(json.dumps(perf_report, indent=2))


def demo_integrated_orchestrator():
    """Demonstrate orchestrator integrated with InternetAgent."""
    console.print(Panel.fit(
        "[bold cyan]Integrated Orchestrator Demo[/bold cyan]\n"
        "Using orchestrated search within an InternetAgent",
        title="Demo 2"
    ))

    # Create the orchestrator
    search_tools = [WebScraperTool(), NewsSearchTool()]
    orchestrator = create_search_orchestrator(search_tools, verbose=False)

    # Create an orchestrated web search tool
    orchestrated_search_tool = WebSearchTool(
        use_orchestrator=True,
        orchestrator=orchestrator
    )

    # Create an InternetAgent with the orchestrated search tool
    agent = InternetAgent(
        tools=[orchestrated_search_tool, WebScraperTool()],
        verbose=True
    )

    # Run a task that will use orchestrated search
    task = "Search for information about the latest breakthroughs in quantum computing and summarize the key findings"
    console.print(f"\n[bold blue]Task:[/bold blue] {task}\n")

    result = agent.run(task)

    console.print(Panel(
        Markdown(str(result)),
        title="Agent Result with Orchestrated Search"
    ))


def demo_custom_orchestrator():
    """Demonstrate custom orchestrator configuration."""
    console.print(Panel.fit(
        "[bold cyan]Custom Orchestrator Demo[/bold cyan]\n"
        "Creating a custom orchestrator with specialized agents",
        title="Demo 3"
    ))

    # Create orchestrator with custom configuration
    orchestrator = SearchOrchestrator(
        max_workers=2,  # Limit parallel execution
        verbose=True
    )

    # Manually add specialized agents if model is available
    try:
        from smolagents import LiteLLMModel, ToolCallingAgent

        # Try to get a model
        model = None
        if os.getenv("OPENAI_API_KEY"):
            model = LiteLLMModel(
                model_id="gpt-3.5-turbo",
                api_key=os.getenv("OPENAI_API_KEY")
            )

        if model:
            # Create tools
            tools = [WebSearchTool(), NewsSearchTool()]

            # Add custom specialized agents
            academic_agent = ToolCallingAgent(tools=tools, model=model)
            orchestrator.add_agent(
                "academic_researcher",
                academic_agent,
                "Focuses on academic and research sources",
                "academic"
            )

            market_agent = ToolCallingAgent(tools=tools, model=model)
            orchestrator.add_agent(
                "market_analyst",
                market_agent,
                "Analyzes market trends and business data",
                "market"
            )

            # Run specialized search
            query = "Impact of large language models on software development industry"
            results = orchestrator.search(
                query,
                agents_to_use=["academic_researcher", "market_analyst"],
                parallel=True
            )

            console.print("\n[bold green]Specialized Search Results:[/bold green]")
            if "agent_results" in results:
                for agent_name, agent_data in results["agent_results"].items():
                    console.print(f"\n[bold]{agent_name}[/bold]:")
                    console.print(agent_data.get('result', 'No results')[:500] + "...")
        else:
            console.print("[yellow]No model available for custom agents. Skipping...[/yellow]")

    except Exception as e:
        console.print(f"[red]Error creating custom agents: {e}[/red]")


async def demo_async_orchestrator():
    """Demonstrate async orchestrator functionality."""
    console.print(Panel.fit(
        "[bold cyan]Async Orchestrator Demo[/bold cyan]\n"
        "Using async methods for concurrent execution",
        title="Demo 4"
    ))

    # Create tools
    tools = [WebSearchTool(), NewsSearchTool()]

    # Create orchestrator with async support
    orchestrator = SearchOrchestrator(use_async=True, verbose=True)
    orchestrator.setup_default_agents(tools)

    # Run multiple searches concurrently
    queries = [
        "Future of artificial intelligence",
        "Climate change solutions 2024",
        "Space exploration recent discoveries"
    ]

    console.print("[bold blue]Running concurrent searches:[/bold blue]")
    for q in queries:
        console.print(f"  • {q}")

    # Execute searches concurrently
    tasks = [orchestrator.search_async(q) for q in queries]
    results = await asyncio.gather(*tasks)

    # Display results
    for i, (query, result) in enumerate(zip(queries, results, strict=False), 1):
        console.print(f"\n[bold green]Query {i}: {query}[/bold green]")
        if "successful_agents" in result:
            console.print(f"Successful agents: {result['successful_agents']}/{result['total_agents']}")
        if result.get("synthesis"):
            console.print(f"Synthesis: {result['synthesis'][:300]}...")


def main():
    """Run all demonstrations."""
    console.print(Panel.fit(
        "[bold magenta]Search Orchestrator Demonstrations[/bold magenta]\n"
        "Showing different ways to use orchestrated search with multiple agents",
        title="Orchestrated Search Examples"
    ))

    # Check for required dependencies
    has_deps = True
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        console.print("[yellow]Warning: duckduckgo-search not installed[/yellow]")
        has_deps = False

    if not has_deps:
        console.print("\nInstall required dependencies:")
        console.print("pip install duckduckgo-search")
        return

    # Run demonstrations
    try:
        # Demo 1: Basic orchestrator
        console.print("\n" + "="*60)
        demo_basic_orchestrator()

        # Demo 2: Integrated with InternetAgent
        console.print("\n" + "="*60)
        demo_integrated_orchestrator()

        # Demo 3: Custom configuration
        console.print("\n" + "="*60)
        demo_custom_orchestrator()

        # Demo 4: Async execution
        console.print("\n" + "="*60)
        asyncio.run(demo_async_orchestrator())

    except KeyboardInterrupt:
        console.print("\n[yellow]Demonstrations interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during demonstrations: {e}[/red]")
        import traceback
        traceback.print_exc()

    console.print("\n[bold green]✅ All demonstrations complete![/bold green]")
    console.print("\n[bold]Key Takeaways:[/bold]")
    console.print("• Orchestrator runs multiple specialized agents in parallel")
    console.print("• Results can be synthesized for comprehensive answers")
    console.print("• Easy integration with existing InternetAgent")
    console.print("• Supports both sync and async execution")
    console.print("• Customizable with different agent specializations")


if __name__ == "__main__":
    main()
