"""Advanced usage examples demonstrating different agent types and capabilities."""

from agentic_internet import (
    InternetAgent,
    BrowserAutomationAgent,
    DataAnalysisAgent,
    ContentCreationAgent,
    MarketResearchAgent,
    TechnicalSupportAgent
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv
import os

load_dotenv()
console = Console()


def example_agent_types():
    """Demonstrate different agent types."""
    console.print("\n[bold cyan]Example: Different Agent Types[/bold cyan]")
    
    # ToolCallingAgent - Best for tool-heavy tasks
    console.print("\n[yellow]1. ToolCallingAgent (default)[/yellow]")
    agent1 = InternetAgent(
        model_id="openrouter/openai/gpt-5",
        agent_type="tool_calling",
        verbose=True
    )
    result1 = agent1.run("""
        - Search for the latest Agentic Internet articles and summarize the top 5 articles
        - Use others tools and Agents to perfom this task
    """)
    
    # CodeAgent - Best for data analysis and code generation
    # console.print("\n[yellow]2. CodeAgent[/yellow]")
    # agent2 = InternetAgent(
    #     model_id="openrouter/openai/gpt-5-chat",
    #     agent_type="code",
    #     verbose=True
    # )
    # result2 = agent2.run("""
    # Generate a Python function that calculates compound interest
    # and demonstrate it with an example of $1000 at 5% for 10 years.
    # """)
    
    # MultiStepAgent - Best for complex reasoning
    #console.print("\n[yellow]3. MultiStepAgent[/yellow]")
    # agent3 = InternetAgent(
    #     model_id="openrouter/anthropic/claude-sonnet-4",
    #     agent_type="multi_step",
    #     verbose=True
    # )
    # result3 = agent3.run("""
    # Plan a 3-day trip to Tokyo including:
    # 1. Must-see attractions
    # 2. Restaurant recommendations
    # 3. Transportation tips
    # 4. Budget estimate
    # """)
    
    return {
        "tool_calling": result1,
        # "code": result2,
        #"multi_step": result3
    }


def example_browser_automation():
    """Demonstrate browser automation capabilities."""
    console.print("\n[bold cyan]Example: Browser Automation[/bold cyan]")
    
    # Check if Browser Use API key is available
    if not os.environ.get("BROWSER_USE_API_KEY"):
        console.print("[yellow]Note: Set BROWSER_USE_API_KEY for full browser automation features[/yellow]")
        console.print("Get your API key at: https://cloud.browser-use.com/billing")
        return None
    
    agent = BrowserAutomationAgent(
        model_id="openrouter/anthropic/claude-sonnet-4",
        verbose=True
    )
    
    # Example 1: Scrape structured data
    console.print("\n[green]Scraping Hacker News top posts...[/green]")
    schema = {
        "posts": [
            {
                "title": "string",
                "url": "string",
                "points": "number",
                "comments": "number"
            }
        ]
    }
    
    result = agent.scrape_structured_data(
        url="https://news.ycombinator.com",
        data_schema=schema
    )
    
    return result


def example_data_analysis():
    """Demonstrate data analysis capabilities."""
    console.print("\n[bold cyan]Example: Data Analysis Agent[/bold cyan]")
    
    agent = DataAnalysisAgent(
        model_id="openrouter/anthropic/claude-sonnet-4",
        verbose=True
    )
    
    # Sample sales data
    sales_data = [
        {"month": "Jan", "sales": 45000, "costs": 32000, "region": "North"},
        {"month": "Feb", "sales": 52000, "costs": 35000, "region": "North"},
        {"month": "Mar", "sales": 48000, "costs": 33000, "region": "North"},
        {"month": "Jan", "sales": 38000, "costs": 28000, "region": "South"},
        {"month": "Feb", "sales": 41000, "costs": 29000, "region": "South"},
        {"month": "Mar", "sales": 44000, "costs": 30000, "region": "South"}
    ]
    
    result = agent.analyze_dataset(
        data=sales_data,
        analysis_type="comprehensive"
    )
    
    return result


def example_content_creation():
    """Demonstrate content creation capabilities."""
    console.print("\n[bold cyan]Example: Content Creation Agent[/bold cyan]")
    
    agent = ContentCreationAgent(
        model_id="openrouter/anthropic/claude-sonnet-4",
        verbose=True
    )
    
    # Write an article
    article = agent.write_article(
        topic="The Future of AI Agents in Business",
        style="informative",
        word_count=300,
        sources_required=True
    )
    
    # Summarize content
    summary = agent.summarize_content(
        content=article["content"],
        summary_type="bullet_points"
    )
    
    return {
        "article": article,
        "summary": summary
    }


def example_market_research():
    """Demonstrate market research capabilities."""
    console.print("\n[bold cyan]Example: Market Research Agent[/bold cyan]")
    
    agent = MarketResearchAgent(
        model_id="openrouter/anthropic/claude-sonnet-4",
        verbose=True
    )
    
    # Analyze market trends
    trends = agent.market_trends(
        industry="AI and Machine Learning",
        timeframe="2024"
    )
    
    return trends


def example_technical_support():
    """Demonstrate technical support capabilities."""
    console.print("\n[bold cyan]Example: Technical Support Agent[/bold cyan]")
    
    agent = TechnicalSupportAgent(
        model_id="openrouter/anthropic/claude-sonnet-4",
        verbose=True
    )
    
    # Code review example
    code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    avg = total / len(numbers)
    return avg
"""
    
    review = agent.code_review(
        code=code,
        language="python",
        focus_areas=["performance", "error_handling", "best_practices"]
    )
    
    return review


def display_capabilities_table():
    """Display a table of agent capabilities."""
    table = Table(title="Agent Types and Their Best Use Cases")
    
    table.add_column("Agent Type", style="cyan", no_wrap=True)
    table.add_column("Best For", style="green")
    table.add_column("Key Features", style="yellow")
    
    table.add_row(
        "ToolCallingAgent",
        "Tool-heavy tasks, web searches",
        "Efficient tool use, structured responses"
    )
    table.add_row(
        "CodeAgent",
        "Data analysis, code generation",
        "Can execute Python code, data manipulation"
    )
    table.add_row(
        "MultiStepAgent",
        "Complex reasoning, planning",
        "Step-by-step reasoning, detailed analysis"
    )
    
    console.print(table)
    
    # Specialized agents table
    table2 = Table(title="Specialized Agents")
    
    table2.add_column("Agent Class", style="cyan", no_wrap=True)
    table2.add_column("Purpose", style="green")
    table2.add_column("Key Methods", style="yellow")
    
    table2.add_row(
        "BrowserAutomationAgent",
        "Web automation & scraping",
        "scrape_structured_data, fill_form, monitor_website"
    )
    table2.add_row(
        "DataAnalysisAgent",
        "Data analysis & visualization",
        "analyze_dataset, compare_datasets"
    )
    table2.add_row(
        "ContentCreationAgent",
        "Writing & content generation",
        "write_article, summarize_content"
    )
    table2.add_row(
        "MarketResearchAgent",
        "Market & competitive analysis",
        "analyze_competitor, market_trends"
    )
    table2.add_row(
        "TechnicalSupportAgent",
        "Tech support & troubleshooting",
        "troubleshoot, code_review"
    )
    
    console.print(table2)


def main():
    """Run advanced examples."""
    console.print(Panel.fit(
        "[bold green]Agentic Internet - Advanced Usage Examples[/bold green]\n"
        "Demonstrating different agent types and specialized capabilities",
        title="Welcome"
    ))
    
    # Display capabilities
    display_capabilities_table()
    
    examples_to_run = [
        ("Agent Types Demo", example_agent_types),
        #("Browser Automation", example_browser_automation),
        #("Data Analysis", example_data_analysis),
        # ("Content Creation", example_content_creation),
        #("Market Research", example_market_research),
        # ("Technical Support", example_technical_support)
    ]
    
    results = {}
    
    for name, example_func in examples_to_run:
        try:
            console.print(f"\n{'='*60}")
            console.print(f"[bold]Running: {name}[/bold]")
            console.print(f"{'='*60}")
            
            result = example_func()
            results[name] = {"status": "success", "result": result}
            
        except Exception as e:
            console.print(f"[bold red]Error in {name}: {str(e)}[/bold red]")
            results[name] = {"status": "error", "error": str(e)}
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold green]Examples Completed![/bold green]")
    
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] == "error")
    
    console.print(f"Successful: {successful}")
    console.print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
