"""Basic usage examples for the Agentic Internet agent."""

from agentic_internet import InternetAgent, ResearchAgent
from rich.console import Console
from dotenv import load_dotenv
import os
load_dotenv()

os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

console = Console()


def example_simple_task():
    """Run a simple task with the agent."""
    console.print("\n[bold cyan]Example 1: Simple Web Search[/bold cyan]")
    
    # Create an agent
    agent = InternetAgent(verbose=True)
    
    # Run a simple search task
    result = agent.run("Search for the latest developments in quantum computing")
    
    return result


def example_web_scraping():
    """Example of web scraping with the agent."""
    console.print("\n[bold cyan]Example 2: Web Scraping[/bold cyan]")
    
    agent = InternetAgent(verbose=True)
    
    # Scrape content from a website
    result = agent.run(
        "Scrape the main content from https://www.python.org and summarize what Python is about"
    )
    
    return result


def example_data_analysis():
    """Example of data analysis with the agent."""
    console.print("\n[bold cyan]Example 3: Data Analysis[/bold cyan]")
    
    agent = InternetAgent(verbose=True)
    
    # Analyze some data
    result = agent.run("""
    Analyze this dataset and provide insights:
    [
        {"month": "Jan", "sales": 15000, "costs": 8000},
        {"month": "Feb", "sales": 18000, "costs": 9000},
        {"month": "Mar", "sales": 22000, "costs": 10000},
        {"month": "Apr", "sales": 19000, "costs": 9500}
    ]
    Calculate profit margins and identify trends.
    """)
    
    return result


def example_research():
    """Example of conducting research with the Research Agent."""
    console.print("\n[bold cyan]Example 4: Research Agent[/bold cyan]")
    
    # Create a research agent
    researcher = ResearchAgent(verbose=True)
    
    # Conduct quick research
    quick_result = researcher.research(
        topic="Artificial General Intelligence",
        depth="quick"
    )
    
    console.print(f"\nQuick research completed: {quick_result['topic']}")
    
    # Conduct deep research
    deep_result = researcher.research(
        topic="Impact of AI on job market",
        depth="deep"
    )
    
    console.print(f"\nDeep research completed: {deep_result['topic']}")
    
    # Show research history
    history = researcher.get_research_history()
    console.print(f"\nTotal research tasks completed: {len(history)}")
    
    return history


def example_multi_tool_task():
    """Example of a task using multiple tools."""
    console.print("\n[bold cyan]Example 5: Multi-Tool Task[/bold cyan]")
    
    agent = InternetAgent(
        model_id="openrouter/anthropic/claude-sonnet-4",
        verbose=True, 
        max_iterations=15
    )
    
    # Complex task requiring multiple tools
    result = agent.run("""
    1. Search for the current Bitcoin price
    2. Search for recent news about cryptocurrency regulation
    3. Based on the findings, provide an analysis of the current crypto market sentiment
    """)
    
    return result


def example_code_execution():
    """Example of code execution with the agent."""
    console.print("\n[bold cyan]Example 6: Code Execution[/bold cyan]")
    
    agent = InternetAgent(verbose=True)
    
    # Execute Python code
    result = agent.run("""
    Write and execute Python code to:
    1. Generate a list of the first 20 Fibonacci numbers
    2. Calculate their sum
    3. Find the average
    4. Create a simple visualization of the growth pattern
    """)
    
    return result


def example_news_analysis():
    """Example of news analysis."""
    console.print("\n[bold cyan]Example 7: News Analysis[/bold cyan]")
    
    agent = InternetAgent(verbose=True)
    
    # Analyze recent news
    result = agent.run("""
    Find recent news about renewable energy and:
    1. Summarize the top 3 stories
    2. Identify common themes
    3. Analyze the overall sentiment (positive/negative/neutral)
    """)
    
    return result


def main():
    """Run all examples."""
    console.print("[bold green]Agentic Internet - Usage Examples[/bold green]\n")
    
    examples = [
        # ("Multi-Tool Task", example_multi_tool_task),
        ("Research", example_research),
        ("Code Execution", example_code_execution),
        ("News Analysis", example_news_analysis)
    ]
    
    results = {}
    
    for name, example_func in examples:
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
    console.print(f"Successful: {sum(1 for r in results.values() if r['status'] == 'success')}")
    console.print(f"Failed: {sum(1 for r in results.values() if r['status'] == 'error')}")


if __name__ == "__main__":
    main()
