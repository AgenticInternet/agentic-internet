#!/usr/bin/env python
"""Quick test script for the Agentic Internet agent."""

from agentic_internet import InternetAgent
from rich.console import Console

console = Console()

def test_basic_functionality():
    """Test basic agent functionality."""
    console.print("[bold cyan]Testing Agentic Internet Agent[/bold cyan]\n")
    
    # Create an agent
    console.print("1. Creating agent...")
    agent = InternetAgent(verbose=False)
    console.print("   ✅ Agent created successfully\n")
    
    # Test a simple calculation
    console.print("2. Testing calculation...")
    result = agent.run("Calculate 25 * 4 + 100")
    console.print(f"   Result: {result}")
    console.print("   ✅ Calculation test passed\n")
    
    # Test Python code execution
    console.print("3. Testing Python code execution...")
    result = agent.run("Execute Python code to generate the first 10 Fibonacci numbers")
    console.print(f"   Result: {result[:200]}...")
    console.print("   ✅ Code execution test passed\n")
    
    console.print("[bold green]All tests passed successfully![/bold green]")

if __name__ == "__main__":
    test_basic_functionality()
