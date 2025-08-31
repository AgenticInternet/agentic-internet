"""
Example usage of the Multi-Model SerpAPI Enhanced System
"""

import asyncio
import os
from agentic_internet.agents.multi_model_serpapi import MultiModelSerpAPISystem
from rich.console import Console
import json
from dotenv import load_dotenv

load_dotenv()


console = Console()

# Workflow Templates
class MultiModelWorkflowTemplates:
    """Advanced workflow templates optimized for multi-model coordination"""
    
    @staticmethod
    def comprehensive_competitive_analysis(company: str, industry: str = None) -> str:
        industry_context = f" in the {industry} industry" if industry else ""
        return f"""
        Conduct comprehensive competitive analysis for "{company}"{industry_context}:
        
        PHASE 1 - Strategic Intelligence Gathering:
        1. Company overview and strategic positioning analysis
        2. Recent strategic moves and market developments
        3. Leadership and organizational structure assessment
        
        PHASE 2 - Multi-Engine Market Research:
        1. Cross-platform news and media analysis
        2. Industry trend identification and market context
        3. Regulatory and policy impact assessment
        
        PHASE 3 - Academic and Research Foundation:
        1. Scholarly analysis and research papers about the company/industry
        2. Citation network analysis for credibility assessment
        3. Academic perspectives on market positioning
        
        PHASE 4 - Financial and Product Intelligence:
        1. Product portfolio analysis and pricing strategy
        2. E-commerce presence and digital strategy assessment
        3. Customer review sentiment and market perception
        
        DELIVERABLE: Multi-model validated competitive intelligence dossier with strategic recommendations
        """
    
    @staticmethod
    def advanced_market_opportunity_assessment(market: str, geography: str = "global") -> str:
        return f"""
        Advanced market opportunity assessment for "{market}" ({geography} scope):
        
        PRIMARY RESEARCH:
        - Multi-engine market intelligence gathering
        - News, trends, and current market dynamics
        - Regulatory and policy landscape analysis
        
        QUANTITATIVE ANALYSIS:
        - Market size calculations and projections
        - Statistical trend analysis and modeling
        - ROI and investment opportunity calculations
        
        BUSINESS INTELLIGENCE:
        - Key player identification and analysis
        - Business model evaluation
        - Strategic partnership opportunities
        - Market entry strategy development
        
        ACADEMIC FOUNDATION:
        - Scholarly research on market dynamics
        - Academic validation of market assumptions
        - Research-based market theories and frameworks
        
        COMMERCIAL ANALYSIS:
        - Product/service demand analysis
        - Pricing strategy and competitive positioning
        - Customer behavior and preference analysis
        - E-commerce and digital opportunity assessment
        
        DELIVERABLE: Comprehensive market opportunity report with multi-model validated insights
        """
    
    @staticmethod
    def technology_landscape_analysis(technology: str, timeframe: str = "2024-2026") -> str:
        return f"""
        Advanced technology landscape analysis for "{technology}" ({timeframe}):
        
        ACADEMIC RESEARCH FOUNDATION:
        1. Latest research papers and scientific breakthroughs
        2. Citation analysis and research trend identification
        3. Academic institution and researcher network mapping
        
        INNOVATION AND STARTUP INTELLIGENCE:
        1. Emerging startups and innovation patterns
        2. Venture capital trends and funding analysis
        3. Disruptive technology identification
        
        ENTERPRISE ADOPTION ANALYSIS:
        1. Enterprise implementation strategies and challenges
        2. Market adoption curves and customer segments
        3. Business value proposition analysis
        
        QUANTITATIVE MARKET MODELING:
        1. Market size projections and growth modeling
        2. Technology adoption curve analysis
        3. Investment flow analysis and ROI projections
        
        DELIVERABLES:
        - Technology landscape map with innovation clusters
        - Market projections and strategic recommendations
        - Risk-adjusted opportunity assessment
        """

async def demonstrate_system():
    """Demonstrate the Multi-Model SerpAPI System"""
    
    # Check for API keys
    serpapi_key = os.environ.get("SERPAPI_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    
    if not serpapi_key:
        console.print("[yellow]Warning: No SERPAPI_API_KEY found. System will run with limited functionality.[/yellow]")
        console.print("To use SerpAPI features, set your API key:")
        console.print("export SERPAPI_API_KEY='your_key_here'")
    
    if not openrouter_key:
        console.print("[yellow]Warning: No OPENROUTER_API_KEY found. Using free models.[/yellow]")
        console.print("To use premium models via OpenRouter, set your API key:")
        console.print("export OPENROUTER_API_KEY='your_key_here'")
    
    # Initialize the system
    console.print("\n[bold cyan]🚀 Initializing Multi-Model SerpAPI System[/bold cyan]")
    orchestrator = MultiModelSerpAPISystem(
        serpapi_key=serpapi_key,
        context_window_size=32768
    )
    
    # Setup workers
    console.print("[bold]Setting up specialized workers...[/bold]")
    orchestrator.setup_multi_model_workers()
    
    # Display system configuration
    console.print("\n[bold green]✅ System Configuration:[/bold green]")
    console.print(f"  • Context Window: 32,768 tokens")
    console.print(f"  • Workers Created: {len(orchestrator.workers)}")
    console.print(f"  • SerpAPI Status: {'✅ Connected' if serpapi_key else '❌ Not configured'}")
    console.print(f"  • Model Provider: {'OpenRouter' if openrouter_key else 'Free HuggingFace'}")
    
    if orchestrator.workers:
        console.print("\n[bold]Available Specialists:[/bold]")
        for name in orchestrator.workers.keys():
            console.print(f"  • {name}")
    
    # Example 1: Simple search task
    console.print("\n" + "="*60)
    console.print("[bold cyan]Example 1: Simple Web Search[/bold cyan]")
    console.print("="*60)
    
    simple_task = "Search for the latest developments in AI agents and multi-agent systems"
    
    try:
        result = await orchestrator.execute_multi_model_workflow(
            simple_task,
            timeout=300  # 5 minutes
        )
        
        result_data = json.loads(result)
        console.print("\n[bold green]Results:[/bold green]")
        
        if 'error' in result_data:
            console.print(f"[red]Error: {result_data['error']}[/red]")
        else:
            if 'primary_result' in result_data:
                console.print(f"Primary Result: {str(result_data['primary_result'])[:500]}...")
            if 'search_performance' in result_data:
                console.print(f"\nSearch Performance: {result_data['search_performance']}")
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
    
    # Example 2: Market Analysis (if API keys are available)
    if serpapi_key or openrouter_key:
        console.print("\n" + "="*60)
        console.print("[bold cyan]Example 2: Market Opportunity Assessment[/bold cyan]")
        console.print("="*60)
        
        market_task = MultiModelWorkflowTemplates.advanced_market_opportunity_assessment(
            "AI-powered code generation tools",
            "North America"
        )
        
        try:
            result = await orchestrator.execute_multi_model_workflow(
                market_task,
                timeout=600  # 10 minutes
            )
            
            result_data = json.loads(result)
            console.print("\n[bold green]Market Analysis Results:[/bold green]")
            
            if 'error' in result_data:
                console.print(f"[red]Error: {result_data['error']}[/red]")
            else:
                console.print(json.dumps(result_data, indent=2)[:1000] + "...[truncated]")
        
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
    
    # System Summary
    console.print("\n" + "="*60)
    console.print("[bold cyan]System Performance Summary[/bold cyan]")
    console.print("="*60)
    
    summary = orchestrator.get_comprehensive_summary()
    console.print(json.dumps(summary, indent=2))
    
    console.print("\n[bold green]✅ Demonstration Complete![/bold green]")
    console.print("\nTo fully utilize this system:")
    console.print("1. Set SERPAPI_API_KEY for advanced search capabilities")
    console.print("2. Set OPENROUTER_API_KEY for premium model access")
    console.print("3. Customize workflow templates for your specific use cases")

def main():
    """Main entry point"""
    console.print("[bold magenta]Multi-Model SerpAPI Enhanced System Demo[/bold magenta]")
    console.print("="*60)
    
    try:
        asyncio.run(demonstrate_system())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {str(e)}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
