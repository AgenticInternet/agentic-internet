"""
Search Orchestrator for coordinating multiple agents in web search tasks.
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
import threading

from smolagents import (
    ToolCallingAgent,
    CodeAgent,
    Tool
)
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..utils.model_utils import initialize_model

console = Console()


@dataclass
class SearchTask:
    """Represents a search task to be executed by an agent."""
    query: str
    task_id: str
    agent_name: str
    search_type: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SearchResult:
    """Represents the result from a search agent."""
    task_id: str
    agent_name: str
    result: Any
    success: bool
    execution_time: float
    error: Optional[str] = None


class SearchAgentWrapper:
    """Wrapper for individual search agents."""
    
    def __init__(self, name: str, agent: Union[ToolCallingAgent, CodeAgent], 
                 description: str = "", specialization: str = "general"):
        self.name = name
        self.agent = agent
        self.description = description
        self.specialization = specialization
        self.execution_count = 0
        self.success_count = 0
        
    def execute(self, task: SearchTask) -> SearchResult:
        """Execute a search task."""
        start_time = datetime.now()
        self.execution_count += 1
        
        try:
            # Customize the prompt based on specialization
            specialized_prompt = self._create_specialized_prompt(task)
            result = self.agent.run(specialized_prompt)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.success_count += 1
            
            return SearchResult(
                task_id=task.task_id,
                agent_name=self.name,
                result=result,
                success=True,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return SearchResult(
                task_id=task.task_id,
                agent_name=self.name,
                result=None,
                success=False,
                execution_time=execution_time,
                error=str(e)
            )
    
    def _create_specialized_prompt(self, task: SearchTask) -> str:
        """Create a specialized prompt based on the agent's role."""
        base_prompt = task.query
        
        if self.specialization == "news":
            return f"Search for recent news and updates about: {base_prompt}. Focus on the latest developments and breaking news."
        elif self.specialization == "academic":
            return f"Search for academic papers, research, and scholarly articles about: {base_prompt}. Include citations and credible sources."
        elif self.specialization == "technical":
            return f"Search for technical documentation, tutorials, and implementation details about: {base_prompt}. Focus on practical information."
        elif self.specialization == "market":
            return f"Search for market analysis, trends, and business information about: {base_prompt}. Include data and statistics where available."
        elif self.specialization == "comprehensive":
            return f"Conduct a comprehensive search about: {base_prompt}. Include multiple perspectives, sources, and detailed information."
        else:
            return base_prompt
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this agent."""
        return {
            "name": self.name,
            "specialization": self.specialization,
            "executions": self.execution_count,
            "successes": self.success_count,
            "success_rate": self.success_count / self.execution_count if self.execution_count > 0 else 0
        }


class SearchOrchestrator:
    """
    Orchestrates multiple search agents to perform comprehensive searches.
    """
    
    def __init__(self, 
                 max_workers: int = 3,
                 verbose: bool = True,
                 use_async: bool = False):
        """
        Initialize the Search Orchestrator.
        
        Args:
            max_workers: Maximum number of parallel agent executions
            verbose: Whether to print detailed output
            use_async: Whether to use async execution (experimental)
        """
        self.max_workers = max_workers
        self.verbose = verbose
        self.use_async = use_async
        self.agents: Dict[str, SearchAgentWrapper] = {}
        self.orchestrator_agent: Optional[ToolCallingAgent] = None
        self.execution_history: List[Dict[str, Any]] = []
        
    def add_agent(self, name: str, agent: Union[ToolCallingAgent, CodeAgent], 
                  description: str = "", specialization: str = "general"):
        """Add a search agent to the orchestrator."""
        wrapper = SearchAgentWrapper(name, agent, description, specialization)
        self.agents[name] = wrapper
        
        if self.verbose:
            console.print(f"[green]✓[/green] Added agent: {name} ({specialization})")
    
    def setup_default_agents(self, tools: List[Tool], model: Optional[Any] = None):
        """Set up a default set of search agents with different specializations."""
        
        # If no model provided, try to get a default one
        if model is None:
            model = self._get_default_model()
        
        if not model:
            console.print("[yellow]Warning: No model available. Cannot create agents.[/yellow]")
            return
        
        # Create specialized agents
        specializations = [
            ("news_researcher", "news", "Specializes in finding recent news and updates"),
            ("tech_researcher", "technical", "Specializes in technical documentation and tutorials"),
            ("general_researcher", "comprehensive", "Performs comprehensive general searches"),
        ]
        
        for name, spec, desc in specializations:
            try:
                agent = ToolCallingAgent(
                    tools=tools,
                    model=model,
                    max_steps=10
                )
                self.add_agent(name, agent, desc, spec)
            except Exception as e:
                if self.verbose:
                    console.print(f"[yellow]Could not create {name}: {e}[/yellow]")
        
        # Create the orchestrator agent that coordinates the workers
        try:
            self.orchestrator_agent = CodeAgent(
                tools=tools,
                model=model,
                max_steps=20,
                additional_authorized_imports=["json", "re", "datetime"]
            )
            if self.verbose:
                console.print("[green]✓[/green] Orchestrator agent created")
        except Exception as e:
            if self.verbose:
                console.print(f"[yellow]Could not create orchestrator: {e}[/yellow]")
    
    def _get_default_model(self, model_id: Optional[str] = None):
        """Get a default model using the centralized model initialization."""
        return initialize_model(model_id, verbose=self.verbose)
    
    def search(self, query: str, agents_to_use: Optional[List[str]] = None,
               parallel: bool = True) -> Dict[str, Any]:
        """
        Execute a search using multiple agents.
        
        Args:
            query: The search query
            agents_to_use: Specific agents to use (None = use all)
            parallel: Whether to run agents in parallel
            
        Returns:
            Dictionary containing aggregated results
        """
        if not self.agents:
            return {
                "error": "No agents available. Please add agents first.",
                "query": query
            }
        
        # Select agents to use
        selected_agents = agents_to_use or list(self.agents.keys())
        selected_agents = [a for a in selected_agents if a in self.agents]
        
        if not selected_agents:
            return {
                "error": "No valid agents selected",
                "query": query
            }
        
        if self.verbose:
            console.print(Panel.fit(
                f"[bold blue]Orchestrating search:[/bold blue] {query}\n"
                f"Using {len(selected_agents)} agents: {', '.join(selected_agents)}",
                title="Search Orchestration"
            ))
        
        # Create tasks
        tasks = []
        for agent_name in selected_agents:
            task = SearchTask(
                query=query,
                task_id=f"{agent_name}_{datetime.now().timestamp()}",
                agent_name=agent_name,
                metadata={"original_query": query}
            )
            tasks.append(task)
        
        # Execute tasks
        if parallel and len(tasks) > 1:
            results = self._execute_parallel(tasks)
        else:
            results = self._execute_sequential(tasks)
        
        # Aggregate results
        aggregated = self._aggregate_results(query, results)
        
        # Store in history
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "agents_used": selected_agents,
            "results": aggregated
        })
        
        return aggregated
    
    def _execute_parallel(self, tasks: List[SearchTask]) -> List[SearchResult]:
        """Execute tasks in parallel using ThreadPoolExecutor."""
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            
            task_id = progress.add_task(
                f"Running {len(tasks)} agents in parallel...", 
                total=len(tasks)
            )
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self.agents[task.agent_name].execute, task): task 
                    for task in tasks
                }
                
                for future in as_completed(future_to_task):
                    result = future.result()
                    results.append(result)
                    progress.advance(task_id)
                    
                    if self.verbose and result.success:
                        console.print(f"[green]✓[/green] {result.agent_name} completed")
                    elif self.verbose:
                        console.print(f"[red]✗[/red] {result.agent_name} failed: {result.error}")
        
        return results
    
    def _execute_sequential(self, tasks: List[SearchTask]) -> List[SearchResult]:
        """Execute tasks sequentially."""
        results = []
        
        for task in tasks:
            if self.verbose:
                console.print(f"Running {task.agent_name}...")
            
            result = self.agents[task.agent_name].execute(task)
            results.append(result)
            
            if self.verbose and result.success:
                console.print(f"[green]✓[/green] {result.agent_name} completed")
            elif self.verbose:
                console.print(f"[red]✗[/red] {result.agent_name} failed: {result.error}")
        
        return results
    
    def _aggregate_results(self, query: str, results: List[SearchResult]) -> Dict[str, Any]:
        """Aggregate results from multiple agents."""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        aggregated = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(results),
            "successful_agents": len(successful_results),
            "failed_agents": len(failed_results),
            "execution_times": {r.agent_name: r.execution_time for r in results},
            "agent_results": {},
            "failures": {}
        }
        
        # Add successful results
        for result in successful_results:
            agent = self.agents[result.agent_name]
            aggregated["agent_results"][result.agent_name] = {
                "specialization": agent.specialization,
                "result": str(result.result)[:1000] if result.result else None  # Truncate for readability
            }
        
        # Add failure information
        for result in failed_results:
            aggregated["failures"][result.agent_name] = result.error
        
        # If we have an orchestrator, use it to synthesize results
        if self.orchestrator_agent and successful_results:
            aggregated["synthesis"] = self._synthesize_results(query, successful_results)
        
        return aggregated
    
    def _synthesize_results(self, query: str, results: List[SearchResult]) -> str:
        """Use the orchestrator agent to synthesize results from multiple agents."""
        if not self.orchestrator_agent:
            return None
        
        # Prepare the synthesis prompt
        results_text = "\n\n".join([
            f"Results from {r.agent_name} ({self.agents[r.agent_name].specialization}):\n{str(r.result)[:500]}"
            for r in results if r.success
        ])
        
        synthesis_prompt = f"""
        Synthesize the following search results for the query: "{query}"
        
        {results_text}
        
        Provide a comprehensive summary that:
        1. Identifies common themes across results
        2. Highlights unique findings from each source
        3. Resolves any contradictions
        4. Provides a unified answer to the original query
        """
        
        try:
            if self.verbose:
                console.print("[blue]Synthesizing results...[/blue]")
            
            synthesis = self.orchestrator_agent.run(synthesis_prompt)
            return str(synthesis)
        except Exception as e:
            if self.verbose:
                console.print(f"[yellow]Could not synthesize results: {e}[/yellow]")
            return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get a performance report for all agents."""
        report = {
            "total_executions": len(self.execution_history),
            "agents": {}
        }
        
        for name, agent in self.agents.items():
            report["agents"][name] = agent.get_performance_stats()
        
        return report
    
    async def search_async(self, query: str, agents_to_use: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Async version of search (experimental).
        """
        if not self.use_async:
            return await asyncio.to_thread(self.search, query, agents_to_use)
        
        # Async implementation would go here
        # For now, just wrap the sync version
        return await asyncio.to_thread(self.search, query, agents_to_use, parallel=True)


def create_search_orchestrator(tools: List[Tool], model: Optional[Any] = None, 
                              verbose: bool = True) -> SearchOrchestrator:
    """
    Convenience function to create and set up a search orchestrator.
    
    Args:
        tools: List of tools to provide to agents
        model: Model to use for agents (None = auto-detect)
        verbose: Whether to print verbose output
        
    Returns:
        Configured SearchOrchestrator instance
    """
    orchestrator = SearchOrchestrator(verbose=verbose)
    orchestrator.setup_default_agents(tools, model)
    return orchestrator
