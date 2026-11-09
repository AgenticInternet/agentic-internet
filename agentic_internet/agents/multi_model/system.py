"""
SerpAPI Enhanced Context-Engineered AgenticInternet System
Multi-model orchestration with specialized model assignment for optimal performance
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Union

from dotenv import load_dotenv
from smolagents import CodeAgent, Tool, ToolCallingAgent

from agentic_internet.config.settings import settings
from agentic_internet.tools.code_execution import DataAnalysisTool, PythonExecutorTool
from agentic_internet.tools.web_search import NewsSearchTool, WebScraperTool, WebSearchTool

from ..basic_agent import BasicAgent
from ..orchestration_runtime import ResolvedUseCaseTools, resolve_use_case_tools, summarize_use_case
from ..use_cases import UseCaseRecipe, get_use_case_recipe
from .context import ContextEngineeringMixin, ContextWindow, TaskContext
from .model_manager import ModelManager
from .tools import (
    AgentTool,
    GoogleMapsLocalTool,
    GoogleScholarTool,
    GoogleSearchTool,
    GoogleShoppingTool,
    MultiEngineSearchTool,
)

load_dotenv()

logger = logging.getLogger(__name__)


class MultiModelSerpAPISystem(ContextEngineeringMixin):
    """
    Enhanced orchestrator with strategic multi-model assignment and SerpAPI integration
    """

    def __init__(self, serpapi_key: str | None = None, context_window_size: int = 16384):
        # Initialize context engineering
        super().__init__()
        self.context_window = ContextWindow(max_tokens=context_window_size)

        # Initialize model manager
        self.model_manager = ModelManager()

        # Initialize SerpAPI
        self.serpapi_key = serpapi_key or os.environ.get("SERPAPI_API_KEY")

        self.workers: dict[str, Union[CodeAgent, ToolCallingAgent]] = {}
        self.worker_tools: dict[str, AgentTool] = {}  # Store wrapped agents as tools
        self.model_performance_tracker: dict[str, Any] = {}
        self.active_use_case_recipe: UseCaseRecipe | None = None
        self.active_use_case_tools: ResolvedUseCaseTools | None = None

    def create_serpapi_tools(self) -> list[Tool]:
        """Create comprehensive SerpAPI tool suite"""
        if not self.serpapi_key:
            logger.warning("No SerpAPI key provided. Search tools will be limited.")
            return []

        return [
            GoogleSearchTool(self.serpapi_key, self),
            GoogleShoppingTool(self.serpapi_key, self),
            GoogleMapsLocalTool(self.serpapi_key, self),
            GoogleScholarTool(self.serpapi_key, self),
            MultiEngineSearchTool(self.serpapi_key, self),
        ]

    def create_use_case_tool_inventory(self) -> list[Tool]:
        """Create the tool inventory that use-case recipes can resolve against."""
        tools: list[Tool] = [
            WebSearchTool(),
            WebScraperTool(),
            NewsSearchTool(),
        ]
        if settings.tools.code_execution_enabled:
            tools.extend([PythonExecutorTool(), DataAnalysisTool()])
        tools.extend(self.create_serpapi_tools())
        return tools

    def _check_model_tool_support(self, model) -> bool:
        """Check if a model supports tool calling"""
        # Models known not to support tool calling through OpenRouter
        # IMPORTANT: Perplexity models technically support tools via LiteLLM,
        # but OpenRouter cannot route them with tool use (returns 404 error)
        non_tool_models = [
            "perplexity/sonar",
            "perplexity/sonar-reasoning",
            "perplexity/llama",  # Any Perplexity models
        ]

        # Check if model has a model_id attribute and if it contains non-tool model names
        if hasattr(model, "model_id"):
            model_id = str(model.model_id).lower()
            for non_tool_model in non_tool_models:
                if non_tool_model in model_id:
                    return False

        # By default, assume the model supports tools
        # Most models through LiteLLM/OpenRouter support tool calling
        return True

    def _check_model_code_support(self, model) -> bool:
        """Check if a model supports code generation for CodeAgent"""
        # Models that are good at code generation
        code_capable_models = ["claude", "gpt", "deepseek", "mistral", "gemini", "qwen"]

        if hasattr(model, "model_id"):
            model_id = str(model.model_id).lower()
            for code_model in code_capable_models:
                if code_model in model_id:
                    return True

        # Models like Perplexity are optimized for search/QA, not code generation
        return False

    def create_specialized_worker(
        self,
        name: str,
        description: str,
        tools: list[Tool],
        agent_type: str = "ToolCallingAgent",
        model_override: str | None = None,
    ) -> None:
        """Create worker with optimal model selection and enhanced capabilities"""

        # Get optimal model for this role
        model = self.model_manager.get_model_for_role(model_override or name)

        if model is None:
            logger.warning("No model available for %s. Skipping worker creation.", name)
            return

        model_info = self.model_manager.get_model_capabilities()
        model_name = model_override or name
        capabilities = model_info.get(model_name, {})

        # Enhanced context with model-specific optimization
        model_context = f"""
SPECIALIZED MODEL ASSIGNMENT:
Model strengths: {", ".join(capabilities.get("strengths", ["general"]))}

SERPAPI MULTI-ENGINE CAPABILITIES:
- Google Web Search: Comprehensive with location/language/date filters
- Google Shopping: Product search with price/review filtering
- Google Maps/Local: Business search with ratings/reviews
- Google Scholar: Academic papers and citation analysis
- Multi-Engine Search: Cross-platform comparison and validation

CONTEXT-AWARE OPTIMIZATION:
1. Learn from previous successful search patterns
2. Adapt search parameters based on query type and past performance
3. Use cross-engine validation for critical information
4. Track and optimize search performance metrics
5. Synthesize results with model-specific analytical strengths
"""

        base_context = self.prepare_agent_context("initialization", name)
        enhanced_description = f"""
{description}

{model_context}

CONTEXT AWARENESS:
{base_context}

MODEL-OPTIMIZED APPROACH:
- Leverage your specific model strengths: {", ".join(capabilities.get("strengths", []))}
- Focus on tasks you excel at: {capabilities.get("best_for", "general tasks")}
- Coordinate with other specialized agents for comprehensive results
- Provide analysis depth appropriate to your model capabilities
"""

        try:
            # Create prompt templates with the enhanced description as system prompt
            from smolagents.agents import (
                FinalAnswerPromptTemplate,
                ManagedAgentPromptTemplate,
                PlanningPromptTemplate,
                PromptTemplates,
            )

            # Create empty sub-templates with required keys
            planning_template = PlanningPromptTemplate(
                initial_plan=None, update_plan_pre_messages=None, update_plan_post_messages=None
            )

            managed_template = ManagedAgentPromptTemplate(task=None, report=None)

            final_template = FinalAnswerPromptTemplate(pre_messages=None, post_messages=None)

            # Create full prompt templates
            prompt_templates = PromptTemplates(
                system_prompt=enhanced_description,
                planning=planning_template,
                managed_agent=managed_template,
                final_answer=final_template,
            )

            # Check model capabilities
            model_supports_tools = self._check_model_tool_support(model)
            model_supports_code = self._check_model_code_support(model)

            if agent_type == "CodeAgent" and model_supports_code:
                # Use CodeAgent for models that support code generation
                agent = CodeAgent(
                    tools=tools,
                    model=model,
                    prompt_templates=prompt_templates,
                    max_steps=20,
                    additional_authorized_imports=[
                        "pandas",
                        "numpy",
                        "json",
                        "csv",
                        "re",
                        "datetime",
                        "time",
                        "requests",
                        "urllib",
                        "math",
                        "statistics",
                    ],
                )
            elif model_supports_tools:
                # Use ToolCallingAgent for models that support tool calling
                agent = ToolCallingAgent(tools=tools, model=model, prompt_templates=prompt_templates)
            else:
                # Use BasicAgent for models that don't support tools or code
                agent = BasicAgent(model=model, tools=tools, prompt_templates=prompt_templates)
                logger.info("Model for %s does not support tools or code generation, using BasicAgent", name)

            # Store both the raw agent and create an AgentTool wrapper
            self.workers[name] = agent

            # Create AgentTool wrapper for orchestrator use
            agent_tool = AgentTool(agent=agent, agent_name=name, agent_description=description)
            self.worker_tools[name] = agent_tool

        except Exception as e:
            logger.error("Failed to create specialized worker %s: %s", name, e)

    def setup_multi_model_workers(self, default_model: str | None = None):
        """Setup workers with strategic model assignments

        Args:
            default_model: Model to use for all workers (if not specified, uses role-based assignment)
        """
        self.setup_use_case_workers("research", default_model=default_model)

    def setup_use_case_workers(
        self,
        use_case_id: str = "research",
        default_model: str | None = None,
        worker_model_overrides: dict[str, str] | None = None,
    ) -> UseCaseRecipe:
        """Setup workers from a declarative K-LLM use-case recipe."""
        recipe = get_use_case_recipe(use_case_id)
        tool_inventory = self.create_use_case_tool_inventory()
        resolved_tools = resolve_use_case_tools(recipe, tool_inventory)

        self.workers = {}
        self.worker_tools = {}
        self.active_use_case_recipe = recipe
        self.active_use_case_tools = resolved_tools

        overrides = worker_model_overrides or {}
        for worker in recipe.workers:
            model_override = overrides.get(worker.name) or default_model or worker.model_role
            self.create_specialized_worker(
                name=worker.name,
                description=worker.description,
                tools=resolved_tools.worker_tools.get(worker.name, []),
                agent_type=worker.agent_type,
                model_override=model_override,
            )

        if not self.workers:
            logger.warning("No workers were created for use case %s", recipe.id)

        return recipe

    def get_active_use_case_summary(self) -> dict[str, Any] | None:
        """Return a summary for the active use-case setup."""
        if not self.active_use_case_recipe or not self.active_use_case_tools:
            return None
        return summarize_use_case(self.active_use_case_recipe, self.active_use_case_tools)

    async def execute_multi_model_workflow(
        self,
        task: str,
        timeout: float | None = None,
        orchestrator_model: str = "orchestrator",
        use_case_id: str = "research",
    ) -> str:
        """Execute workflow with multi-model coordination and optimization"""

        # Initialize enhanced task context
        task_id = f"multimodel_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_task_context = TaskContext(task_id=task_id, objective=task, total_steps=10)

        recipe = self.active_use_case_recipe or get_use_case_recipe(use_case_id)
        if not self.workers:
            # Use the same model for all workers as the orchestrator for consistency
            recipe = self.setup_use_case_workers(use_case_id, default_model=orchestrator_model)

        if not self.workers:
            return json.dumps({"error": "No workers available. Please check API keys and configuration.", "task": task})

        try:
            # Get optimal orchestrator model
            orchestrator_model_instance = self.model_manager.get_model_for_role(orchestrator_model)

            if orchestrator_model_instance is None:
                return json.dumps({"error": "No orchestrator model available. Please check API keys.", "task": task})

            # Prepare multi-model orchestrator context
            orchestrator_context = self.prepare_agent_context(task, "orchestrator")

            recipe_workers = [worker for worker in recipe.workers if worker.name in self.workers]
            available_specialists = "\n".join([f"- {worker.name}: {worker.description}" for worker in recipe_workers])
            usage_examples = "\n".join(
                [f'- {worker.name}(task="...") - {worker.description}' for worker in recipe_workers]
            )

            # Enhanced orchestrator with multi-model awareness
            orchestrator_prompt = f"""
You are a CODE-BASED ORCHESTRATOR managing a team of AI specialist agents through Python code.

{orchestrator_context}

USE CASE:
- id: {recipe.id}
- description: {recipe.description}
- routing policy: {recipe.routing_policy}
- output contract: {recipe.output_contract}

AVAILABLE SPECIALIST AGENT TOOLS (call them as functions):
{available_specialists}

HOW TO USE THE AGENTS:
Each specialist agent is available as a tool that can be called like a function:
{usage_examples}

IMPORTANT INSTRUCTIONS:
1. You MUST write Python code to call these agent tools
2. Once you have obtained a satisfactory answer to the user's question, call final_answer() to complete
3. DO NOT continue generating completion messages after calling final_answer()
4. Use this exact code pattern to signal completion:
   ```python
   final_answer("Your complete answer here")
   ```

Example:
```python
result = {recipe_workers[0].name if recipe_workers else "search_researcher"}(task="Research the most important evidence")
print(result)
final_answer("Complete answer that follows the use-case output contract")
```

CURRENT TASK CONTEXT:
{self.current_task_context.get_progress_summary()}

Write Python code to execute the task. Remember to STOP after providing the final answer.
"""

            # Create multi-model orchestrator with both SerpAPI tools and wrapped agent tools
            orchestrator_tools = list(self.active_use_case_tools.direct_tools) if self.active_use_case_tools else []

            # Add the wrapped worker agents as tools for the orchestrator
            orchestrator_tools.extend(list(self.worker_tools.values()))

            # Create prompt templates for orchestrator
            from smolagents.agents import (
                FinalAnswerPromptTemplate,
                ManagedAgentPromptTemplate,
                PlanningPromptTemplate,
                PromptTemplates,
            )

            # Create sub-templates for orchestrator
            orchestrator_planning = PlanningPromptTemplate(
                initial_plan=None, update_plan_pre_messages=None, update_plan_post_messages=None
            )

            orchestrator_managed = ManagedAgentPromptTemplate(task=None, report=None)

            orchestrator_final = FinalAnswerPromptTemplate(pre_messages=None, post_messages=None)

            orchestrator_templates = PromptTemplates(
                system_prompt=orchestrator_prompt,
                planning=orchestrator_planning,
                managed_agent=orchestrator_managed,
                final_answer=orchestrator_final,
            )

            orchestrator = CodeAgent(
                tools=orchestrator_tools,
                model=orchestrator_model_instance,
                prompt_templates=orchestrator_templates,
                additional_authorized_imports=[
                    "pandas",
                    "numpy",
                    "json",
                    "csv",
                    "re",
                    "datetime",
                    "time",
                    "requests",
                    "urllib",
                    "math",
                    "statistics",
                ],
                max_steps=recipe.max_steps,
            )

            # Execute with comprehensive tracking
            effective_timeout = timeout if timeout is not None else recipe.timeout_seconds
            if effective_timeout:
                result = await asyncio.wait_for(asyncio.to_thread(orchestrator.run, task), timeout=effective_timeout)
            else:
                result = await asyncio.to_thread(orchestrator.run, task)

            # Enhanced result processing
            final_result = {
                "primary_result": result,
                "use_case": self.get_active_use_case_summary(),
                "search_performance": self.analyze_search_performance(),
                "cross_engine_analysis": self._analyze_cross_engine_results(),
            }

            # Update memory with results
            self.update_context_from_result("multi_model_orchestrator", task, result, True)

            return json.dumps(final_result, indent=2, default=str)

        except TimeoutError:
            error_msg = f"Multi-model workflow execution exceeded {timeout} seconds"
            self.update_context_from_result("multi_model_orchestrator", task, error_msg, False)
            return json.dumps({"error": error_msg, "task": task})
        except Exception as e:
            error_msg = f"Multi-model workflow execution failed: {e!s}"
            self.update_context_from_result("multi_model_orchestrator", task, error_msg, False)
            return json.dumps({"error": error_msg, "task": task})
