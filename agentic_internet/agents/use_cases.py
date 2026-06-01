"""Use-case recipes for K-LLM orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AgentMode = Literal["ToolCallingAgent", "CodeAgent", "BasicAgent"]
RoutingPolicy = Literal["coordinator_delegates", "parallel_then_synthesize"]


@dataclass(frozen=True)
class WorkerRecipe:
    """Worker role definition inside a use-case recipe."""

    name: str
    description: str
    model_role: str
    tool_bundles: tuple[str, ...] = ()
    agent_type: AgentMode = "ToolCallingAgent"
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Worker recipe requires a name")
        if not self.description:
            raise ValueError(f"Worker recipe {self.name!r} requires a description")
        if not self.model_role:
            raise ValueError(f"Worker recipe {self.name!r} requires a model role")


@dataclass(frozen=True)
class UseCaseRecipe:
    """Declarative recipe for coordinating K model workers plus tool bundles."""

    id: str
    description: str
    coordinator_model_role: str
    workers: tuple[WorkerRecipe, ...]
    direct_tool_bundles: tuple[str, ...] = ()
    routing_policy: RoutingPolicy = "coordinator_delegates"
    max_steps: int = 10
    timeout_seconds: float = 600
    output_contract: str = "Return a concise answer with evidence and caveats."
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Use-case recipe requires an id")
        if not self.workers:
            raise ValueError(f"Use-case recipe {self.id!r} requires at least one worker")
        names = [worker.name for worker in self.workers]
        if len(names) != len(set(names)):
            raise ValueError(f"Use-case recipe {self.id!r} has duplicate worker names")
        if self.max_steps < 1:
            raise ValueError(f"Use-case recipe {self.id!r} requires max_steps >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError(f"Use-case recipe {self.id!r} requires a positive timeout")

    @property
    def k(self) -> int:
        """Number of worker LLM roles in this use case."""
        return len(self.workers)

    def worker_names(self) -> tuple[str, ...]:
        """Return worker names in execution order."""
        return tuple(worker.name for worker in self.workers)


BUILT_IN_USE_CASES: dict[str, UseCaseRecipe] = {
    "research": UseCaseRecipe(
        id="research",
        description="General internet research with web, academic, commerce, and local search specialists.",
        coordinator_model_role="orchestrator",
        direct_tool_bundles=("web",),
        workers=(
            WorkerRecipe(
                name="search_researcher",
                description="Strategic web research specialist with cross-source validation.",
                model_role="search_researcher",
                tool_bundles=("web", "multi_engine"),
            ),
            WorkerRecipe(
                name="ecommerce_analyst",
                description="Commerce and pricing intelligence specialist.",
                model_role="ecommerce_analyst",
                tool_bundles=("shopping", "web"),
            ),
            WorkerRecipe(
                name="local_business_analyst",
                description="Local business and map-oriented intelligence specialist.",
                model_role="local_business_analyst",
                tool_bundles=("maps", "web"),
            ),
            WorkerRecipe(
                name="academic_researcher",
                description="Academic research and citation analysis specialist.",
                model_role="academic_researcher",
                tool_bundles=("scholar", "web"),
            ),
        ),
        output_contract="Return a sourced research synthesis with disagreements, confidence, and next steps.",
        tags=("search", "research", "serpapi"),
    ),
    "technical_due_diligence": UseCaseRecipe(
        id="technical_due_diligence",
        description="Technical assessment that combines web research, code analysis, and synthesis.",
        coordinator_model_role="orchestrator",
        direct_tool_bundles=("web",),
        workers=(
            WorkerRecipe(
                name="technical_researcher",
                description="Finds external technical context, documentation, and comparable projects.",
                model_role="search_researcher",
                tool_bundles=("web", "scraper"),
            ),
            WorkerRecipe(
                name="code_analyst",
                description="Analyzes code, data, and technical evidence with executable Python helpers.",
                model_role="agentic_coder",
                tool_bundles=("code_execution",),
                agent_type="CodeAgent",
            ),
            WorkerRecipe(
                name="risk_synthesizer",
                description="Synthesizes architecture, reliability, security, and delivery risks.",
                model_role="deep_reasoner",
                tool_bundles=(),
                agent_type="ToolCallingAgent",
            ),
        ),
        max_steps=12,
        output_contract="Return a due-diligence memo with strengths, risks, evidence, and recommended next actions.",
        tags=("technical", "code", "diligence"),
    ),
    "market_intelligence": UseCaseRecipe(
        id="market_intelligence",
        description="Market and business analysis with search, commerce, and local signals.",
        coordinator_model_role="orchestrator",
        direct_tool_bundles=("web",),
        workers=(
            WorkerRecipe(
                name="market_researcher",
                description="Collects broad market evidence and trend signals.",
                model_role="market_synthesizer",
                tool_bundles=("web", "multi_engine"),
            ),
            WorkerRecipe(
                name="commerce_analyst",
                description="Compares product, pricing, and availability signals.",
                model_role="ecommerce_analyst",
                tool_bundles=("shopping", "web"),
            ),
            WorkerRecipe(
                name="local_signal_analyst",
                description="Checks local business and geographic demand signals.",
                model_role="local_business_analyst",
                tool_bundles=("maps", "web"),
            ),
        ),
        output_contract="Return a market intelligence brief with segments, competitors, evidence, and opportunities.",
        tags=("market", "commerce", "business"),
    ),
}


def get_use_case_recipe(use_case_id: str) -> UseCaseRecipe:
    """Return a built-in use-case recipe by id."""
    normalized_id = use_case_id.strip().lower().replace("-", "_")
    try:
        return BUILT_IN_USE_CASES[normalized_id]
    except KeyError as exc:
        available = ", ".join(sorted(BUILT_IN_USE_CASES))
        raise ValueError(f"Unknown use case {use_case_id!r}. Available use cases: {available}") from exc


def list_use_case_recipes() -> list[UseCaseRecipe]:
    """List built-in use-case recipes in stable order."""
    return [BUILT_IN_USE_CASES[key] for key in sorted(BUILT_IN_USE_CASES)]
