"""Runtime helpers for use-case recipe orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from smolagents import Tool

from .use_cases import UseCaseRecipe

logger = logging.getLogger(__name__)

TOOL_BUNDLES: dict[str, tuple[str, ...]] = {
    "web": ("google_search", "multi_engine_search", "web_search", "news_search", "exa_search"),
    "scraper": ("web_scraper", "structured_browser_use", "browser_use"),
    "multi_engine": ("multi_engine_search", "google_search", "web_search"),
    "shopping": ("google_shopping", "google_search", "web_search"),
    "maps": ("google_maps_local", "google_search", "web_search"),
    "scholar": ("google_scholar", "google_search", "web_search"),
    "code_execution": ("python_executor", "data_analysis"),
    "browser": ("browser_use", "async_browser_use", "structured_browser_use"),
}


@dataclass(frozen=True)
class ResolvedUseCaseTools:
    """Resolved tool assignments for a use-case recipe."""

    direct_tools: list[Tool]
    worker_tools: dict[str, list[Tool]]
    missing_bundles: tuple[str, ...]

    def tool_names_for_worker(self, worker_name: str) -> tuple[str, ...]:
        """Return resolved tool names for a worker."""
        return tuple(tool.name for tool in self.worker_tools.get(worker_name, []))


def parse_worker_model_overrides(raw_overrides: list[str] | None) -> dict[str, str]:
    """Parse CLI worker overrides in the form worker=model."""
    parsed: dict[str, str] = {}
    for raw_override in raw_overrides or []:
        if "=" not in raw_override:
            raise ValueError(f"Worker model override must use worker=model format: {raw_override!r}")
        worker, model = raw_override.split("=", 1)
        worker = worker.strip()
        model = model.strip()
        if not worker or not model:
            raise ValueError(f"Worker model override must include both worker and model: {raw_override!r}")
        parsed[worker] = model
    return parsed


def resolve_tool_bundles(
    available_tools: list[Tool], bundle_names: tuple[str, ...]
) -> tuple[list[Tool], tuple[str, ...]]:
    """Resolve bundle names into available tools while reporting missing bundles."""
    tools_by_name = {tool.name: tool for tool in available_tools}
    resolved: list[Tool] = []
    seen_tool_names: set[str] = set()
    missing_bundles: list[str] = []

    for bundle_name in bundle_names:
        expected_tool_names = TOOL_BUNDLES.get(bundle_name)
        if expected_tool_names is None:
            missing_bundles.append(bundle_name)
            continue

        matched_bundle = any(tool_name in tools_by_name for tool_name in expected_tool_names)
        for tool_name in expected_tool_names:
            tool = tools_by_name.get(tool_name)
            if tool is None or tool.name in seen_tool_names:
                continue
            resolved.append(tool)
            seen_tool_names.add(tool.name)

        if not matched_bundle:
            missing_bundles.append(bundle_name)

    return resolved, tuple(missing_bundles)


def resolve_use_case_tools(recipe: UseCaseRecipe, available_tools: list[Tool]) -> ResolvedUseCaseTools:
    """Resolve direct and worker tool assignments for a recipe."""
    direct_tools, missing = resolve_tool_bundles(available_tools, recipe.direct_tool_bundles)
    missing_bundles = list(missing)
    worker_tools: dict[str, list[Tool]] = {}

    for worker in recipe.workers:
        resolved_tools, worker_missing = resolve_tool_bundles(available_tools, worker.tool_bundles)
        worker_tools[worker.name] = resolved_tools
        missing_bundles.extend(f"{worker.name}:{bundle}" for bundle in worker_missing)

    deduped_missing = tuple(dict.fromkeys(missing_bundles))
    if deduped_missing:
        logger.warning("use_case_missing_tool_bundles", extra={"use_case": recipe.id, "bundles": deduped_missing})

    return ResolvedUseCaseTools(direct_tools=direct_tools, worker_tools=worker_tools, missing_bundles=deduped_missing)


def summarize_use_case(recipe: UseCaseRecipe, resolved_tools: ResolvedUseCaseTools) -> dict[str, Any]:
    """Return a JSON-serializable summary of a prepared use case."""
    return {
        "id": recipe.id,
        "description": recipe.description,
        "k": recipe.k,
        "coordinator_model_role": recipe.coordinator_model_role,
        "routing_policy": recipe.routing_policy,
        "workers": [
            {
                "name": worker.name,
                "model_role": worker.model_role,
                "agent_type": worker.agent_type,
                "tool_bundles": list(worker.tool_bundles),
                "resolved_tools": list(resolved_tools.tool_names_for_worker(worker.name)),
            }
            for worker in recipe.workers
        ],
        "direct_tools": [tool.name for tool in resolved_tools.direct_tools],
        "missing_bundles": list(resolved_tools.missing_bundles),
        "output_contract": recipe.output_contract,
    }
