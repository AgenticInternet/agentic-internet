# Plan 004: Split multi-model orchestration into focused modules

> **Executor instructions**: Preserve public import paths and behavior. Extract cohesive classes into a new `agentic_internet.agents.multi_model` package and leave `multi_model_serpapi.py` as a compatibility re-export. Do not change model mappings, workflow logic, CLI behavior, dependencies, or lifecycle semantics. Run focused tests after every extraction and the full gate at the end.

## Status

- **Priority**: P2
- **Effort**: L
- **Risk**: MED (large import/re-export refactor)
- **Depends on**: plans/003-repository-hygiene.md (DONE)
- **Category**: tech-debt, architecture, tests
- **Planned at**: commit `7b761cb`, 2026-11-09

## Why this matters

`agentic_internet/agents/multi_model_serpapi.py` is approximately 1,400 lines and mixes model selection, result schemas, memory/context primitives, five SerpAPI tool classes, an agent wrapper, and the workflow system. Recent fixes added exact model routing, safe search memory, and code-execution settings, giving the extraction useful regression coverage. A compatibility-preserving split improves navigation without changing the public API or redesigning orchestration.

## Current class map

- `ModelManager` — model construction, exact central aliases, role defaults.
- `SearchResult`, `LocalResult`, `ShoppingResult`, `NewsResult`, `ScholarResult`, `ImageResult` — Pydantic result schemas.
- `ContextWindow`, `AgentMemory`, `TaskContext`, `ContextEngineeringMixin`, `NON_REUSABLE_SEARCH_PARAMS` — memory/context primitives and task tracking.
- `GoogleSearchTool`, `GoogleShoppingTool`, `GoogleMapsLocalTool`, `GoogleScholarTool`, `MultiEngineSearchTool`, `AgentTool` — tool adapters and worker-as-tool wrapper.
- `MultiModelSerpAPISystem` — recipe setup, worker construction, and workflow execution.

Existing tests import these classes from `agentic_internet.agents.multi_model_serpapi`. The compatibility module must keep those imports working. Tests that patch implementation globals must point at the new owning modules rather than the compatibility wrapper.

## Target structure

```text
agentic_internet/agents/multi_model/
├── __init__.py          # canonical re-exports
├── model_manager.py     # ModelManager
├── schemas.py           # six Pydantic result models
├── context.py           # ContextWindow, AgentMemory, TaskContext, ContextEngineeringMixin, NON_REUSABLE_SEARCH_PARAMS
├── tools.py             # five SerpAPI tools + AgentTool
└── system.py            # MultiModelSerpAPISystem

agentic_internet/agents/multi_model_serpapi.py  # thin compatibility re-export only
```

Use `git mv agentic_internet/agents/multi_model_serpapi.py agentic_internet/agents/multi_model/system.py` to preserve history, then extract the other classes to their owning modules and recreate the compatibility module at the original path.

## Import requirements

- `model_manager.py` may import smolagents, project settings, and standard library modules only.
- `context.py` may import dataclasses, datetime, typing, and `NON_REUSABLE_SEARCH_PARAMS` only; it must not import the workflow system.
- `tools.py` may import SerpAPI, smolagents, standard library modules, and context types only if actually needed. It must not import `MultiModelSerpAPISystem`.
- `system.py` imports its collaborators from the new package plus existing `basic_agent`, `orchestration_runtime`, `use_cases`, config, and tool modules. Do not introduce a cycle through the compatibility wrapper.
- `multi_model/__init__.py` and `multi_model_serpapi.py` re-export the original public classes. Keep `ModelManager`, `MultiModelSerpAPISystem`, all tool classes, all result schemas, `AgentTool`, and all context names available at the original path.
- Re-export external names used by existing consumers/tests (`LiteLLMModel`, `GoogleSearch`, `BingSearch`, `YahooSearch`, `BaiduSearch`, `GoogleScholarSearch`, `CodeAgent`, `ToolCallingAgent`, `Tool`) where practical, but update internal tests to patch the owning module's globals. Compatibility of import paths, not monkeypatch behavior, is required.

## Scope

**Create:**

- `agentic_internet/agents/multi_model/__init__.py`
- `agentic_internet/agents/multi_model/model_manager.py`
- `agentic_internet/agents/multi_model/schemas.py`
- `agentic_internet/agents/multi_model/context.py`
- `agentic_internet/agents/multi_model/tools.py`
- `agentic_internet/agents/multi_model/system.py`
- `tests/test_multi_model_package.py`

**Modify:**

- `agentic_internet/agents/multi_model_serpapi.py` to become only a compatibility re-export.
- Tests that patch `agentic_internet.agents.multi_model_serpapi.LiteLLMModel` or `.GoogleSearch`, especially `tests/test_model_manager.py` and `tests/test_serpapi_tools.py`, to patch the actual owning module.
- `docs/ARCHITECTURE.md`, `docs/CODEBASE_DIAGRAM.md`, `docs/QUALITY_SCORE.md`, and `README.md` only to document the package layout and code anchors.

**Out of scope:**

- Workflow, routing, model catalog, timeout, cancellation, memory semantics, CLI flags, dependency versions, and lockfile changes.
- CLI decomposition, `src/` layout, Browser Use, MCP, code execution, or new abstractions beyond this extraction.
- Main-checkout duplicate backups.

## Steps

### Step 1: Pin import compatibility before moving code

Create `tests/test_multi_model_package.py` with tests that import every original public class from both `agentic_internet.agents.multi_model` and `agentic_internet.agents.multi_model_serpapi`, and assert the objects are identical. Include at least `ModelManager`, `MultiModelSerpAPISystem`, `GoogleSearchTool`, `AgentTool`, `AgentMemory`, and one result schema. Run existing focused tests before extraction.

### Step 2: Extract schemas and context primitives

Move the six result schemas to `schemas.py` and context classes/constants to `context.py` without behavior changes. Add only necessary imports. Re-export through the new package and original compatibility module.

**Verify**: `uv run --frozen --no-sync pytest -p no:cacheprovider tests/test_context_engineering.py tests/test_serpapi_tools.py` passes.

### Step 3: Extract ModelManager and tool adapters

Move `ModelManager` to `model_manager.py` and the five SerpAPI tools plus `AgentTool` to `tools.py`. Preserve central exact aliases, safe parameter memory, and role defaults. Update tests to patch `agentic_internet.agents.multi_model.model_manager.LiteLLMModel` and `agentic_internet.agents.multi_model.tools.GoogleSearch` respectively.

**Verify**: `uv run --frozen --no-sync pytest -p no:cacheprovider tests/test_model_manager.py tests/test_serpapi_tools.py tests/test_context_engineering.py` passes.

### Step 4: Keep only the workflow system and compatibility re-exports

Leave `MultiModelSerpAPISystem` in `system.py`, importing collaborators from their new owning modules. Complete canonical re-exports and the original-path compatibility module. Do not leave duplicate class implementations.

**Verify**: `uv run --frozen --no-sync pytest -p no:cacheprovider tests/test_multi_model_package.py tests/test_orchestration_runtime.py tests/test_cli_use_cases.py tests/test_model_manager.py` passes and all compatibility imports resolve.

### Step 5: Update architecture references and run full verification

Update `docs/ARCHITECTURE.md`, `docs/CODEBASE_DIAGRAM.md`, `docs/QUALITY_SCORE.md`, and README structure to describe the package split without claiming orchestration was redesigned. Then run:

```bash
uv run --frozen --no-sync ruff check agentic_internet tests
uv run --frozen --no-sync ruff format --check agentic_internet tests
uv run --frozen --no-sync mypy agentic_internet
uv run --frozen --no-sync pytest -p no:cacheprovider
python3 .opencode/tools/golden_principles.py
uv lock --check
uv build
uv run --frozen --no-sync agentic-internet --help
```

Inspect the full diff for accidental behavior changes and confirm `uv.lock` is unchanged.

## Done criteria

- [ ] The target package contains each class exactly once and the original module is a thin re-export.
- [ ] All original public imports work and compatibility tests prove object identity across both paths.
- [ ] Full quality gates and package build pass; no dependency or lockfile changes.
- [ ] Existing SerpAPI, memory, exact-routing, recipe, and CLI tests pass without changing production behavior.
- [ ] Architecture docs and code anchors point to the new owning modules.

## STOP conditions

- An extraction introduces a circular import or changes a class's behavior/signature.
- Compatibility cannot be preserved without duplicating implementations.
- A test fails for a reason other than updating a patch target to the new owning module.
- The change requires dependency updates, lifecycle redesign, or out-of-scope files.

## Maintenance notes

- New model-routing behavior belongs in `model_manager.py`; search-pattern memory belongs in `context.py`; provider-specific tool adapters belong in `tools.py`; recipe execution belongs in `system.py`.
- Keep the original compatibility module until a deliberate major-version migration is announced.
- Execution lifecycle and timeout redesign must be reviewed separately after this structural change.
