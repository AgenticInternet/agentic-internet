---
type: API guide
title: Python API
description: Public package exports, subpackage extension points, result conventions, and complete change surfaces for library consumers.
tags: [python-api, public-api, extensions]
---

# Python API

The root package is a curated facade; concrete tools and recipe internals live in subpackages. `__version__` is `0.1.0`.

## Root exports

`agentic_internet/__init__.py` exports:

- `InternetAgent`, `ResearchAgent` and all five [Specialized Agents](../agents/specialized-agents.md)
- `SearchOrchestrator`, `create_search_orchestrator`
- `ToolFacade`, `create_code_mode_agent`
- global `settings`
- principal custom exceptions
- conditionally: `ModelManager`, `MultiModelSerpAPISystem`, five SerpAPI tool classes, and `MULTI_MODEL_AVAILABLE`

The multi-model import catches `ImportError`; unavailable names become `None`. Consumers must check `MULTI_MODEL_AVAILABLE`, not assume declared dependencies imported successfully.

```python
from agentic_internet import InternetAgent, ResearchAgent

agent = InternetAgent()
result = agent.run("Find and summarize current sources")
research = ResearchAgent().research("agent security", depth="deep")
```

Runtime results are often strings, including failures. `InternetAgent.run`, `BasicAgent.run`, provider tools, worker wrappers, and multi-model execution convert exceptions into error text/JSON text. Construction/configuration errors such as `ModelInitializationError` can still raise. Do not use type/exit status alone as success evidence.

## Subpackage surfaces

`agentic_internet.agents` additionally exports `BasicAgent`, recipe dataclasses/lookups, and core factories. Runtime bundle helpers and many K-LLM context/result classes require direct module imports. `agentic_internet.tools` exports all concrete web/Exa/browser/code tools and optional MCP APIs; root does not. `agentic_internet.config` exposes settings classes/singleton. `agentic_internet.utils` exposes model initialization helpers but not live OpenRouter helpers.

For exact behavior follow [Internet and Research Agents](../agents/internet-and-research.md), [Search Orchestrator](../orchestration/search-orchestrator.md), [Code Mode](../orchestration/code-mode.md), [K-LLM Use Cases](../orchestration/k-llm-use-cases.md), and the individual tool pages.

## Exception taxonomy

`exceptions.py` defines the project hierarchy for package/config/model/tool/search/browser/code/MCP failures. Many adapters catch these or broader exceptions and return strings, so exceptions mainly describe construction/validation boundaries and direct helper use. Preserve typed exceptions where callers can recover before execution; preserve documented string contracts where smolagents expects tool output.

## Public extension change surfaces

### Add a built-in tool

1. Implement a `smolagents.Tool` with stable `name`, `description`, `inputs`, `output_type`, and `forward` behavior.
2. Export it from `agentic_internet/tools/__init__.py`.
3. If default, add feature/key gating in `InternetAgent._get_default_tools`.
4. If recipe-addressable, construct it in `MultiModelSerpAPISystem.create_use_case_tool_inventory` and map its exact name in `TOOL_BUNDLES`.
5. Document the consumer import path and add schema, success, unavailable/error, registration, and security-boundary tests.

### Add an agent facade

Implement/subclass under `agents`, export from `agents/__init__.py`, add root export if broadly public, preserve constructor override behavior, and test malformed/failure outputs as well as happy metadata.

### Add a K-LLM recipe

Update the built-in registry and bundle inventory, then test normalization/listing, missing tools, model overrides/type selection, coordinator behavior, and CLI discoverability. See the detailed [recipe procedure](../orchestration/k-llm-use-cases.md#adding-a-use-case).

### Add a CLI route

Keep behavior in package modules, register on `app` or `mcp_app`, add CLI runner tests for validation/routing/output/exit status, and update [CLI](cli.md). The installed console script and module entrypoints already share `cli:app`/`main`.

## Validation

Package export behavior has focused coverage mainly for MCP and conditional surfaces. Run the implementation’s focused test plus `uv run python -m agentic_internet.cli --help`, `uv run pytest`, and `uv run ruff check agentic_internet tests` for public-surface changes. Build with `uv build` when packaging/export metadata changes.