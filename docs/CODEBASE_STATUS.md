# Codebase Status

Status date: 2026-06-01

This map captures the current repository state, recent committed updates, unfinished local work, and external ecosystem changes that should drive the next implementation pass.

## Evidence Sources

- Local repository: `git log`, `git status`, `git diff`, `docs/plans/`, `docs/exec-plans/`, `agentic_internet/config/settings.py`, `agentic_internet/utils/model_utils.py`.
- OpenRouter: [List all models and their properties](https://openrouter.ai/docs/api/api-reference/models/get-models) and [`/api/v1/models`](https://openrouter.ai/api/v1/models), checked 2026-05-31.
- SmolAgents: [Guided tour](https://huggingface.co/docs/smolagents/guided_tour), [Tools tutorial](https://huggingface.co/docs/smolagents/tutorials/tools), [GitHub releases](https://github.com/huggingface/smolagents/releases), and Context7 `/huggingface/smolagents`, checked 2026-05-31.

## Current Shape

```mermaid
flowchart TD
    CLI["CLI commands"]
    Config["Settings and model catalog"]
    ModelUtils["Model initialization"]
    Agents["InternetAgent, ResearchAgent, specialized agents"]
    Tools["Search, browser, code execution, MCP, Exa"]
    Plans["Plans, specs, harness docs"]
    External["OpenRouter and SmolAgents ecosystem"]

    CLI --> Agents
    CLI --> Tools
    Agents --> ModelUtils
    ModelUtils --> Config
    Agents --> Tools
    Tools --> External
    Config -. "stale snapshot risk" .-> External
    Plans -. "tracks unfinished work" .-> CLI
    Plans -. "tracks unfinished work" .-> Agents
```

## Recent Committed Updates

| Commit | Area | What Changed | Follow-up |
|--------|------|--------------|-----------|
| `3ed2908` | Docs | Required Mermaid for architecture, flow, lifecycle, and system diagrams | Keep all future diagrams in Mermaid |
| `0e8ea39` | Harness | Added AGENTS map, docs knowledge base, `.opencode`, CI skeleton, golden-principles check | Finish first real feature exec-plan |
| `45d205a` | Search | Merged Exa search support | Verify Exa docs, tests, and env docs are complete |
| `a4c1a7c` | Search | Added Exa AI-powered search tool | Ensure `EXA_API_KEY` is represented in `.env.example` if runtime uses it |
| `49cfcac` | Specs | Added codebase specs | Migrate durable architecture flow into Mermaid docs where still useful |
| `5e8f43f` | Research output | Avoided duplicate research output | Add regression coverage if missing |
| `819a7ef` | MCP | Normalized MCP HTTP URLs without endpoints | Keep aligned with SmolAgents streamable HTTP guidance |
| `b617ea2` | MCP | Added MCP feature with SmolAgents | Recheck against current `MCPClient` structured-output support |
| `c9f5ec9` | Models | Updated/fixed model selection | Now stale against May 2026 OpenRouter inventory |

## Unfinished Work

| Work | Evidence | Status | Next Action |
|------|----------|--------|-------------|
| Code Mode MCP agent | `agentic_internet/agents/code_mode.py`, `agentic_internet/cli.py`, exports, focused tests, CLI smoke test, and completed exec-plan | Implemented in current pass | Add live MCP integration smoke test when a local server fixture is available |
| Code Mode acceptance criteria | `tests/test_code_mode.py` covers facade, meta-tools, factory, and E2B fallback; `tests/test_cli_mcp.py` covers `mcp run --agent-type code --structured-output` routing | Covered by focused tests | Add a real local MCP server fixture for process-level smoke coverage |
| Model catalog freshness | `settings.py` refreshed and `agentic_internet/utils/openrouter_models.py` added | Dynamic live listing available via `models --live` | Add cached snapshot writing command if offline catalog automation is needed |
| Initial harness setup | `docs/exec-plans/completed/2026-05-30-initial-setup.md` is complete | Closed | Keep future completed plans under `docs/exec-plans/completed/` |
| Local quality gates | `uv`, `ruff`, `mypy`, and `pytest` run via local PATH setup; `bd` is still missing from PATH | Mostly unblocked | Restore `bd` so bead sync hooks run during commit and push |
| Legacy plan location | Completed Code Mode plan migrated to `docs/exec-plans/completed/` | Resolved for current pass | Keep future plans in `docs/exec-plans/active/` or `docs/exec-plans/completed/` |

## External Findings

### OpenRouter

OpenRouter's model endpoint reports 354 models as of 2026-05-31. The newest agent-relevant models include:

- `stepfun/step-3.7-flash`
- `anthropic/claude-opus-4.8` and `anthropic/claude-opus-4.8-fast`
- `qwen/qwen3.7-max`
- `x-ai/grok-build-0.1`
- `google/gemini-3.5-flash`
- `google/gemini-3.1-flash-lite`
- `openai/gpt-chat-latest`

The repository currently stores OpenRouter models as LiteLLM-style IDs prefixed with `openrouter/`. OpenRouter's API returns provider IDs without that prefix, and `agentic_internet/utils/model_utils.py` adds `openrouter/` when needed. Keep that distinction explicit in any generated catalog.

See [OpenRouter model snapshot](references/openrouter-models-2026-05-31.md).

### SmolAgents

Current SmolAgents docs and releases emphasize:

- `CodeAgent` for expressive Python-based tool composition.
- `ToolCallingAgent` for predictable JSON tool calls.
- `MCPClient` and `ToolCollection.from_mcp()` for stdio and streamable HTTP MCP servers.
- `structured_output=True` for MCP tool output schemas and structured content.
- Stronger executor security guidance: local execution is not a security boundary; prefer remote/sandbox executors for untrusted code.
- Recent releases include Exa as a `WebSearchTool` engine, executor hardening, final-answer checks, GPT-5.2 support adjustments, and model parameter handling improvements.

See [SmolAgents feature snapshot](references/smolagents-features-2026-05-31.md).

## Recommended Next Pass

1. Add cached snapshot writing for OpenRouter if offline catalog automation is needed.
2. Add a local MCP server fixture for a real process-level `mcp run --agent-type code --structured-output` smoke test.
3. Restore local tooling availability (`uv`, `bd`, and `mypy`) so release gates can run completely.
4. Clean up existing Ruff failures under `agentic_internet/examples/`.
