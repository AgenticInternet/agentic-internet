---
date: 2026-02-23
topic: code-mode-mcp-agent
status: brainstorm
---

# Code Mode Agent with MCP and Tool Facade

## What We're Building

A `create_code_mode_agent()` factory in `agents/code_mode.py` that constructs a
`smolagents.CodeAgent` where **all tools** (MCP-sourced + default) are accessible
through a single `ToolFacade` object (`api`). The agent writes Python code to
interact with tools rather than calling them one-at-a-time via tool-calling.

Two meta-tools are injected into the agent:
- `search(query)` — discovers tools by name/description keyword
- `execute(code)` — runs arbitrary Python against the `api` facade object

The CLI's `mcp run` command gains an `--agent-type` flag; when set to `code`,
it routes through this factory instead of the default `ToolCallingAgent`.

## Why This Approach

Three approaches were considered:

| Approach | Decision |
|----------|----------|
| **A: Thin factory** (chosen) | Standalone module, zero changes to `InternetAgent`, no regression risk |
| B: Extend `InternetAgent` | Adds `facade_mode` flag to an already-central class — complexity without proportional benefit |
| C: `CodeModeAgent` subclass | Inheriting `_get_default_tools()` override is surprising; requires threading `executor_type` through `InternetAgent._create_agent()` (a gap) |

Approach A ships faster, is independently testable, and follows the same
"standalone module" pattern as `MultiModelSerpAPISystem`.

## Key Decisions

- **Facade scope**: All tools are included — both MCP-sourced tools and existing
  default tools (WebSearchTool, BrowserUseTool, etc.) — so the agent has one
  unified `api` object for everything.

- **Executor default**: `executor_type="local"` on day one. E2B support is
  configurable via parameter (pass `executor_type="e2b"` + `E2B_API_KEY`) but
  not the default. The `to_dict()` serialization workaround from the prototype
  will be preserved for E2B compatibility.

- **File location**: `agentic_internet/agents/code_mode.py` — new file, peer to
  `internet_agent.py` and `specialized_agents.py`.

- **CLI integration**: `mcp run` gets `--agent-type [tool_calling|code]` flag
  (default: `tool_calling` to preserve existing behavior). When `code` is
  selected, the command calls `create_code_mode_agent(tools=list(mcp_tools))`.

- **No changes to `InternetAgent`**: The `max_steps` inconsistency bug
  (ToolCallingAgent missing it) and the absent `executor_type` support are
  documented gaps but left for a separate fix to keep this PR focused.

- **`add_base_tools=True`**: The CodeAgent should include smolagents' built-in
  tools (e.g., `final_answer`) in addition to the facade tools.

## Module Interface (sketch — not implementation)

```python
# agents/code_mode.py

def create_code_mode_agent(
    tools: list[Tool],
    model_id: str = "openrouter/anthropic/claude-opus-4.5",
    verbosity_level: int = 2,
    max_steps: int = 25,
    executor_type: str = "local",  # or "e2b"
    **kwargs,
) -> CodeAgent:
    ...
```

## Open Questions

- **MCP context manager lifetime**: `mcp_tools()` is a context manager — the
  caller holds the connection open. The factory receives already-materialized
  `list[Tool]`. This is correct, but the CLI `mcp run --agent-type code` path
  needs to ensure the MCP connection stays open for the duration of agent
  execution. This is already the case in the current `mcp run` implementation
  but worth verifying.

- **Facade exposure in multi-model system**: `MultiModelSerpAPISystem` uses
  worker agents as `AgentTool` wrappers. Could a `code_mode` agent be embedded
  as a worker inside `MultiModelSerpAPISystem`? Out of scope for now but worth
  flagging.

## Resolved Questions

| Question | Resolution |
|----------|------------|
| Core motivation | All three: reduce tool-count overhead, enable multi-step code logic, sandbox isolation |
| Integration point | Thin factory (Approach A) |
| Facade scope | MCP tools + existing default tools combined |
| E2B priority | Nice-to-have; local executor first, E2B configurable |
| CLI exposure | Extend `mcp run` with `--agent-type` flag |
| `additional_authorized_imports` | Yes — factory accepts the list; callers extend defaults (mirrors `InternetAgent` pattern) |
| `ToolFacade.search()` matching | Exact case-insensitive substring — simple, matches the prototype |

## Affected Files

| File | Change |
|------|--------|
| `agentic_internet/agents/code_mode.py` | **New** — `ToolFacade`, `SearchTool`, `ExecuteTool`, `create_code_mode_agent()` |
| `agentic_internet/cli.py:718` | Add `--agent-type` option to `mcp run` command |
| `agentic_internet/__init__.py` | Export `create_code_mode_agent` |
| `agentic_internet/agents/__init__.py` | Export `create_code_mode_agent` |

## Next Steps

→ `/workflows:plan` for implementation details
