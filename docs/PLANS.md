# Plans

Exec-plans capture non-trivial work before implementation and preserve handoff context after implementation. This is the canonical plan index; reviewers own status transitions during dispatched execution.

## Active Plans

| Plan | Area | Depends on | Status |
|------|------|------------|--------|
| [005: Upgrade LiteLLM and pin uv security tooling](exec-plans/active/2026-11-09-litellm-dependency-security.md) | Security, dependencies, CI | 004 | IN PROGRESS |

## Status Maps

| Map | Purpose |
|-----|---------|
| [Codebase Status](CODEBASE_STATUS.md) | Current repository map, completed stabilization, and remaining follow-ups |

## Completed Plans

| Plan | Area | Depends on | Status |
|------|------|------------|--------|
| [001: Restore a safe, testable stabilization baseline](exec-plans/completed/2026-11-09-stabilization-baseline.md) | Execution and configuration contracts | — | DONE |
| [002: Refresh and validate OpenRouter model mappings](exec-plans/completed/2026-11-09-refresh-openrouter-models.md) | OpenRouter catalog and routing | 001 | DONE |
| [003: Consolidate repository hygiene and documentation structure](exec-plans/completed/2026-11-09-repository-hygiene.md) | Tech debt, DX, docs | 002 | DONE |
| [004: Split multi-model orchestration into focused modules](exec-plans/completed/2026-11-09-split-multi-model.md) | Multi-model orchestration | 003 | DONE |
| [K-LLM Use-Case Recipes](exec-plans/completed/2026-06-01-k-llm-use-case-recipes.md) | Multi-model orchestration | — | completed |
| [Initial Project Setup](exec-plans/completed/2026-05-30-initial-setup.md) | Global project foundation | — | completed |
| [Code Mode MCP Agent](exec-plans/completed/2026-02-23-feat-code-mode-mcp-agent-plan.md) | Code mode and MCP integration | — | completed |

## Stabilization and Dependency Notes

- Plan 001 completed with 213 passing tests, clean Ruff/mypy/golden checks, and verified wheel/sdist/console-script behavior.
- Plan 002 completed with exact ModelManager routing and live key/catalog checks. A follow-up cost constraint removed Pro variants and added standard Sol, leaving 65 live-validated aliases focused on Terra, Sol, and Luna. Bounded free-provider response limitations are documented without a paid retry.
- Plan 003 consolidated specs/plans/examples into canonical locations and removed obsolete root/generated files without changing production modules.
- Plan 004 split 18 multi-model classes into focused modules with original-path compatibility. All 249 tests plus one opt-in skip and an independent AST equivalence check passed.
- Packaging required uv-generated lock metadata normalization and a minimal `MANIFEST.in`; no dependency versions or resolved dependency identities changed.
- Broader dependency modernization, model-registry consolidation, and execution lifecycle redesign must follow this baseline. They were intentionally outside Plan 001.

## Legacy Plan Location

Older plans may exist under `docs/plans/`. Prefer `docs/exec-plans/active/` for new work and move completed plans to `docs/exec-plans/completed/`.
