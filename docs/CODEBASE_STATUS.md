# Codebase Status

Status date: 2026-11-09

## Current Map

| Area | Location | Current role |
|------|----------|--------------|
| Runtime package | `agentic_internet/` | Agents, orchestration, tools, configuration, and CLI |
| Usage examples | `examples/` | Standalone documentation scripts outside the runtime package |
| Tests | `tests/` | Pytest suite; no live credentials required for the default gate |
| Architecture | [Codebase diagrams](CODEBASE_DIAGRAM.md) | Current Mermaid system views and code anchors |
| Planning | [Canonical plan index](PLANS.md) | Active/completed exec-plans and dependencies |

## Completed Stabilization

- Plan 001 restored allowlisted generated-code imports, CLI credential redaction, safe SerpAPI parameter memory, explicit agent tool/step controls, and package build behavior. Its full quality and packaging gates passed.
- Plan 002 refreshed the OpenRouter mappings, added CLI/central catalog parity and exact ModelManager routing, and validated the live key/catalog without exposing credentials.
- Plan 003 consolidated examples, architecture diagrams, and exec-plans into canonical locations and removed obsolete root/generated files without changing production behavior.
- Plan 004 split 18 multi-model classes into focused modules with original-path compatibility. Independent AST comparison confirmed identical implementations, and 249 tests plus one opt-in skip passed.
- Packaging metadata normalization preserved dependency versions and resolved dependency identities. The static catalog remains an offline fallback; `models --live` is the dynamic inventory path.

## OpenRouter Catalog

The curated catalog contains **65 aliases**, focused on standard GPT-6 Terra, Sol, and Luna tiers rather than Pro variants. Explicit aliases resolve to exact configured provider IDs. Bounded free-provider response limitations are documented without an implicit paid retry. See the [2026-11-09 OpenRouter snapshot](references/openrouter-models-2026-11-09.md).

## Remaining Follow-ups

- **Execution lifecycle:** process isolation, resource limits, cancellation, concurrent-execution safety, and provider timeout/outage policy still require design and implementation.
- **CLI:** decomposition and behavior-coverage improvements remain; preserve existing command behavior while restructuring.
- **Dependencies:** review advisory reachability and compatibility before targeted upgrades rather than applying blanket version changes; Python 3.11 remains supported.
- **Structure and routing:** The compatibility-preserving multi-model split is complete. Full model-registry consolidation and provider fallback attribution remain separate follow-ups.
