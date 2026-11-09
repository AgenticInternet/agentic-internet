# Implementation Plans

Generated during the repository improvement review on 2026-11-09. Execute plans in dependency order. Reviewers own this index during dispatched execution.

## Execution order & status

| Plan | Title | Priority | Effort | Depends on | Status |
|------|-------|----------|--------|------------|--------|
| [001](001-stabilization-baseline.md) | Restore a safe, testable stabilization baseline | P1 | M | — | DONE |
| [002](002-refresh-openrouter-models.md) | Refresh and validate OpenRouter model mappings | P1 | M | 001 | DONE |

Status values: TODO | IN PROGRESS | DONE | BLOCKED | REJECTED

## Dependency notes

- Plan 001 completed with 213 passing tests, clean Ruff/mypy/golden checks, and verified wheel/sdist/console-script behavior.
- Plan 002 completed with 65 live-validated aliases, exact ModelManager routing, and 220 passing tests plus one opt-in skip. Live key and catalog checks passed; bounded free-provider response limitations are documented without a paid retry.
- Packaging required uv-generated lock metadata normalization and a minimal `MANIFEST.in`; no dependency versions or resolved dependency identities changed.
- Broader dependency modernization, model-registry consolidation, and execution lifecycle redesign must follow this baseline. They are intentionally outside Plan 001.

## Findings considered and rejected for this slice

- Raising the Python 3.11 minimum: unnecessary because it remains security-supported.
- Blanket upgrades for every audit record: advisory reachability and compatibility need targeted review.
- Large orchestration refactor before characterization tests: too risky.
- Restoring Python's unrestricted importer for generated code: rejected; only a narrow allowlist is acceptable.
