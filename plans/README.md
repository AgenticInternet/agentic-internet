# Implementation Plans

Generated during the repository improvement review on 2026-11-09. Execute plans in dependency order. Reviewers own this index during dispatched execution.

## Execution order & status

| Plan | Title | Priority | Effort | Depends on | Status |
|------|-------|----------|--------|------------|--------|
| [001](001-stabilization-baseline.md) | Restore a safe, testable stabilization baseline | P1 | M | — | IN PROGRESS |

Status values: TODO | IN PROGRESS | DONE | BLOCKED | REJECTED

## Dependency notes

- Broader dependency modernization, model-registry consolidation, and execution lifecycle redesign must follow this baseline. They are intentionally outside Plan 001.

## Findings considered and rejected for this slice

- Raising the Python 3.11 minimum: unnecessary because it remains security-supported.
- Blanket upgrades for every audit record: advisory reachability and compatibility need targeted review.
- Large orchestration refactor before characterization tests: too risky.
- Restoring Python's unrestricted importer for generated code: rejected; only a narrow allowlist is acceptable.
