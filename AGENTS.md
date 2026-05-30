# AGENTS.md - Agentic Internet

> This file is a map, not a manual. Read it, then navigate to the source of truth.
> Repository knowledge that is not documented here or linked from here is effectively invisible to agents.

---

## 1. Repository Knowledge Map

| Document | Purpose |
|----------|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System shape, layer rules, golden principles |
| [docs/DESIGN.md](docs/DESIGN.md) | Python conventions, API patterns, logging conventions |
| [docs/PLANS.md](docs/PLANS.md) | Active exec-plans index |
| [docs/QUALITY_SCORE.md](docs/QUALITY_SCORE.md) | Domain health grades and tech debt tracker |
| [docs/RELIABILITY.md](docs/RELIABILITY.md) | Critical paths, SLO placeholders, rollback policy |
| [docs/SECURITY.md](docs/SECURITY.md) | Security invariants, auth/secrets guidance, data classes |
| [docs/PRODUCT_SENSE.md](docs/PRODUCT_SENSE.md) | Product purpose, core beliefs, non-goals |

## 2. Where to Look

```text
AGENTS.md                          <- you are here
├── agentic_internet/               <- package source
├── tests/                          <- pytest suite
├── docs/ARCHITECTURE.md            <- layer model and dependency rules
├── docs/design-docs/index.md       <- architectural decisions
├── docs/exec-plans/active/         <- in-flight plans
├── docs/product-specs/index.md     <- product specs
└── docs/references/README.md       <- library/reference index
```

## 3. Agent Operating Principles

1. Repository is ground truth. Write durable decisions into docs, specs, or plans.
2. Plans before broad code changes. Complex work needs an exec-plan first.
3. Keep source changes scoped. Do not rewrite unrelated Python modules or lockfiles.
4. Preserve user work. Never revert uncommitted changes unless explicitly asked.
5. Golden principles are mechanical. Run `make golden` or `python3 .opencode/tools/golden_principles.py`.
6. Update docs with architecture changes; stale docs are treated as defects.

## 4. Issue Workflow

This project uses `bd` (beads) for issue tracking.

```bash
bd ready
bd show <id>
bd update <id> --status in_progress
bd close <id>
bd sync
```

## 5. Quality Gates

Run the narrowest useful gate during development and the full gate before handoff:

```bash
uv run ruff check agentic_internet tests
uv run ruff format --check agentic_internet tests
uv run mypy agentic_internet
uv run pytest
python3 .opencode/tools/golden_principles.py
```

`make check` runs the default local gate set.

## 6. SDLC States

```text
INTENT -> SPEC -> PLAN -> IMPLEMENT -> VERIFY -> DOCS -> REVIEW -> RELEASE -> MONITOR
```

## 7. Ask-for-Help Triggers

Stop and surface to the human when:

- Requirements or acceptance criteria are ambiguous.
- Test failure root-cause confidence is below 60%.
- A new external dependency is needed.
- The change touches secrets, authentication, sandboxing, or untrusted execution.
- A design decision affects multiple domains or public interfaces.

Use this format:

```markdown
## Help Request
**What I tried**: [actions]
**Evidence**: [logs, diffs]
**Options**: [with tradeoffs]
**Recommendation**: [choice] (confidence: X%)
```

## 8. Session Completion

Before ending a work session:

1. File `bd` issues for known follow-up work.
2. Run relevant quality gates.
3. Update issue status for work you changed.
4. Sync beads and push committed work when the session owns the change.
5. Hand off remaining context clearly.

Do not commit or push unrelated uncommitted user changes.
