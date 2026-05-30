# Quality Score

Initial grades are intentionally conservative. Raise them with evidence from tests, CI, docs, and production usage.

| Area | Grade | Evidence | Next Step |
|------|-------|----------|-----------|
| Architecture | C | Package has clear modules; formal layer rules now documented | Add ADRs for major orchestration decisions |
| Tests | C | Pytest suite exists | Separate unit and integration gates in CI |
| Typing | C | Mypy configured; public boundary typing varies | Tighten types on tools and agent constructors |
| Reliability | C | Fallback providers exist in places | Define provider outage behavior and timeouts |
| Security | C | Secret placeholders documented | Audit code execution and browser automation boundaries |
| Documentation | C | README and MCP docs exist | Keep docs linked from AGENTS.md and PLANS.md |

## Tech Debt Tracker

| ID | Area | Issue | Owner | Status |
|----|------|-------|-------|--------|
| TBD | TBD | Add entries as debt is discovered | TBD | open |

## Grading Rubric

- A: Enforced in CI, documented, and covered by tests.
- B: Documented and mostly tested, with known gaps tracked.
- C: Exists and is usable, but enforcement or coverage is partial.
- D: Fragile, inconsistent, or mostly implicit.
- F: Missing or actively misleading.
