# Plan 003: Consolidate repository hygiene and documentation structure

> **Executor instructions**: Execute only this hygiene plan. Use `git mv` for preserved files, delete only explicitly listed files, and run verification after each group. Do not change dependencies, runtime behavior, or module boundaries beyond moving examples. Do not touch the main checkout or its untracked ` 2` backups; the reviewer handles those separately.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED (file deletion and documentation/example paths)
- **Depends on**: plans/002-refresh-openrouter-models.md (DONE)
- **Category**: tech-debt, dx, docs
- **Planned at**: commit `9181cbc`, 2026-11-09

## Why this matters

The repository contains a root `main.py` that only prints a greeting, two legacy live-provider test scripts, third-party documentation copied from unrelated projects, a generated SpecStory session, and parallel locations for plans, examples, and architecture documentation. This makes the project harder to navigate and its ownership/history unclear. The cleanup preserves all current runtime code and consolidates durable content in conventional locations.

## Verified current state

- Root `main.py` only prints `Hello from agentic-internet!` and is not referenced by `pyproject.toml` or the package entry point.
- Root `test_agent.py` and `test_simple_agent.py` are legacy live-provider scripts without assertions. They are outside pytest's configured `testpaths = ["tests"]` and duplicate maintained examples/tests.
- `specs/CODEBASE_DIAGRAM.md` is current after the 2026-11-09 update and must be preserved by moving it to `docs/CODEBASE_DIAGRAM.md`.
- Other files in `specs/` are unrelated third-party content, generated chat history, or stale duplicates:
  - `AUTHORS.md` is an unrelated Black project authors list.
  - `Readme.md` documents LiteLLM Google GenAI, not this package.
  - `README_SCIM.md` documents LiteLLM Proxy SCIM endpoints, which this project does not implement.
  - `datasetcard_template.md` and `modelcard_template.md` are HuggingFace templates unrelated to this package.
  - `2025-08-30_20-45Z-update-cli-based-on-current-updates.md` is generated SpecStory chat history.
  - `.what-is-this.md` is a SpecStory directory explanation, not project documentation.
  - `CLI_ENHANCEMENTS.md` and `FEATURES.md` are stale duplicates of README/docs and contain incorrect environment names, models, and safety claims.
- `plans/001-stabilization-baseline.md` and `plans/002-refresh-openrouter-models.md` are completed. `plans/README.md` duplicates status that belongs in canonical `docs/PLANS.md`. After the move, `plans/README.md` remains only as a short pointer to `docs/PLANS.md`; this plan and Plan 004 belong in `docs/exec-plans/active/`.
- `agentic_internet/examples/` contains usage examples, not production modules. Move all seven files together to root `examples/` and update the two MCP command references and README project structure. Do not leave import shims that keep examples in the distribution.
- The main checkout has 11 untracked ` 2` backups verified byte-identical to tracked files. They are outside this worktree and outside this plan; the reviewer will delete them only after implementation is green.

## Scope

**Delete (tracked, with no replacements):**

- `main.py`
- `test_agent.py`
- `test_simple_agent.py`
- `specs/.what-is-this.md`
- `specs/2025-08-30_20-45Z-update-cli-based-on-current-updates.md`
- `specs/AUTHORS.md`
- `specs/CLI_ENHANCEMENTS.md`
- `specs/FEATURES.md`
- `specs/README_SCIM.md`
- `specs/Readme.md`
- `specs/datasetcard_template.md`
- `specs/modelcard_template.md`

**Move, preserving content/history:**

- `specs/CODEBASE_DIAGRAM.md` → `docs/CODEBASE_DIAGRAM.md`
- All files in `agentic_internet/examples/` → `examples/` with the same filenames
- `plans/001-stabilization-baseline.md` → `docs/exec-plans/completed/2026-11-09-stabilization-baseline.md`
- `plans/002-refresh-openrouter-models.md` → `docs/exec-plans/completed/2026-11-09-refresh-openrouter-models.md`
- `plans/003-repository-hygiene.md` → `docs/exec-plans/active/2026-11-09-repository-hygiene.md`
- `plans/004-split-multi-model.md` → `docs/exec-plans/active/2026-11-09-split-multi-model.md`

**Update only as needed:**

- `AGENTS.md`: link `docs/CODEBASE_DIAGRAM.md`, root `examples/`, and canonical `docs/PLANS.md`.
- `README.md`: project structure and example invocation paths.
- `docs/PLANS.md`: merge the current root plan index, listing Plans 001/002 completed and Plans 003/004 active with dependencies. Replace root `plans/README.md` with a short pointer to the canonical index.
- `docs/MCP_INTEGRATION.md`: change example paths/commands from `agentic_internet/examples/` to `examples/` and `python -m agentic_internet.examples.<module>` to `python examples/<module>.py`.
- `docs/CODEBASE_STATUS.md`: replace stale June status with a short current map and follow-up list; do not preserve outdated counts or completed actions as pending.
- `MANIFEST.in`: add `prune examples` if needed to keep the moved examples out of distributions. Do not change `pyproject.toml` or `uv.lock`.

**Out of scope:**

- Any production module deletion, split, rename, or behavior change.
- `src/` layout, CLI decomposition, dependency upgrades, and generated cache/venv deletion.
- Root `plans/` content not explicitly listed for a move.
- Main-checkout backups and ignored files.

## Steps

### Step 1: Delete only verified obsolete files

Use `git rm` for the exact tracked deletion list. Do not delete any file whose content was not reviewed. Confirm `docs/CODEBASE_DIAGRAM.md` exists before removing the old `specs/` files.

**Verify**: `git status --short` shows only the intended deletions and the diagram move.

### Step 2: Move examples and update references

Use `git mv agentic_internet/examples examples`. Update README, `docs/MCP_INTEGRATION.md`, `docs/CODEBASE_STATUS.md`, and `AGENTS.md` references. Add `prune examples` to `MANIFEST.in` so root examples remain source documentation, not package runtime modules.

**Verify**: `uv run --frozen --no-sync python examples/example_mcp_server.py --help` may be syntax-checked without starting a server if it has no help flag; use `python -m py_compile examples/*.py` instead. All example files must compile without requiring credentials.

### Step 3: Consolidate plans and architecture docs

Use `git mv` for the diagram and all plan files listed above. Merge status/dependency notes from `plans/README.md` into `docs/PLANS.md`, then replace `plans/README.md` with a short pointer to the canonical index. Keep `docs/PLANS.md` as the only substantive plan index. Update AGENTS and README links to `docs/CODEBASE_DIAGRAM.md`.

**Verify**: `test -f docs/PLANS.md && test -f docs/CODEBASE_DIAGRAM.md` succeeds. No files remain under `specs/`; only the pointer README remains under `plans/`.

### Step 4: Refresh current status and validate

Update `docs/CODEBASE_STATUS.md` with current 2026-11-09 status, current 65-alias catalog, completed stabilization work, and remaining execution-lifecycle/CLI/dependency follow-ups. Then run:

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

Inspect the wheel and sdist. They must contain no `agentic_internet/examples/` or `tests/` files. Confirm no dependency or lockfile content changed.

## Done criteria

- [ ] Every listed obsolete tracked file is removed and every preserved file is moved with history.
- [ ] `specs/` no longer exists and `plans/` contains only the pointer README; canonical docs, examples, and exec-plan locations contain the preserved content.
- [ ] README, AGENTS, MCP docs, status, and plan index links resolve to existing paths.
- [ ] Full quality gates and package build pass, with no examples/tests in wheel or sdist.
- [ ] No dependency/version or lockfile change and no production module behavior change.

## STOP conditions

- A referenced file is not present or contains content different from the reviewed baseline.
- A move would break a production import or public package API.
- Any example requires credentials for syntax validation.
- The package build includes moved examples or tests, or `uv.lock` changes.
- The change requires deleting anything not explicitly listed.

## Maintenance notes

- New examples belong under root `examples/`, not inside the package.
- New active/completed plans belong only under `docs/exec-plans/` and are indexed only in `docs/PLANS.md`.
- Do not reintroduce third-party reference documents into project specs; link official sources in `docs/references/README.md` instead.
