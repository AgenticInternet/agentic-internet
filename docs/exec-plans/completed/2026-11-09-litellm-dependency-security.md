# Plan 005: Upgrade LiteLLM and pin uv security tooling

> **Execution model**: Two independent units may run in parallel because their file sets do not overlap. Unit agents must not commit or run the full project suite. The reviewer performs final integration, full tests, live checks, commits, and status updates.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED (large client dependency jump and CI tooling change)
- **Depends on**: plans/004-split-multi-model.md (DONE) and docs/references/security/litellm-uv-cve-review-2026-11-09.md
- **Category**: security, migration, dx
- **Planned at**: commit `024b144`, 2026-11-09

## Why this matters

The repository locks LiteLLM 1.76.1 while current LTS is 1.108.3. Recent critical proxy advisories are not reachable through this client-only project, but the old client still carries historical dependency debt and should be upgraded with compatibility evidence. Current CI already resolves uv 0.13.7, which is outside the reviewed advisory ranges; the remaining work is to pin setup-uv and uv so that resolution is reproducible and uses current hardening.

## Current facts and constraints

- `pyproject.toml` declares `litellm>=1.0.0`; `uv.lock` resolves 1.76.1. Update the declaration to `litellm>=1.108.3,<2` and run `uv lock --upgrade-package litellm`, not a blanket upgrade.
- Current stable LiteLLM is 1.108.3, released 2026-11-07. Do not use a development/prerelease version.
- `setup-uv@v5` currently resolves uv 0.13.7. Pin the action to `astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba460eec2 # v10.3.3` and uv to `version: "0.13.7"`.
- Local uv 0.13.5 and current CI uv 0.13.7 are outside the reviewed CVE ranges. Do not claim an emergency downgrade is needed.
- LiteLLM is used only through smolagents' client adapter. Do not start a LiteLLM proxy or introduce proxy configuration.
- `api_base`, `base_url`, and provider environment overrides remain trusted configuration. Document the risk without breaking custom provider support or adding a new runtime dependency.
- No paid provider request is allowed. Live validation may use the key from `~/.zshenv` only for `/api/v1/key` and a zero-priced model with a small output cap. No paid fallback.

## Unit U1 — LiteLLM upgrade

**Files allowed:**

- `pyproject.toml`
- `uv.lock`
- `tests/test_model_utils.py` only if a compatibility assertion needs adjustment

**Required work:**

1. Change the direct requirement to `litellm>=1.108.3,<2` and run `uv lock --upgrade-package litellm`. Verify the resolved LiteLLM version is 1.108.3. Other direct dependency versions must not change unless the resolver proves a conflict; stop and report any such conflict.
2. Run `uv sync --frozen --dev` to install the updated lock.
3. Run focused tests without the full project suite: `uv run --frozen --no-sync pytest -p no:cacheprovider tests/test_model_utils.py tests/test_model_manager.py tests/test_model_catalog.py tests/test_openrouter_models.py`.
4. If a test fails only because LiteLLM changed a constructor or response shape, make the smallest compatible test adjustment and document it. Do not change production behavior to hide a compatibility problem; stop instead.
5. Report exact resolved dependency changes and any LiteLLM-specific advisories remaining from `uv audit --locked`, distinguishing them from other package advisories.

## Unit U2 — CI and security controls

**Files allowed:**

- `.github/workflows/ci.yml`
- `docs/SECURITY.md`
- `docs/references/security/litellm-uv-cve-review-2026-11-09.md`
- `tests/test_ci_security.py` (create)

**Required work:**

1. Replace `astral-sh/setup-uv@v5` with the full-SHA v10.3.3 pin and add `version: "0.13.7"`. Preserve all existing quality/test steps and triggers.
2. Add a focused test asserting the exact full action SHA and uv version are present in CI. Do not add a YAML dependency.
3. Extend `docs/SECURITY.md` with trusted destination configuration: `api_base`, `base_url`, `OPENROUTER_API_BASE`, `OPENAI_API_BASE`, `ANTHROPIC_BASE_URL`, and direct LiteLLM kwargs. Require environment/config change review, provider endpoint allowlists where practical, no task/model-text-to-destination mapping, and credential rotation after untrusted configuration changes. Clarify that normal CLI model text was not found to control these destinations.
4. Update the security review remediation sections only after CI pin is complete: mark SEC-03 complete and leave SEC-01 for the reviewer. Do not claim LiteLLM compatibility or live completion succeeded.

## Final reviewer verification

- Inspect all changes for unit scope and dependency conflicts.
- Run `make check`, `uv lock --check`, `uv build`, and `uv run --frozen --no-sync agentic-internet --help`.
- Run `OPENROUTER_LIVE_TESTS=1 uv run --frozen --no-sync pytest -p no:cacheprovider -m integration tests/test_openrouter_models.py` for catalog validation.
- Source `~/.zshenv` and verify `/api/v1/key` returns HTTP 200 without printing the key.
- Through the normal `initialize_model()` path, construct a zero-priced model with at most 32 output tokens. A valid provider response proves connectivity/response shape; if reasoning leaves final content empty, record the limitation instead of treating it as a successful final answer. Do not retry with a paid model.
- Run `uv audit --locked` and summarize only LiteLLM-specific and new high-relevance changes, not a blanket vulnerability count.
- Apply `uv self update`, `uv python upgrade`, and pip upgrade inside the worktree virtualenv as separate maintenance steps. Record versions; do not alter the main checkout's files.

## Done criteria

- [ ] LiteLLM resolves exactly 1.108.3 with no unreviewed direct dependency changes.
- [ ] CI uses the exact setup-uv full SHA and uv 0.13.7, with a regression test.
- [ ] Security docs cover destination overrides and credential rotation conditions.
- [ ] Full quality/package gates and live catalog validation pass.
- [ ] Key endpoint and a zero-priced model response are verified without paid fallback.
- [ ] `uv self update`, `uv python upgrade`, and worktree pip upgrade are recorded.
- [ ] Final implementation commits are pushed only after reviewer approval.

## STOP conditions

- LiteLLM 1.108.3 is unavailable or the resolver changes another direct dependency version.
- A provider compatibility test fails without a minimal test-only explanation.
- A task requires paid calls, proxy startup, new runtime dependencies, or exposing credentials.
- CI quality behavior changes beyond the setup-uv pin.
- Any unit modifies files outside its assigned scope.
