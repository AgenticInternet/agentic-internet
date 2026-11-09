# Plan 002: Refresh and validate OpenRouter model mappings

> **Executor instructions**: Implement only this model-refresh slice. Add focused tests before production changes, run every verification command, and do not expand into dependency upgrades or orchestration redesign. Do not expose credentials or make paid provider calls.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED (model selection changes which provider model users receive)
- **Depends on**: plans/001-stabilization-baseline.md (DONE)
- **Category**: migration, bug, tests
- **Planned at**: commit `3ec4453`, 2026-11-09

## Why this matters

A live 2026-11-09 comparison found 461 OpenRouter models and 20 stale aliases among 62 configured aliases. The repository also has separate static CLI and multi-model catalogs: previously, an explicit alias such as `claude-opus-4.8` could silently select Claude 4.5 in `ModelManager`. Users need current, exact, testable model mappings without an unexpected default-model or provider-cost change.

## Current state and constraints

- `agentic_internet/config/settings.py` stores 62 OpenRouter aliases and the default `openrouter/anthropic/claude-opus-4.8`. The default and OpenRouter fallback remain available live; do not change them.
- The following 20 aliases were absent from the live catalog and must be removed or replaced: `claude-opus-4.7-fast`, `claude-opus-4.8-fast`, `deepseek-v3.2-speciale`, `gemini-3-pro`, `granite-4.1-8b`, `grok-4`, `grok-4-fast`, `grok-4.1-fast`, `grok-code-fast`, `magistral-medium`, `mimo-v2-flash`, `minimax-m2.1`, `o3-mini`, `o4-mini`, `owl-alpha`, `qwen3-235b`, `qwen3-coder-plus`, `qwen3-max`, `ring-2.6-1t`, `tongyi-deepsearch`.
- `agentic_internet/cli.py` has a separate local `model_catalog` dictionary. Move it to a module-level constant and add a parity test against the central aliases.
- `ModelManager.get_model_for_role()` in `agentic_internet/agents/multi_model_serpapi.py` has independent hardcoded aliases and role defaults. Add exact central-alias resolution before legacy role mappings; do not rewrite the whole manager.
- OpenRouter's `/api/v1/key` endpoint returned HTTP 200 with the operator-provided key. Never print or commit that key.
- The live model endpoint is public. Live alias tests use it only when `OPENROUTER_LIVE_TESTS=1` is explicitly set and are skipped otherwise.
- Live generation must use only zero-priced models. Start with `openrouter/liquid/lfm-3-1.7b:free`; if it fails, one alternate approved by the operator is allowed. Use at most four output tokens initially; a reasoning-model diagnostic may use 32 zero-cost tokens. Do not fall back to a paid model.

## Curated recent aliases

Preserve every currently valid existing alias not listed for removal. Add a focused recent set including at least:

- `gpt-6.1-terra` and `gpt-6.1-terra-pro` → `openai/gpt-6.1-terra` and `openai/gpt-6.1-terra-pro`
- `gpt-6-luna` and `gpt-6-luna-pro` → `openai/gpt-6-luna` and `openai/gpt-6-luna-pro`
- `claude-sonnet-5.5` and `claude-haiku-5.5` → corresponding `anthropic/` IDs
- `gemini-4-argon` → `google/gemini-4-argon`
- `kimi-k3.1-pro` → `moonshotai/kimi-k3.1-pro`
- `qwen3.8-max-prime` → `qwen/qwen3.8-max-prime`
- `deepseek-v4.1-flash` → `deepseek/deepseek-v4.1-flash-1017`
- `glm-5.3-prime` → `z-ai/glm-5.3-prime`
- `mistral-large-4` → `mistralai/mistral-large-4-0`
- `minimax-m3.1-flash` → `minimax/minimax-m3.1-flash`
- `step-5-preview` → `stepfun/step-5-preview`
- `command-a-plus` → `cohere/command-a-plus`
- `solar-mini4` → `upstage/solar-mini4`
- `perceptron-mk1.5` → `perceptron/perceptron-mk1.5`
- `lfm-3-1.7b` → `liquid/lfm-3-1.7b`
- `lfm-3-1.7b-free` → `liquid/lfm-3-1.7b:free`
- `hy4` → `tencent/hy4`
- `particle-1` → `particle-ai/particle-1`
- `openrouter-auto` and `openrouter-auto-code` → `openrouter/auto` and `openrouter/auto-code`

Every value stored in `ModelConfig.openrouter_models` must retain the LiteLLM `openrouter/` prefix, including aliases whose provider is itself `openrouter`.

## Scope

**In scope:**

- `agentic_internet/config/settings.py`
- `agentic_internet/cli.py`
- `agentic_internet/agents/multi_model_serpapi.py`
- `tests/test_openrouter_models.py`
- `tests/test_settings.py`
- `tests/test_model_manager.py` (create)
- `tests/test_model_catalog.py` (create)
- `README.md`
- `docs/references/README.md`
- `docs/references/openrouter-models-2026-11-09.md` (create)
- `plans/README.md`

**Out of scope:**

- Dependency or lockfile changes.
- Default/fallback model changes.
- Full ModelManager or orchestration refactors, cancellation, tool execution, MCP, Browser Use, and unrelated CLI behavior.
- Paid provider calls, credential output, or persistent storage of credentials.
- Main-checkout duplicate files.

## Steps

### Step 1: Add regression tests first

- Central aliases have unique non-empty names, valid `openrouter/` prefixes, and contain the default/fallback IDs.
- Every static CLI catalog entry resolves through `settings.model.openrouter_models`.
- `ModelManager` resolves `claude-opus-4.8`, `o3`, and at least two recent aliases to their exact configured IDs using a mocked `LiteLLMModel`; no fallback to older models is allowed.
- Curated recent aliases are present in the central map.
- An opt-in `integration` test fetches the public model catalog and asserts every configured alias and the default/fallback exists live. Skip it unless `OPENROUTER_LIVE_TESTS=1`.

Run focused tests before editing production and record failures caused by missing recent aliases, CLI parity, and exact routing.

### Step 2: Refresh central aliases and static CLI catalog

Remove the 20 stale aliases, preserve valid existing aliases, and add the curated recent set. Keep the default/fallback unchanged and update the snapshot comment to 2026-11-09.

Move the CLI catalog to `STATIC_MODEL_CATALOG` and update its entries to a smaller curated set spanning general, code, research, news, and science. Every name must resolve centrally. Keep `--live`, `--category`, and `--details` behavior intact.

### Step 3: Resolve explicit aliases exactly in ModelManager

- Store the OpenRouter key on the manager instance without logging it.
- Before legacy model-name/role mappings, resolve a case-insensitive match in `settings.model.openrouter_models` and construct `LiteLLMModel` with that exact ID and existing central generation settings.
- Keep existing role-based fallback behavior for named roles; do not route explicit aliases through arbitrary provider fallback.
- Add no network calls during construction or tests.

### Step 4: Document the snapshot and current flagship examples

Create `docs/references/openrouter-models-2026-11-09.md` with the live count, verification date, removal rationale, curated additions, and links to official OpenRouter model documentation. Link it from `docs/references/README.md`. Update README model examples to include recent aliases and explain that `models --live` is the freshest inventory while the static map is a curated offline fallback.

### Step 5: Verify unit, live metadata, and free execution

Run:

```bash
uv run --frozen --no-sync ruff check agentic_internet tests
uv run --frozen --no-sync ruff format --check agentic_internet tests
uv run --frozen --no-sync mypy agentic_internet
uv run --frozen --no-sync pytest -p no:cacheprovider
python3 .opencode/tools/golden_principles.py
uv lock --check
OPENROUTER_LIVE_TESTS=1 uv run --frozen --no-sync pytest -p no:cacheprovider -m integration tests/test_openrouter_models.py
uv build
uv run --frozen --no-sync agentic-internet --help
```

Source `~/.zshenv` only for commands needing the key. Do not print its value. Verify the key endpoint returns HTTP 200. Construct the free model through the normal `initialize_model()` path and attempt a bounded nonempty response. Record any free-provider limitation without a paid retry.

## Done criteria

- [ ] All 20 stale aliases are absent and the curated recent set is present.
- [ ] Static CLI and central catalog parity tests pass.
- [ ] Explicit aliases resolve to exact configured IDs in ModelManager tests.
- [ ] Default/fallback model IDs are unchanged and live integration validation passes when explicitly enabled.
- [ ] Free live generation succeeds or is reported as a clearly bounded external-service failure without a paid retry.
- [ ] Full quality gates pass with no dependency or lockfile changes.
- [ ] No credential value appears in source, tests, docs, command output, or commits.

## STOP conditions

- The live inventory shape or IDs differ unexpectedly from this plan; re-check rather than blindly replacing the map.
- A change requires dependency updates or broader orchestration/model-routing work.
- A test needs a paid request or would expose a credential.
- The free live smoke request cannot complete; do not substitute a paid model.

## Maintenance notes

- Static aliases are an offline convenience, not a guarantee of provider availability. `models --live` remains the dynamic inventory path.
- Future catalog updates should preserve CLI/central parity and rerun the opt-in live alias test.
- Actual provider fallback attribution and full registry consolidation remain follow-up work.
