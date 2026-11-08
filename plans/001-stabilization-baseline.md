# Plan 001: Restore a safe, testable stabilization baseline

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving on. If a STOP condition occurs, stop and report; do not improvise. The reviewer maintains `plans/README.md`, so do not edit that index.
>
> **Drift check (run first)**: `git diff --stat bb2dcd0..HEAD -- agentic_internet/cli.py agentic_internet/config/settings.py agentic_internet/agents/internet_agent.py agentic_internet/agents/multi_model_serpapi.py agentic_internet/tools/code_execution.py pyproject.toml README.md docs/QUALITY_SCORE.md tests/`
> The plan-only commit may change `plans/`; source drift requires review before implementation.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: HIGH (the import hook touches a documented non-sandbox execution boundary)
- **Depends on**: none
- **Category**: bug, security, dx
- **Planned at**: commit `bb2dcd0`, 2026-11-09

## Why this matters

The current tracked suite passes on macOS but remote Linux CI fails in the Python executor. Independent reproduction also showed that an allowed `import math` fails because `__import__` is absent. Separately, `config --show` prints provider credentials, SerpAPI memory can replay the first successful query while retaining its credential, and explicit tool/step controls are not consistently honored. Finally, the golden check requires an untracked empty directory and fresh `uv sync` skips the console entry point because no build system is declared.

This slice restores those specific contracts with focused regression tests. It deliberately does not attempt the broader dependency, model-registry, concurrency, or cancellation redesign.

## Current state

- `agentic_internet/tools/code_execution.py` supplies a restricted builtins dictionary without `__import__`, preloads selected modules, and executes compiled AST with `exec`. The module docstring explicitly says this is **not a full sandbox**.
- `agentic_internet/cli.py:128-134` calls `settings.model_dump()` and prints the result. `agentic_internet/config/settings.py:190-208` contains provider credential fields. Four models also use deprecated class-based Pydantic configuration.
- `agentic_internet/agents/multi_model_serpapi.py:422-466` builds current search params, overwrites them with remembered `best_params`, then stores the full request dictionary. The dictionary contains `q` and `api_key`.
- `agentic_internet/agents/internet_agent.py:66` uses `tools or self._get_default_tools()`, so `[]` enables defaults. Lines 93-119 pass `max_steps` only to `CodeAgent`; `planning_enabled` is stored but unused. The installed smolagents `ToolCallingAgent` accepts `max_steps` through `**kwargs` and `planning_interval` directly; this was verified by construction.
- `MultiModelSerpAPISystem.create_use_case_tool_inventory()` always adds Python execution tools, ignoring `settings.tools.code_execution_enabled`.
- `pyproject.toml` has project metadata but no `[build-system]`. `.opencode/tools/golden_principles.py` requires `docs/exec-plans/active` to exist; Git cannot retain an empty directory.
- `README.md` says code executes safely in a sandbox, while implementation documentation correctly says it is not a full sandbox. Live docs contain stale hardcoded test/file counts.

### Reproduced behaviors to preserve as tests

1. `PythonExecutorTool().forward("import math\nresult = math.sqrt(4)")` currently returns an import error. Approved imports must work; non-allowlisted imports must remain rejected.
2. A no-network fake SerpAPI client given `first-query` then `second-query` currently sends `first-query` twice. After the fix it must send both distinct queries, retain only safe tuning fields, and never overwrite an explicit current argument.
3. `ExecuteTool` construction is unrelated to this slice; do not change its serialization path.
4. No live provider, MCP, E2B, Browser Use, or repository `.env` values are needed for any test.

### Repository conventions

- Python 3.11+, typed public boundaries, `pathlib.Path`, small capability-focused changes.
- Use existing pytest patterns: `monkeypatch`, `unittest.mock.patch`, and Typer's `CliRunner`.
- Use `ruff` for lint/format and `mypy` for package typing. Do not reformat unrelated modules.
- Do not weaken the documented execution boundary. The restricted importer must be an allowlist, never Python's unrestricted `__import__` exposed directly.
- Secret regression tests use obvious synthetic sentinel values and assert those values never appear. Do not read or print real credentials.

## Commands you will need

Run from the worktree root. Install/sync once before implementation; subsequent commands should use `--frozen --no-sync` unless verifying package installation.

| Purpose | Command | Expected |
|---|---|---|
| Initial environment | `uv sync --frozen --dev` | exit 0 |
| Focused tests | `uv run --frozen --no-sync pytest -p no:cacheprovider <paths>` | all selected tests pass |
| Full tests | `uv run --frozen --no-sync pytest -p no:cacheprovider` | all tracked tests pass |
| Lint | `uv run --frozen --no-sync ruff check agentic_internet tests` | `All checks passed!` |
| Format | `uv run --frozen --no-sync ruff format --check agentic_internet tests` | all files formatted |
| Types | `uv run --frozen --no-sync mypy agentic_internet` | zero errors |
| Golden | `python3 .opencode/tools/golden_principles.py` | `golden-principles passed` |
| Lock consistency | `uv lock --check` | exit 0, no lock rewrite |
| Package artifact | `uv build` | wheel and sdist build successfully in ignored `dist/` |
| Console script | `uv run agentic-internet --help` | exit 0 and top-level CLI help |

## Scope

**In scope — the only source/test/config/docs files to modify:**

- `agentic_internet/tools/code_execution.py`
- `agentic_internet/cli.py`
- `agentic_internet/config/settings.py`
- `agentic_internet/agents/internet_agent.py`
- `agentic_internet/agents/multi_model_serpapi.py`
- `pyproject.toml`
- `README.md`
- `docs/QUALITY_SCORE.md`
- `docs/exec-plans/active/.gitkeep` (create)
- `tests/test_code_execution.py`
- `tests/test_context_engineering.py`
- `tests/test_serpapi_tools.py` (create)
- `tests/test_internet_agent.py` (create)
- `tests/test_cli_config.py` (create)
- `tests/test_orchestration_runtime.py`

**Out of scope:**

- `uv.lock` and all dependency version changes/removals.
- Timeout/cancellation/process isolation, shared-state concurrency, model routing/catalogs, MCP trust/environment, scraper SSRF, Browser Use, E2B serialization, and broad CLI output/flag redesign.
- CI action version changes and Dependabot configuration.
- Any `.env` file, credential values, ignored artifacts, or main-checkout files ending in ` 2`.
- `plans/README.md` (reviewer maintains it).

If a required fix needs an out-of-scope file, STOP instead of expanding scope.

## Git workflow

- Branch: `fix/stabilization-baseline`, already created from `origin/master` at `bb2dcd0` in this worktree.
- The plan and index are committed before implementation.
- Make one implementation commit with a conventional message such as `fix: stabilize execution and configuration contracts` after all checks pass. Smaller logical commits are acceptable.
- Do not push. The reviewer will independently verify and handle the operator-authorized push.
- Do not edit `plans/README.md` in the implementation commit.

## Steps

### Step 1: Add failing regression tests first

Add the tests below and run them against the unchanged implementation. Keep synthetic credentials only.

- `tests/test_code_execution.py`: actual `import math` execution returns 2; an unapproved module is rejected both by validation and by `forward`; existing NumPy test remains.
- `tests/test_cli_config.py`: use `CliRunner` and a `Settings` object with a temporary `cache_dir`, synthetic values in every credential field plus `model.api_key`, and a recognizable non-secret model name. Assert exit 0, every sentinel is absent from output, a fixed redaction marker is present, and the non-secret model name remains.
- `tests/test_context_engineering.py`: `AgentMemory.remember_search_pattern` preserves safe tuning fields but drops `api_key`, `q`, and `query`.
- `tests/test_serpapi_tools.py`: patch `GoogleSearch` with a no-network fake returning one organic result. Call one `GoogleSearchTool` twice with different queries and different explicit languages. Assert both current queries/languages are sent and memory contains neither query nor credential.
- `tests/test_internet_agent.py`: patch model initialization and agent constructors. Prove `tools=[]` remains empty; nondefault `max_iterations` reaches both agent types; planning true/false maps deliberately to `planning_interval` for both types.
- `tests/test_orchestration_runtime.py`: parameterize `code_execution_enabled` false/true and assert the multi-model inventory excludes/includes `python_executor` and `data_analysis` accordingly without live calls.

**Verify**: run the focused new tests and record that each new regression fails for the expected pre-fix reason. Constructor tests for behavior already correct may pass, but query replay, credential output, import execution, empty tools, and disabled inventory must demonstrate the bug before editing source. If a test cannot run without live services, rewrite it to use fakes before proceeding.

### Step 2: Implement a narrow executor import allowlist

In `agentic_internet/tools/code_execution.py`:

- Define a single explicit allowlist for importable module roots covering the currently documented capabilities: `numpy`, `pandas`, `requests`, `datetime`, `json`, `re`, and `math`.
- Implement a restricted import hook using `importlib.import_module`, rejecting relative imports and every root outside that allowlist with a clear `ImportError`.
- Expose this hook as `__import__` in the restricted builtins. Do not expose the standard importer.
- Make AST import validation use the same allowlist so unapproved imports fail before execution. Preserve existing dunder/builtin validation and preloaded modules.
- Keep the existing warning that this is not a full sandbox. Do not attempt timeout, concurrency, or output-capture changes.

**Verify**: `uv run --frozen --no-sync pytest -p no:cacheprovider tests/test_code_execution.py` → all pass, including actual imports and existing NumPy coverage.

### Step 3: Redact configuration output and remove Pydantic deprecation syntax

In `agentic_internet/cli.py`, create an explicit safe serialization path for `config --show`:

- Redact every top-level credential field from `Settings` and nested `ModelConfig.api_key` using a fixed marker when configured; leave unset values unset/empty and preserve non-secret settings.
- Prefer an explicit field list over a broad substring match so future non-secret fields remain visible. Add a short comment that new secret fields must be added to the list.
- Do not change provider runtime configuration or print settings to verify the fix.

In `agentic_internet/config/settings.py`, replace each inner `class Config` with `model_config = ConfigDict(protected_namespaces=())` and update the Pydantic import. Make no other settings refactor.

**Verify**: `uv run --frozen --no-sync pytest -p no:cacheprovider tests/test_cli_config.py tests/test_settings.py` → all pass, no synthetic sentinel appears, and no Pydantic class-config deprecation warning is emitted.

### Step 4: Store only reusable SerpAPI tuning fields

In `AgentMemory.remember_search_pattern` and `GoogleSearchTool.forward`:

- Before storing `best_params`, drop credential fields and request-specific query fields (`api_key`, `q`, and `query`). Preserve non-sensitive tuning fields.
- Apply remembered defaults before current request params, so explicit current query/language/location/filter arguments always win. Do not store or reuse credentials from memory.
- Apply the central memory sanitization for all callers, including shopping; do not redesign the memory system.

**Verify**: `uv run --frozen --no-sync pytest -p no:cacheprovider tests/test_context_engineering.py tests/test_serpapi_tools.py` → all pass; two different queries and explicit languages remain distinct and no credentials enter recommendations.

### Step 5: Honor tool and agent controls

In `agentic_internet/agents/internet_agent.py`:

- Use defaults only when `tools is None`; preserve an explicit empty list.
- Pass `max_steps=self.max_iterations` to both `ToolCallingAgent` and `CodeAgent`.
- Map `planning_enabled` deliberately to `planning_interval` for both agent types (true → 1, false → `None`) and pass no additional planning flags.

In `MultiModelSerpAPISystem.create_use_case_tool_inventory`:

- Include `PythonExecutorTool` and `DataAnalysisTool` only when `settings.tools.code_execution_enabled` is true; preserve all other inventory behavior.

**Verify**: `uv run --frozen --no-sync pytest -p no:cacheprovider tests/test_internet_agent.py tests/test_orchestration_runtime.py tests/test_cli_use_cases.py` → all pass.

### Step 6: Restore packaging and the golden directory

- Add an explicit PEP 517 build system using `setuptools.build_meta` and a current setuptools lower bound. Configure package discovery so only `agentic_internet*` packages are included; do not add runtime dependencies or edit `uv.lock`.
- Create tracked empty `docs/exec-plans/active/.gitkeep` so a clean checkout satisfies the existing golden check.
- Run `uv sync --frozen --dev` once after the build-system change so the project and console script are installed.

**Verify**: `python3 .opencode/tools/golden_principles.py`, `uv lock --check`, `uv build`, and `uv run agentic-internet --help` → all succeed. Confirm `git status` shows no `uv.lock` modification.

### Step 7: Correct live documentation metrics and safety wording

- In `README.md`, describe local Python execution as restricted but not a security sandbox and direct untrusted workloads to actual isolation. Remove stale hardcoded test counts rather than replacing them with another manually maintained count.
- In `docs/QUALITY_SCORE.md`, remove stale hardcoded pytest/mypy file counts and retain the current risk-based coverage/quality assessment. Do not rewrite historical completed plans.
- Do not claim live provider integration was tested.

**Verify**: documentation contains no `102 tests` or `184 pytest tests` claim and no statement that local execution is safely sandboxed.

### Step 8: Run the complete gate and review the diff

Run:

```bash
uv run --frozen --no-sync ruff check agentic_internet tests
uv run --frozen --no-sync ruff format --check agentic_internet tests
uv run --frozen --no-sync mypy agentic_internet
uv run --frozen --no-sync pytest -p no:cacheprovider
python3 .opencode/tools/golden_principles.py
uv lock --check
uv build
uv run agentic-internet --help
```

Inspect `git status` and the full diff. Only in-scope files may change. Commit after every command passes.

## Test plan

- Follow existing fake/mock patterns in `tests/test_browser_use.py`, `tests/test_cli_use_cases.py`, and `tests/test_code_execution.py`.
- Cover happy paths (approved imports, visible non-secret settings, distinct queries), error paths (unapproved imports), boundaries (empty tools, planning on/off, execution setting on/off), and integration through actual constructors/CLI rendering with no live network.
- Do not test only implementation details: the SerpAPI test must inspect actual sequential requests, and the CLI test must inspect rendered output.
- Full suite must pass with zero failures. New tests must not depend on test order or real environment credentials.

## Done criteria

- [ ] Every listed Step 1 regression has a meaningful test and passes after the fix.
- [ ] `make check` equivalent commands listed in Step 8 all exit 0.
- [ ] `uv build` succeeds and `uv run agentic-internet --help` exits 0.
- [ ] `uv.lock` is unchanged.
- [ ] No out-of-scope file is modified; no secret value or ignored artifact is staged.
- [ ] The implementation commit exists on `fix/stabilization-baseline`.

## STOP conditions

Stop and report if:

- Any in-scope source file differs from the described baseline before edits.
- A verification step fails twice after a reasonable focused fix.
- The import fix seems to require unrestricted builtins, a new runtime dependency, or changes outside the listed file.
- Package build configuration pulls test/example/root-level legacy modules into the distribution or requires editing `uv.lock`.
- A test requires real credentials or external network calls.
- The implementation would benefit from changing timeout, concurrency, model-routing, MCP, Browser Use, or E2B behavior.

## Maintenance notes

- New credential fields must be added to the explicit CLI redaction list and its sentinel regression test.
- New approved execution imports must be added consistently to AST validation and the runtime import hook; adding a library expands the trusted-code surface and requires review.
- Planning defaults now affect agent trajectories; reviewers should check the constructor assertions, not merely test counts.
- Process isolation, resource limits, cancellation, and concurrent execution safety remain necessary follow-up work and are not solved by this plan.
