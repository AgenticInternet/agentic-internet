# Design

## Python Conventions

- Keep public imports stable through `agentic_internet/__init__.py`.
- Prefer small modules organized by capability: `agents/`, `tools/`, `config/`, `utils/`.
- Use descriptive function and class names; reserve abbreviations for common provider names.
- Keep CLI argument parsing in `cli.py`; move behavior into package modules when it grows.
- Use `pathlib.Path` for filesystem paths in new code.
- Prefer typed function signatures at public boundaries.

## Agent Patterns

- Agents should compose tools and configuration instead of constructing provider clients deep in execution paths.
- Long-running orchestration should make timeout, iteration, and model choices explicit.
- New agent modes should have a focused example and at least one test covering construction or routing.

## Tool Patterns

- Tool modules should expose one capability family.
- Validate untrusted inputs before network, browser, or code execution operations.
- Return structured data where downstream agents need to reason over fields.
- Keep provider fallback behavior explicit and documented.

## CLI Patterns

- CLI commands should be thin and predictable.
- Output intended for humans can use `rich`; output intended for automation should support JSON or file output.
- New flags need README or docs coverage when they affect common workflows.

## Logging

Use structured, key-value-oriented messages where practical:

```python
logger.info("search_completed", extra={"query": query, "results": len(results)})
```

Avoid logging secrets, full provider responses containing user data, or raw browser session tokens.

## Tests

- Unit tests belong under `tests/`.
- Root-level `test_*.py` files are tolerated legacy entry points; prefer `tests/` for new tests.
- Mark external-service tests with `@pytest.mark.integration`.
