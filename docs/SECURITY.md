# Security

## Hard Invariants

- Never commit real API keys, tokens, cookies, or browser session credentials.
- Validate untrusted input before browser automation, code execution, or filesystem access.
- Do not log secrets or raw provider credentials.
- Keep `.env` local and use `.env.example` for safe placeholders.
- Treat generated code, scraped content, and browser results as untrusted data.
- Prefer least-privilege provider keys for local development and CI.

## Auth and Secrets

This package is CLI/library-first and does not own end-user authentication. Provider authentication is environment-variable based:

- `HUGGINGFACE_TOKEN`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `SERPAPI_API_KEY`
- `OPENROUTER_API_KEY`
- `BROWSER_USE_API_KEY`

Secrets should be loaded at runtime through settings and never passed through README examples as real values.

## Data Classification

| Class | Examples | Handling |
|-------|----------|----------|
| Public | README text, docs, public search snippets | Safe to store in repo |
| Internal | Plans, architecture notes, local logs | Review before publishing |
| User-provided | Prompts, URLs, uploaded data, scraped pages | Avoid unnecessary persistence |
| Secret | API keys, tokens, cookies, browser sessions | Environment/secret manager only |

## Sensitive Areas

- `agentic_internet/tools/code_execution.py`
- `agentic_internet/tools/browser_use.py`
- `agentic_internet/tools/mcp_integration.py`
- Provider/model routing in orchestration modules

Changes in these areas require focused tests and explicit review of side effects.
