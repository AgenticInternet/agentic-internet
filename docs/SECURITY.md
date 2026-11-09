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

## Trusted Provider Destinations

Provider destination settings are trusted configuration, not ordinary task or model input. These include LiteLLM's `api_base` and `base_url`, the `OPENROUTER_API_BASE`, `OPENAI_API_BASE`, and `ANTHROPIC_BASE_URL` environment variables, and direct LiteLLM keyword arguments or global configuration that can redirect requests. These settings can determine where provider credentials and prompt/tool data are sent.

- Require review for environment and configuration changes that affect provider destinations, including shell environment files and CI variables.
- Where practical, enforce provider endpoint allowlists at the network or egress layer and restrict destination overrides to approved providers.
- Never map task text, model text, or other untrusted input to provider destinations or arbitrary LiteLLM keyword arguments.
- Rotate affected provider credentials after an untrusted party changes destination-related environment, global configuration, or direct arguments, and review possible credential and prompt/tool-data exposure.

Normal CLI model text was not found to control provider destinations. This does not make untrusted environment changes or direct LiteLLM overrides safe.

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
