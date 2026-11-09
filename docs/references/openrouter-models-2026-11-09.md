# OpenRouter Model Snapshot — 2026-11-09

## Verification

- Source: [OpenRouter model overview](https://openrouter.ai/docs/guides/overview/models) and [`GET /api/v1/models`](https://openrouter.ai/api/v1/models).
- Live inventory: 461 models at review time.
- Static project aliases after refresh: 65, including the unchanged default and OpenRouter fallback.
- Every configured alias was checked against the live inventory with `OPENROUTER_LIVE_TESTS=1 uv run --frozen --no-sync pytest -p no:cacheprovider -m integration tests/test_openrouter_models.py`.
- The operator-provided OpenRouter key was validated through `/api/v1/key` with HTTP 200; its value was not printed or stored.

Static aliases are a curated offline fallback. Use `agentic-internet models --live` for the freshest inventory and capability metadata.

## Removed stale aliases

The following 20 aliases were absent from the live catalog and were removed:

- `claude-opus-4.7-fast`, `claude-opus-4.8-fast`
- `deepseek-v3.2-speciale`
- `gemini-3-pro`
- `granite-4.1-8b`
- `grok-4`, `grok-4-fast`, `grok-4.1-fast`, `grok-code-fast`
- `magistral-medium`
- `mimo-v2-flash`
- `minimax-m2.1`
- `o3-mini`, `o4-mini`
- `owl-alpha`
- `qwen3-235b`, `qwen3-coder-plus`, `qwen3-max`
- `ring-2.6-1t`
- `tongyi-deepsearch`

## Curated recent additions

| Alias | OpenRouter ID |
|-------|---------------|
| `gpt-6.1-terra` | `openai/gpt-6.1-terra` |
| `gpt-6.1-terra-pro` | `openai/gpt-6.1-terra-pro` |
| `gpt-6-luna` | `openai/gpt-6-luna` |
| `gpt-6-luna-pro` | `openai/gpt-6-luna-pro` |
| `claude-sonnet-5.5` | `anthropic/claude-sonnet-5.5` |
| `claude-haiku-5.5` | `anthropic/claude-haiku-5.5` |
| `gemini-4-argon` | `google/gemini-4-argon` |
| `kimi-k3.1-pro` | `moonshotai/kimi-k3.1-pro` |
| `qwen3.8-max-prime` | `qwen/qwen3.8-max-prime` |
| `deepseek-v4.1-flash` | `deepseek/deepseek-v4.1-flash-1017` |
| `glm-5.3-prime` | `z-ai/glm-5.3-prime` |
| `mistral-large-4` | `mistralai/mistral-large-4-0` |
| `minimax-m3.1-flash` | `minimax/minimax-m3.1-flash` |
| `step-5-preview` | `stepfun/step-5-preview` |
| `command-a-plus` | `cohere/command-a-plus` |
| `solar-mini4` | `upstage/solar-mini4` |
| `perceptron-mk1.5` | `perceptron/perceptron-mk1.5` |
| `lfm-3-1.7b` | `liquid/lfm-3-1.7b` |
| `lfm-3-1.7b-free` | `liquid/lfm-3-1.7b:free` |
| `nemotron-3.5-super-vl-free` | `nvidia/nemotron-3.5-super-vl:free` |
| `hy4` | `tencent/hy4` |
| `particle-1` | `particle-ai/particle-1` |
| `openrouter-auto` | `openrouter/auto` |
| `openrouter-auto-code` | `openrouter/auto-code` |

The LiteLLM-facing values in `ModelConfig.openrouter_models` prepend `openrouter/` to these provider IDs.

## Routing and default behavior

- The default remains `openrouter/anthropic/claude-opus-4.8`; the OpenRouter fallback is unchanged.
- `ModelManager` resolves explicit aliases through the central catalog before legacy role mappings, preventing `claude-opus-4.8` from silently selecting 4.5 or `o3` selecting O4-mini.
- Legacy role defaults that referenced retired IDs now point to current models.
- Static CLI categories resolve through the same central aliases, with a parity test preventing catalog drift.

## Live execution notes

A free `liquid/lfm-3-1.7b:free` request timed out. An alternate `nvidia/nemotron-3.5-super-vl:free` request returned a valid `ChatMessage` and provider token/cost metadata with zero reported cost. Four-token and 32-token attempts produced reasoning without final answer text, so they are recorded as bounded free-provider limitations, not successful final completions. No paid retry was made.
