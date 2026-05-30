# Reliability

## SLO Placeholders

Define concrete SLOs when runtime deployment targets are known.

| Surface | Target | Current Status |
|---------|--------|----------------|
| CLI command startup | TBD | Not measured |
| Search task completion | TBD | Provider-dependent |
| Browser task completion | TBD | Browser Use dependent |
| Code execution task completion | TBD | Local runtime dependent |

## Critical Paths

| Path | Applies | Notes |
|------|---------|-------|
| Provider API calls | Yes | OpenAI, Anthropic, OpenRouter, HuggingFace, SerpAPI |
| Browser automation | Yes | Requires Browser Use credentials and network access |
| Code execution | Yes | Treat as a safety-sensitive path |
| MCP integration | Yes | Validate server and client behavior separately |
| Payments | No | Not part of current package |
| Database migrations | No | No database layer currently |

## Failure Policy

- Prefer explicit errors over silent fallback when user intent could be compromised.
- Provider fallback should identify which provider/model was actually used.
- Timeouts should be configurable for long-running research or browser tasks.
- Integration tests requiring external services must be marked `integration`.

## Rollback Template

1. Identify affected command, tool, or agent.
2. Revert the smallest commit or feature flag the failing behavior depends on.
3. Run `make check` or the relevant narrow gate.
4. Document the incident in an exec-plan or design note if architecture changed.
