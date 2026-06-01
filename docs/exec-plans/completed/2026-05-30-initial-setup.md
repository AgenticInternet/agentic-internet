---
title: "Initial Project Setup"
area: "Global - project foundation"
status: complete
risk: low
created: 2026-05-30
updated: 2026-06-01
author: agent
---

# Plan: Initial Project Setup

## Intent

Adapt Agentic Internet with a harness engineering structure so agents and humans can navigate architecture, quality, security, reliability, and active plans from stable repository entry points.

## Acceptance Criteria

- [x] Existing repository inspected before edits
- [x] AGENTS.md map created while preserving bead workflow
- [x] docs/ knowledge base created with seven top-level docs
- [x] docs/design-docs/ catalogue created
- [x] docs/exec-plans/ lifecycle created
- [x] docs/product-specs/ index created
- [x] docs/references/ index created
- [x] .opencode/ local config created
- [x] Golden-principles linter stub created
- [x] CI skeleton created
- [x] README references AGENTS.md and core docs
- [x] First real feature exec-plan created

## Non-Goals

- Does not implement product features.
- Does not rewrite Python package internals.
- Does not resolve unrelated uncommitted changes already present in the worktree.

## Verification

- `python3 .opencode/tools/golden_principles.py`
- `uv run ruff check agentic_internet tests`
- `uv run ruff format --check agentic_internet tests`
- `uv run mypy agentic_internet`
- `uv run pytest`

## Progress Log

| Date | Update |
|------|--------|
| 2026-05-30 | Repository adapted with harness engineering structure |
| 2026-06-01 | Moved to completed after the K-LLM feature exec-plan was created, implemented, tested, pushed, and opened for review. |
