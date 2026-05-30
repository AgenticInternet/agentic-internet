.PHONY: dev build test lint format typecheck golden check

dev: ## Show CLI help for local development
	uv run python -m agentic_internet.cli --help

build: ## Build package artifacts
	uv build

test: ## Run test suite
	uv run pytest

lint: ## Run ruff checks
	uv run ruff check agentic_internet tests

format: ## Check formatting
	uv run ruff format --check agentic_internet tests

typecheck: ## Run mypy
	uv run mypy agentic_internet

golden: ## Run harness structure checks
	python3 .opencode/tools/golden_principles.py

check: lint format typecheck test golden ## Run local quality gates
