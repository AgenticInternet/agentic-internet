#!/usr/bin/env python3
"""Lightweight repository structure checks for harness engineering docs."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    "AGENTS.md",
    "README.md",
    ".env.example",
    "docs/ARCHITECTURE.md",
    "docs/DESIGN.md",
    "docs/PLANS.md",
    "docs/QUALITY_SCORE.md",
    "docs/RELIABILITY.md",
    "docs/SECURITY.md",
    "docs/PRODUCT_SENSE.md",
    "docs/design-docs/index.md",
    "docs/design-docs/core-beliefs.md",
    "docs/exec-plans/_template.md",
    "docs/exec-plans/active/2026-05-30-initial-setup.md",
    "docs/product-specs/index.md",
    "docs/references/README.md",
    ".opencode/AGENTS.md",
    ".opencode/config.json",
    ".github/workflows/ci.yml",
]

ARCHITECTURE_MARKERS = [f"GP-{index}" for index in range(1, 7)]


def fail(message: str) -> None:
    raise SystemExit(f"golden-principles failed: {message}")


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        fail("missing required files: " + ", ".join(missing))

    agents_lines = (ROOT / "AGENTS.md").read_text(encoding="utf-8").splitlines()
    if len(agents_lines) > 150:
        fail(f"AGENTS.md should stay below 150 lines; found {len(agents_lines)}")

    architecture = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    missing_markers = [marker for marker in ARCHITECTURE_MARKERS if marker not in architecture]
    if missing_markers:
        fail("missing golden principle markers: " + ", ".join(missing_markers))

    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    if "your_" in env_example.lower():
        fail(".env.example should use empty or safe placeholder values, not realistic secrets")

    print("golden-principles passed")


if __name__ == "__main__":
    main()
