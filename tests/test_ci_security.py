from pathlib import Path

CI_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"


def test_ci_pins_setup_uv_action_and_uv_version() -> None:
    workflow_lines = {line.strip() for line in CI_WORKFLOW.read_text(encoding="utf-8").splitlines()}

    assert "uses: astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba460eec2 # v10.3.3" in workflow_lines
    assert 'version: "0.13.7"' in workflow_lines
