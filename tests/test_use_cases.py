"""Tests for K-LLM use-case recipes."""

import pytest

from agentic_internet.agents.use_cases import UseCaseRecipe, WorkerRecipe, get_use_case_recipe, list_use_case_recipes


def test_research_recipe_exposes_k_workers():
    recipe = get_use_case_recipe("research")
    assert recipe.k == 4
    assert recipe.worker_names() == (
        "search_researcher",
        "ecommerce_analyst",
        "local_business_analyst",
        "academic_researcher",
    )


def test_dash_normalization_finds_use_case():
    recipe = get_use_case_recipe("technical-due-diligence")
    assert recipe.id == "technical_due_diligence"
    assert recipe.k == 3


def test_list_use_case_recipes_is_stable():
    ids = [recipe.id for recipe in list_use_case_recipes()]
    assert ids == sorted(ids)
    assert "market_intelligence" in ids


def test_unknown_use_case_lists_available_ids():
    with pytest.raises(ValueError, match="Available use cases"):
        get_use_case_recipe("missing")


def test_recipe_rejects_duplicate_workers():
    worker = WorkerRecipe(name="same", description="desc", model_role="role")
    with pytest.raises(ValueError, match="duplicate worker names"):
        UseCaseRecipe(id="bad", description="desc", coordinator_model_role="orchestrator", workers=(worker, worker))
