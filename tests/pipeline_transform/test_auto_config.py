#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for Auto-Configuration Agent (Sub-Agent D3).
"""

import pytest

from neuralops.pipeline_transform.auto_config import suggest_config
from neuralops.pipeline_transform.errors import ConfigurationError
from neuralops.pipeline_transform.hardware_planner import estimate_resources
from neuralops.recipe.schema import Recipe


def test_d3_suggest_config_adjusts_parameters_to_satisfy_vram_constraint():
    """Verify suggest_config lowers target_ratio when VRAM constraint is tight."""
    # Model of 5 Billion parameters: baseline peak VRAM is ~15-20 GB
    recipe_dict = {
        "apiVersion": "neuralops/v1",
        "kind": "Recipe",
        "metadata": {
            "name": "auto-config-test",
            "version": "1.0.0",
            "description": "Recipe to be auto-configured",
        },
        "model": {"source": "llama-5b", "dtype": "float16"},
        "operations": [
            {
                "id": "c1",
                "type": "weights.compress",
                "version": "1.0.0",
                "config": {"method": "svd", "target_ratio": 0.8},
            }
        ],
    }
    recipe = Recipe.from_dict(recipe_dict)

    # Tight VRAM constraint: max 12 GB on RTX 4090
    optimized_recipe = suggest_config(
        recipe,
        hardware_constraints={"max_vram_gb": 12.0},
        hardware_profile="rtx4090",
    )

    assert optimized_recipe is not None
    # Check that target_ratio was adjusted lower than original 0.8
    new_ratio = optimized_recipe.operations[0].config["target_ratio"]
    assert new_ratio < 0.8

    # Re-estimate resources on optimized recipe
    est = estimate_resources(optimized_recipe, hardware_profile="rtx4090")
    assert est.peak_vram_mb <= 12.0 * 1024.0


def test_d3_impossible_constraint_raises_configuration_error():
    """Verify suggest_config raises ConfigurationError when constraints are impossible."""
    recipe_dict = {
        "apiVersion": "neuralops/v1",
        "kind": "Recipe",
        "metadata": {
            "name": "impossible-test",
            "version": "1.0.0",
            "description": "Recipe with impossible constraint",
        },
        "model": {"source": "llama-7b", "dtype": "float16"},
        "operations": [
            {
                "id": "c1",
                "type": "weights.compress",
                "version": "1.0.0",
                "config": {"method": "svd", "target_ratio": 0.5},
            }
        ],
    }
    recipe = Recipe.from_dict(recipe_dict)

    # Impossible VRAM budget: 0.05 GB (50 MB for a multi-gigabyte model)
    with pytest.raises(ConfigurationError) as exc_info:
        suggest_config(recipe, hardware_constraints={"max_vram_gb": 0.05})

    assert "infeasible" in str(exc_info.value).lower() or "satisfy" in str(exc_info.value).lower()
