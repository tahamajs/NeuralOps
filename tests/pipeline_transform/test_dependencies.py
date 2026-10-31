#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for Dependency and Conflict Detection Engine (Sub-Agent D1).
"""

import pytest

from neuralops.pipeline_transform.dependencies import (
    ConflictSeverity,
    analyze_recipe,
)
from neuralops.recipe.schema import Recipe


def test_d1_detects_compress_before_attention_replace_as_error():
    """Verify analyze_recipe detects 'compress before attention.replace' as ERROR."""
    recipe_dict = {
        "apiVersion": "neuralops/v1",
        "kind": "Recipe",
        "metadata": {
            "name": "invalid-ordering-recipe",
            "version": "1.0.0",
            "description": "Recipe with compress before attention.replace",
        },
        "model": {"source": "llama-3", "dtype": "float16"},
        "operations": [
            {
                "id": "c1",
                "type": "weights.compress",
                "version": "1.0.0",
                "config": {"method": "svd", "target_ratio": 0.5},
            },
            {
                "id": "r1",
                "type": "attention.replace",
                "version": "1.0.0",
                "config": {"target": "chunk_gla"},
            },
        ],
    }
    recipe = Recipe.from_dict(recipe_dict)

    conflicts = analyze_recipe(recipe)

    assert len(conflicts) >= 1
    # First conflict should be an ERROR
    err = conflicts[0]
    assert err.severity == ConflictSeverity.ERROR
    assert err.code in ("E_CONFLICT_ORDER", "E_CONFLICT_MUTATION")
    assert "attention.replace" in err.message or "low_rank_projection" in err.message


def test_d1_valid_recipe_returns_zero_conflicts():
    """Verify analyze_recipe on a properly ordered recipe returns no conflicts."""
    recipe_dict = {
        "apiVersion": "neuralops/v1",
        "kind": "Recipe",
        "metadata": {
            "name": "valid-ordered-recipe",
            "version": "1.0.0",
            "description": "Properly ordered recipe",
        },
        "model": {"source": "llama-3", "dtype": "float16"},
        "operations": [
            {
                "id": "r1",
                "type": "attention.replace",
                "version": "1.0.0",
                "config": {"target": "chunk_gla"},
            },
            {
                "id": "c1",
                "type": "weights.compress",
                "version": "1.0.0",
                "config": {"method": "svd", "target_ratio": 0.5},
            },
        ],
    }
    recipe = Recipe.from_dict(recipe_dict)

    conflicts = analyze_recipe(recipe)
    # Should contain 0 errors
    errors = [c for c in conflicts if c.severity == ConflictSeverity.ERROR]
    assert len(errors) == 0


def test_d1_shape_simulation_propagation():
    """Verify shape simulation propagates through operations without error."""
    recipe_dict = {
        "apiVersion": "neuralops/v1",
        "kind": "Recipe",
        "metadata": {
            "name": "shape-prop-recipe",
            "version": "1.0.0",
            "description": "Recipe testing shape propagation",
        },
        "model": {"source": "llama-3", "dtype": "float16"},
        "operations": [
            {
                "id": "r1",
                "type": "attention.replace",
                "version": "1.0.0",
                "config": {"target": "chunk_gla"},
            }
        ],
    }
    recipe = Recipe.from_dict(recipe_dict)

    conflicts = analyze_recipe(recipe)
    assert len(conflicts) == 0
