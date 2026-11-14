#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying ConflictDetector raises KeyError/ValueError on unregistered operations.
"""

import pytest

from neuralops.pipeline_transform.dependencies import ConflictDetector, analyze_recipe
from neuralops.recipe.schema import Recipe


def test_missing_operation_raises_key_error():
    """Verify ConflictDetector raises KeyError when an operation is not in OPERATION_REGISTRY."""
    recipe_dict = {
        "apiVersion": "neuralops/v1",
        "kind": "Recipe",
        "metadata": {
            "name": "missing-op-recipe",
            "version": "1.0.0",
            "description": "Recipe with missing operation",
        },
        "model": {"source": "llama-3", "dtype": "float16"},
        "operations": [
            {
                "id": "unregistered-stage",
                "type": "nonexistent.op.type",
                "version": "1.0.0",
                "config": {},
            }
        ],
    }
    recipe = Recipe.from_dict(recipe_dict)

    with pytest.raises((KeyError, ValueError), match="nonexistent.op.type"):
        ConflictDetector(recipe)

    with pytest.raises((KeyError, ValueError), match="nonexistent.op.type"):
        analyze_recipe(recipe)
