#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying Recipe immutability check on recipe_hash (P1).
"""

from __future__ import annotations

import pytest

from neuralops.recipe.canonical import recipe_hash
from neuralops.recipe.schema import Metadata, ModelSpec, Operation, Recipe


def _make_sample_recipe() -> Recipe:
    return Recipe(
        apiVersion="neuralops/v1",
        kind="Recipe",
        metadata=Metadata(name="sample-recipe", version="1.0.0", description="Test recipe"),
        model=ModelSpec(source="bert-base-uncased", dtype="fp16"),
        operations=[Operation(type="attention.replace", version="1.0.0", config={"mechanism": "cosformer"})],
    )


def test_repeated_hashing_unmutated_recipe_succeeds() -> None:
    recipe = _make_sample_recipe()
    h1 = recipe_hash(recipe)
    h2 = recipe_hash(recipe)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_mutating_metadata_after_hashing_raises_value_error() -> None:
    recipe = _make_sample_recipe()
    h1 = recipe_hash(recipe)
    assert h1

    # Mutate metadata name
    recipe.metadata.name = "mutated-recipe-name"

    with pytest.raises(ValueError) as exc_info:
        recipe_hash(recipe)

    assert "Recipe was mutated after hashing. Recreate the recipe." in str(exc_info.value)


def test_mutating_operations_after_hashing_raises_value_error() -> None:
    recipe = _make_sample_recipe()
    recipe_hash(recipe)

    # Append an operation
    recipe.operations.append(Operation(type="quantization.ptq", version="1.0.0", config={"bits": 8}))

    with pytest.raises(ValueError) as exc_info:
        recipe_hash(recipe)

    assert "Recipe was mutated after hashing. Recreate the recipe." in str(exc_info.value)


def test_recreating_recipe_allows_new_hash() -> None:
    recipe1 = _make_sample_recipe()
    h1 = recipe_hash(recipe1)

    recipe2 = _make_sample_recipe()
    recipe2.metadata.name = "fresh-recipe"
    h2 = recipe_hash(recipe2)

    assert h1 != h2
    assert h2.startswith("sha256:")
