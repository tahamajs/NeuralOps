#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying ModelSpec.from_dict accepts branch names, tags, and short SHAs unless strict_sha=True (P1).
"""

from __future__ import annotations

import pytest

from neuralops.recipe.errors import RecipeSchemaError
from neuralops.recipe.schema import ModelSpec, Recipe


def test_model_spec_accepts_branch_name() -> None:
    spec = ModelSpec.from_dict({"source": "meta-llama/Llama-2-7b", "dtype": "bfloat16", "revision": "main"})
    assert spec.revision == "main"


def test_model_spec_accepts_tag() -> None:
    spec = ModelSpec.from_dict({"source": "google/bert_uncased_L-2_H-128_A-2", "dtype": "fp32", "revision": "v1.0.0"})
    assert spec.revision == "v1.0.0"


def test_model_spec_accepts_short_sha() -> None:
    spec = ModelSpec.from_dict(
        {"source": "hf-internal-testing/tiny-random-bert", "dtype": "fp16", "revision": "7f8b9c0"}
    )
    assert spec.revision == "7f8b9c0"


def test_model_spec_accepts_full_40_char_sha() -> None:
    full_sha = "0123456789abcdef0123456789abcdef01234567"
    spec = ModelSpec.from_dict({"source": "some/model", "dtype": "fp16", "revision": full_sha})
    assert spec.revision == full_sha


def test_model_spec_rejects_non_sha_when_strict_sha_is_true() -> None:
    with pytest.raises(RecipeSchemaError) as exc_info:
        ModelSpec.from_dict(
            {"source": "some/model", "dtype": "fp16", "revision": "main"},
            strict_sha=True,
        )
    assert "must be a full 40-character commit SHA" in str(exc_info.value)


def test_model_spec_rejects_empty_revision() -> None:
    with pytest.raises(RecipeSchemaError) as exc_info:
        ModelSpec.from_dict({"source": "some/model", "dtype": "fp16", "revision": "   "})
    assert "must be non-empty string" in str(exc_info.value)


def test_recipe_from_dict_accepts_branch_name() -> None:
    recipe_dict = {
        "apiVersion": "neuralops/v1",
        "kind": "Recipe",
        "metadata": {
            "name": "test-recipe",
            "version": "1.0.0",
            "description": "test",
        },
        "model": {
            "source": "bert-base-uncased",
            "dtype": "fp16",
            "revision": "main",
        },
        "operations": [
            {
                "type": "attention.replace",
                "version": "1.0.0",
                "config": {"mechanism": "cosformer"},
            }
        ],
    }

    recipe = Recipe.from_dict(recipe_dict)
    assert recipe.model.revision == "main"
