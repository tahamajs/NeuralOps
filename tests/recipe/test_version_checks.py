#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for recipe operation version checking against registry (P0-12).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from neuralops.recipe.schema import Metadata, ModelSpec, Operation, Recipe
from neuralops.recipe.versioning import check_operation_versions


def test_registered_operation_produces_no_warnings() -> None:
    # A registered normative operation like attention.replace@1.0.0
    recipe_dict = {
        "operations": [
            {"type": "attention.replace", "version": "1.0.0", "config": {}},
        ]
    }
    warnings = check_operation_versions(recipe_dict)
    # If registered, warnings should be empty
    assert len(warnings) == 0


def test_unregistered_operation_produces_warning() -> None:
    recipe_dict = {
        "operations": [
            {"type": "nonexistent.op", "version": "9.9.9", "config": {}},
        ]
    }
    warnings = check_operation_versions(recipe_dict)
    assert len(warnings) == 1
    assert warnings[0]["operation"] == "nonexistent.op"
    assert warnings[0]["version"] == "9.9.9"
    assert warnings[0]["message"] == "Operation 'nonexistent.op@9.9.9' is not registered"


def test_deprecated_operation_produces_deprecation_warning() -> None:
    mock_spec = MagicMock()
    mock_spec.is_deprecated = True
    mock_spec.deprecation_message = "Operation 'old.op@1.0.0' is deprecated; use new.op instead"

    custom_registry = {
        "old.op": {
            "1.0.0": mock_spec,
        }
    }

    recipe_dict = {
        "operations": [
            {"type": "old.op", "version": "1.0.0", "config": {}},
        ]
    }
    warnings = check_operation_versions(recipe_dict, registry=custom_registry)
    assert len(warnings) == 1
    assert warnings[0]["operation"] == "old.op"
    assert warnings[0]["version"] == "1.0.0"
    assert "deprecated" in warnings[0]["message"]


def test_recipe_object_with_mixed_operations() -> None:
    valid_op = Operation(type="attention.replace", version="1.0.0", config={})
    unregistered_op = Operation(type="custom.unknown", version="2.0.0", config={})

    recipe = Recipe(
        apiVersion="neuralops/v1",
        kind="Recipe",
        metadata=Metadata(name="test", version="1.0.0", description="test"),
        model=ModelSpec(source="test", dtype="fp16"),
        operations=[valid_op, unregistered_op],
    )

    warnings = check_operation_versions(recipe)
    assert len(warnings) == 1
    assert warnings[0]["operation"] == "custom.unknown"
    assert warnings[0]["version"] == "2.0.0"
    assert "is not registered" in warnings[0]["message"]
