#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for surgery recipe serialization, validation, and loading.
"""

import tempfile
from pathlib import Path

import pytest

from neuralops.surgery.pipeline import SurgeryOp, SurgeryRecipe
from neuralops.surgery.recipe import (
    load_recipe,
    save_recipe,
    validate_recipe_schema,
)


def test_recipe_yaml_round_trip():
    r = SurgeryRecipe(
        ops=[
            SurgeryOp(name="replace_attention", params={"attention": "chunk_gla"}),
            SurgeryOp(name="shrink_mlp", params={"ratio": 0.7}),
        ],
        recovery=[
            SurgeryOp(name="distill_from_teacher", params={"steps": 500}),
        ],
        name="test-recipe",
        base_model="meta-llama/Llama-3.1-8B",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "recipe.yaml"
        json_path = Path(tmpdir) / "recipe.json"

        save_recipe(r, yaml_path)
        save_recipe(r, json_path)

        loaded_yaml = load_recipe(yaml_path)
        loaded_json = load_recipe(json_path)

        assert loaded_yaml.to_dict() == r.to_dict()
        assert loaded_json.to_dict() == r.to_dict()
        assert len(loaded_yaml.ops) == 2
        assert len(loaded_yaml.recovery) == 1


def test_recipe_schema_validation():
    valid_dict = {
        "recipe_version": "1.0",
        "surgeries": [{"name": "replace_attention", "params": {}}],
    }
    assert validate_recipe_schema(valid_dict) == []

    invalid_dict = {"recipe_version": "1.0"}
    errors = validate_recipe_schema(invalid_dict)
    assert len(errors) > 0
    assert "Missing required field 'surgeries'" in errors[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "invalid.yaml"
        p.write_text("invalid: true", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid recipe schema"):
            load_recipe(p)
