#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recipe YAML/JSON Serialization, Deserialization, and Hub Publishing Engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from neuralops.surgery.pipeline import SurgeryRecipe


def validate_recipe_schema(yaml_dict: dict[str, Any]) -> list[str]:
    """Validate raw dictionary against NeuralOps recipe schema v1.0."""
    errors: list[str] = []
    if not isinstance(yaml_dict, dict):
        return ["Recipe must be a valid dictionary/mapping."]

    if "surgeries" not in yaml_dict:
        errors.append("Missing required field 'surgeries'.")
    elif not isinstance(yaml_dict["surgeries"], list):
        errors.append("Field 'surgeries' must be a list of operation specs.")
    else:
        for idx, op in enumerate(yaml_dict["surgeries"]):
            if not isinstance(op, dict):
                errors.append(f"Surgery #{idx} must be a dictionary.")
            elif "name" not in op:
                errors.append(f"Surgery #{idx} missing required 'name' key.")

    if "recovery" in yaml_dict and not isinstance(yaml_dict["recovery"], list):
        errors.append("Field 'recovery' must be a list of operation specs if provided.")

    return errors


def save_recipe(recipe: SurgeryRecipe, path: str | Path) -> None:
    """Save recipe to a local YAML or JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = recipe.to_dict()

    if p.suffix.lower() == ".json":
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    else:
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(d, f, sort_keys=False, default_flow_style=False)


def load_recipe(path: str | Path) -> SurgeryRecipe:
    """Load and parse recipe from a local YAML or JSON file."""
    from neuralops.surgery.pipeline import SurgeryRecipe

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Recipe file not found: {path}")

    with open(p, encoding="utf-8") as f:
        if p.suffix.lower() == ".json":
            d = json.load(f)
        else:
            d = yaml.safe_load(f)

    errors = validate_recipe_schema(d)
    if errors:
        raise ValueError(f"Invalid recipe schema in {path}: {'; '.join(errors)}")

    return SurgeryRecipe.from_dict(d)


def save_recipe_to_hub(
    recipe: SurgeryRecipe,
    repo_id: str,
    filename: str = "recipe.yaml",
    commit_message: str = "Upload NeuralOps surgery recipe",
) -> None:
    """Upload recipe directly to a HuggingFace Hub dataset or model repository."""
    import tempfile

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name
        save_recipe(recipe, tmp_path)
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=filename,
            repo_id=repo_id,
            commit_message=commit_message,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to publish recipe to HuggingFace Hub '{repo_id}': {exc}") from exc


__all__ = [
    "validate_recipe_schema",
    "save_recipe",
    "load_recipe",
    "save_recipe_to_hub",
]
