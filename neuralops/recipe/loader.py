#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recipe Loader with YAML/JSON Auto-Detection and Syntax Error Handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.recipe.errors import (
    E_PARSE_DUP_KEY,
    E_PARSE_JSON_SYNTAX,
    E_PARSE_YAML_SYNTAX,
    RecipeParseError,
)
from neuralops.recipe.schema import Recipe
from neuralops.recipe.versioning import check_language_version


def _json_check_dup_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for k, v in pairs:
        if k in res:
            raise RecipeParseError(f"Duplicate key '{k}' detected in JSON payload", E_PARSE_DUP_KEY, k)
        res[k] = v
    return res


def loads_recipe(text: str, format: Optional[str] = None) -> Recipe:
    """Parse recipe from raw text (YAML or JSON)."""
    stripped = text.strip()
    if not stripped:
        raise RecipeParseError("Recipe payload is empty", E_PARSE_JSON_SYNTAX)

    # Determine format
    is_json = False
    if format == "json" or (format is None and stripped.startswith(("{", "["))):
        is_json = True

    data: Dict[str, Any]
    if is_json:
        try:
            data = json.loads(stripped, object_pairs_hook=_json_check_dup_keys)
        except RecipeParseError:
            raise
        except Exception as exc:
            raise RecipeParseError(f"Failed to parse recipe JSON: {exc}", E_PARSE_JSON_SYNTAX) from exc
    else:
        try:
            import yaml
            try:
                from yaml import CSafeLoader as SafeLoader
            except ImportError:
                from yaml import SafeLoader
        except ImportError as exc:
            raise IntegrationNotImplementedError(
                "PyYAML is required to parse YAML recipe files. "
                "Install with 'pip install pyyaml' or supply recipes in canonical JSON format."
            ) from exc

        try:
            # Custom loader to detect duplicate keys in YAML
            class UniqueKeyLoader(SafeLoader):
                pass

            def construct_mapping(loader, node, deep=False):
                loader.flatten_mapping(node)
                mapping = {}
                for key_node, value_node in node.value:
                    key = loader.construct_object(key_node, deep=deep)
                    if key in mapping:
                        raise RecipeParseError(f"Duplicate key '{key}' detected in YAML mapping", E_PARSE_DUP_KEY, str(key))
                    mapping[key] = loader.construct_object(value_node, deep=deep)
                return mapping

            UniqueKeyLoader.add_constructor(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                construct_mapping,
            )

            data = yaml.load(stripped, Loader=UniqueKeyLoader)
            if not isinstance(data, dict):
                raise RecipeParseError(f"YAML root must be a mapping, got {type(data).__name__}", E_PARSE_YAML_SYNTAX)
        except RecipeParseError:
            raise
        except Exception as exc:
            raise RecipeParseError(f"Failed to parse recipe YAML: {exc}", E_PARSE_YAML_SYNTAX) from exc

    check_language_version(data)
    recipe = Recipe.from_dict(data)
    return recipe


def load_recipe(path_or_str: str | Path) -> Recipe:
    """Load a Recipe from a file path or direct string."""
    candidate_path = Path(path_or_str)
    if candidate_path.is_file():
        text = candidate_path.read_text(encoding="utf-8")
        fmt = "json" if candidate_path.suffix.lower() == ".json" else ("yaml" if candidate_path.suffix.lower() in (".yaml", ".yml") else None)
        return loads_recipe(text, format=fmt)
    elif "\n" in str(path_or_str) or str(path_or_str).strip().startswith(("{", "apiVersion:")):
        return loads_recipe(str(path_or_str))
    else:
        # Path does not exist as file, treat as text
        return loads_recipe(str(path_or_str))
