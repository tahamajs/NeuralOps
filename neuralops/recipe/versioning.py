#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recipe Language Version Checking and Policy.

Implements Specification Section 10 (Versioning Policy).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from neuralops.recipe.errors import E_LANG_VERSION_TOO_NEW, E_LANG_VERSION_TOO_OLD, RecipeSchemaError

logger = logging.getLogger("NeuralOps.Recipe.Versioning")

SUPPORTED_API_VERSIONS = {"neuralops/v1"}
HISTORICAL_VERSIONS = {"neuralops/v0"}


def check_language_version(recipe_or_dict: Any) -> None:
    """Verify that the recipe's apiVersion is supported by this runtime."""
    if isinstance(recipe_or_dict, dict):
        version = recipe_or_dict.get("apiVersion") or recipe_or_dict.get("version")
    else:
        version = getattr(recipe_or_dict, "apiVersion", None)

    if not version:
        return

    if version in HISTORICAL_VERSIONS:
        raise RecipeSchemaError(
            f"Recipe apiVersion '{version}' is deprecated and no longer supported",
            E_LANG_VERSION_TOO_OLD,
            "apiVersion",
        )
    elif version not in SUPPORTED_API_VERSIONS:
        raise RecipeSchemaError(
            f"Recipe apiVersion '{version}' is unsupported by this runtime. Expected 'neuralops/v1'",
            E_LANG_VERSION_TOO_NEW,
            "apiVersion",
        )


def check_operation_versions(recipe_or_dict: Any, registry: Any = None) -> list[Dict[str, str]]:
    """Inspect all operations in recipe and return deprecation warnings if any."""
    if registry is None:
        from neuralops.operations.registry import OPERATION_REGISTRY

        registry = OPERATION_REGISTRY

    warnings: list[Dict[str, str]] = []

    # Extract operations list from dict, Recipe object, or direct list
    if isinstance(recipe_or_dict, dict):
        raw_ops = recipe_or_dict.get("operations", [])
    elif hasattr(recipe_or_dict, "operations"):
        raw_ops = recipe_or_dict.operations
    elif isinstance(recipe_or_dict, (list, tuple)):
        raw_ops = list(recipe_or_dict)
    else:
        raw_ops = []

    if not isinstance(raw_ops, (list, tuple)):
        return warnings

    for op in raw_ops:
        if isinstance(op, dict):
            op_type = op.get("type")
            op_version = op.get("version")
        else:
            op_type = getattr(op, "type", None)
            op_version = getattr(op, "version", None)

        if not op_type or not op_version:
            continue

        # Check in registry: support both dict-of-dicts and flat "name@version" formats
        spec: Any = None
        if isinstance(registry, dict):
            if op_type in registry and isinstance(registry[op_type], dict):
                spec = registry[op_type].get(op_version)
            elif f"{op_type}@{op_version}" in registry:
                spec = registry[f"{op_type}@{op_version}"]
        elif hasattr(registry, "get"):
            spec = registry.get(op_type, {}).get(op_version)

        if spec is None:
            warnings.append(
                {
                    "operation": str(op_type),
                    "version": str(op_version),
                    "message": f"Operation '{op_type}@{op_version}' is not registered",
                }
            )
        elif getattr(spec, "is_deprecated", False):
            dep_msg = (
                getattr(spec, "deprecation_message", None)
                or getattr(spec, "deprecation_reason", None)
                or f"Operation '{op_type}@{op_version}' is deprecated"
            )
            warnings.append(
                {
                    "operation": str(op_type),
                    "version": str(op_version),
                    "message": str(dep_msg),
                }
            )

    return warnings
