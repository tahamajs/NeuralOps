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

from neuralops.recipe.errors import (
    E_LANG_VERSION_TOO_NEW,
    E_LANG_VERSION_TOO_OLD,
    RecipeSchemaError,
)

logger = logging.getLogger("NeuralOps.Recipe.Versioning")

SUPPORTED_API_VERSIONS = {"neuralops/v1"}
HISTORICAL_VERSIONS = {"neuralops/v0"}


def check_language_version(recipe_or_dict: Any) -> None:
    """Verify that the recipe's apiVersion is supported by this runtime."""
    if hasattr(recipe_or_dict, "apiVersion") and recipe_or_dict.apiVersion:
        version = recipe_or_dict.apiVersion
    elif hasattr(recipe_or_dict, "version") and recipe_or_dict.version:
        version = recipe_or_dict.version
    elif isinstance(recipe_or_dict, dict):
        version = recipe_or_dict.get("apiVersion") or recipe_or_dict.get("version")
    else:
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
    warnings = []
    # Stored for future deprecation checks
    return warnings
