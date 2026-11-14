#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying check_language_version does not treat metadata.version as apiVersion (P1).
"""

from __future__ import annotations

import pytest

from neuralops.recipe.errors import E_LANG_VERSION_TOO_NEW, E_LANG_VERSION_TOO_OLD, RecipeSchemaError
from neuralops.recipe.schema import Metadata, ModelSpec, Recipe
from neuralops.recipe.versioning import check_language_version


def test_metadata_instance_does_not_raise() -> None:
    meta = Metadata(
        name="test-recipe",
        version="1.0.0",
        description="A recipe with SemVer version in metadata",
    )
    # Previously, this checked hasattr(meta, 'version') and raised because '1.0.0' is not 'neuralops/v1'
    check_language_version(meta)


def test_arbitrary_object_without_api_version_does_not_raise() -> None:
    class DummyObj:
        version = "0.2.0"

    check_language_version(DummyObj())


def test_recipe_instance_validates_api_version() -> None:
    recipe = Recipe(
        apiVersion="neuralops/v1",
        kind="Recipe",
        metadata=Metadata(name="test", version="1.0.0", description="desc"),
        model=ModelSpec(source="test", dtype="fp16"),
        operations=[],
    )
    check_language_version(recipe)


def test_dict_with_api_version_or_version() -> None:
    # apiVersion takes precedence
    check_language_version({"apiVersion": "neuralops/v1", "version": "1.0.0"})
    # fallback to version in dict
    check_language_version({"version": "neuralops/v1"})


def test_dict_with_unsupported_version_raises() -> None:
    with pytest.raises(RecipeSchemaError) as exc_info:
        check_language_version({"apiVersion": "neuralops/v99"})
    assert exc_info.value.error_code == E_LANG_VERSION_TOO_NEW

    with pytest.raises(RecipeSchemaError) as exc_info_old:
        check_language_version({"apiVersion": "neuralops/v0"})
    assert exc_info_old.value.error_code == E_LANG_VERSION_TOO_OLD
