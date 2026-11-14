#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying loader raises E_PARSE_EMPTY on empty payload (P1).
"""

from __future__ import annotations

import pytest

from neuralops.recipe.errors import E_PARSE_EMPTY, RecipeParseError
from neuralops.recipe.loader import loads_recipe


def test_empty_string_raises_e_parse_empty() -> None:
    with pytest.raises(RecipeParseError) as exc_info:
        loads_recipe("")
    assert exc_info.value.error_code == E_PARSE_EMPTY
    assert "empty" in str(exc_info.value).lower()


def test_whitespace_only_raises_e_parse_empty() -> None:
    with pytest.raises(RecipeParseError) as exc_info:
        loads_recipe("   \n\t  \r\n   ")
    assert exc_info.value.error_code == E_PARSE_EMPTY


def test_explicit_format_empty_raises_e_parse_empty() -> None:
    with pytest.raises(RecipeParseError) as exc_json:
        loads_recipe("", format="json")
    assert exc_json.value.error_code == E_PARSE_EMPTY

    with pytest.raises(RecipeParseError) as exc_yaml:
        loads_recipe("", format="yaml")
    assert exc_yaml.value.error_code == E_PARSE_EMPTY
