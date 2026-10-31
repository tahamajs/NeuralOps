#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Category 2 Conformance Tests: Schema Violations per NRL v1.0.0 §14.
"""

import pytest

from neuralops.recipe.errors import (
    E_SCHEMA_INVALID_IDENTIFIER,
    E_SCHEMA_INVALID_SEMVER,
    E_SCHEMA_MISSING_FIELD,
    E_SCHEMA_UNKNOWN_FIELD,
    E_SCHEMA_WRONG_TYPE,
    RecipeSchemaError,
)
from neuralops.recipe.loader import loads_recipe


def test_cat2_case1_missing_root_field():
    yaml_text = """
kind: Recipe
metadata:
  name: missing-api-version
  version: 1.0.0
  description: "Missing apiVersion"
model:
  source: distilbert-base-uncased
  dtype: float32
operations:
  - type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.5
"""
    with pytest.raises(RecipeSchemaError) as exc:
        loads_recipe(yaml_text)
    assert exc.value.error_code in (E_SCHEMA_MISSING_FIELD, "E_LANG_VERSION_TOO_NEW")


def test_cat2_case2_unknown_root_field():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
unexpected_field: "illegal"
metadata:
  name: unknown-root
  version: 1.0.0
  description: "Unknown root field"
model:
  source: distilbert-base-uncased
  dtype: float32
operations:
  - type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.5
"""
    with pytest.raises(RecipeSchemaError) as exc:
        loads_recipe(yaml_text)
    assert exc.value.error_code == E_SCHEMA_UNKNOWN_FIELD


def test_cat2_case3_invalid_dns_identifier():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: Invalid_Identifier_Uppercase!
  version: 1.0.0
  description: "Invalid DNS name"
model:
  source: distilbert-base-uncased
  dtype: float32
operations:
  - type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.5
"""
    with pytest.raises(RecipeSchemaError) as exc:
        loads_recipe(yaml_text)
    assert exc.value.error_code == E_SCHEMA_INVALID_IDENTIFIER


def test_cat2_case4_invalid_semver():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: invalid-semver
  version: "v1.0"
  description: "Invalid SemVer"
model:
  source: distilbert-base-uncased
  dtype: float32
operations:
  - type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.5
"""
    with pytest.raises(RecipeSchemaError) as exc:
        loads_recipe(yaml_text)
    assert exc.value.error_code == E_SCHEMA_INVALID_SEMVER


def test_cat2_case5_invalid_revision_sha():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: invalid-sha
  version: 1.0.0
  description: "Invalid commit SHA"
model:
  source: distilbert-base-uncased
  dtype: float32
  revision: "not-a-40-character-sha"
operations:
  - type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.5
"""
    with pytest.raises(RecipeSchemaError) as exc:
        loads_recipe(yaml_text)
    assert exc.value.error_code == E_SCHEMA_WRONG_TYPE
