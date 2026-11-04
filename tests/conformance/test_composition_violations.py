#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Category 4 Conformance Tests: Composition Violations per NRL v1.0.0 §14.
"""

import pytest

from neuralops.executor.planner import plan_recipe
from neuralops.recipe.loader import loads_recipe
from neuralops.recipe.schema import Operation
from neuralops.validation.errors import (
    E_W3_EMPTY_OPERATIONS,
    E_W10_TYPE_CHAIN_MISMATCH,
    E_W12_COMPOSABILITY_VIOLATION,
    E_W15_DUPLICATE_OPERATION_ID,
    StaticValidationError,
)
from neuralops.validation.static import validate_recipe_static


def test_cat4_case1_duplicate_op_id():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: dup-id
  version: 1.0.0
  description: "Duplicate op ID"
model:
  source: distilbert-base-uncased
  dtype: float32
operations:
  - id: op-1
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.5
  - id: op-1
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.6
"""
    recipe = loads_recipe(yaml_text)
    with pytest.raises(StaticValidationError) as exc:
        validate_recipe_static(recipe)
    assert exc.value.error_code == E_W15_DUPLICATE_OPERATION_ID


def test_cat4_case2_empty_operations():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: empty-ops
  version: 1.0.0
  description: "Empty operations"
model:
  source: distilbert-base-uncased
  dtype: float32
operations: []
"""
    # Empty operations caught by schema or static validator
    with pytest.raises(Exception) as exc:
        loads_recipe(yaml_text)
    assert "operations" in str(exc.value).lower()


def test_cat4_case3_non_composable_self():
    # attention.replace has composable_with_self = False
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: non-composable
  version: 1.0.0
  description: "Illegal self composition"
model:
  source: distilbert-base-uncased
  dtype: float32
operations:
  - id: op-1
    type: attention.replace
    version: 1.0.0
    config:
      target: chunk_gla
  - id: op-2
    type: attention.replace
    version: 1.0.0
    config:
      target: chunk_gla
"""
    # Note: If second attention.replace runs, it triggers composability violation or type mismatch
    recipe = loads_recipe(yaml_text)
    with pytest.raises(StaticValidationError) as exc:
        validate_recipe_static(recipe)
    assert exc.value.error_code in (E_W12_COMPOSABILITY_VIOLATION, E_W10_TYPE_CHAIN_MISMATCH)


def test_cat4_case4_unresolved_dependency():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: bad-dep
  version: 1.0.0
  description: "Unresolved dependency in DAG"
model:
  source: distilbert-base-uncased
  dtype: float32
operations:
  - id: op-2
    type: weights.compress
    version: 1.0.0
    depends_on: ["nonexistent-prior-op"]
    config:
      method: aasvd
      target_ratio: 0.5
"""
    recipe = loads_recipe(yaml_text)
    with pytest.raises(StaticValidationError) as exc:
        plan_recipe(recipe)
    assert "E_PLAN_UNRESOLVED_DEPENDENCY" in str(exc.value)


def test_cat4_case5_type_chain_mismatch():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: type-chain-mismatch
  version: 1.0.0
  description: "Illegal state transition"
model:
  source: distilbert-base-uncased
  dtype: float32
operations:
  - id: op-1
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.5
  - id: op-2
    type: attention.replace
    version: 1.0.0
    config:
      target: chunk_gla
"""
    # weights.compress sets state to 'compressed', but attention.replace expects 'loaded'
    recipe = loads_recipe(yaml_text)
    with pytest.raises(StaticValidationError) as exc:
        validate_recipe_static(recipe)
    assert exc.value.error_code == E_W10_TYPE_CHAIN_MISMATCH
