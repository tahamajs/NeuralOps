#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Category 3 Conformance Tests: Registry Violations per NRL v1.0.0 §14.
"""

import pytest

from neuralops.operations.errors import (
    E_REGISTRY_DUPLICATE,
    E_REGISTRY_UNKNOWN_OP,
    E_REGISTRY_UNKNOWN_VERSION,
    OperationError,
)
from neuralops.operations.registry import get_operation, register_operation
from neuralops.recipe.loader import loads_recipe
from neuralops.validation.errors import (
    E_W7_INVALID_OPERATION_PARAMS,
    StaticValidationError,
)
from neuralops.validation.static import validate_recipe_static


def test_cat3_case1_unknown_operation():
    with pytest.raises(OperationError) as exc:
        get_operation("nonexistent.operation", "1.0.0")
    assert exc.value.error_code == E_REGISTRY_UNKNOWN_OP


def test_cat3_case2_unknown_version():
    with pytest.raises(OperationError) as exc:
        get_operation("weights.compress", "99.0.0")
    assert exc.value.error_code == E_REGISTRY_UNKNOWN_VERSION


def test_cat3_case3_missing_required_param():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: missing-param
  version: 1.0.0
  description: "Missing required parameter target_ratio"
model:
  source: distilbert-base-uncased
  dtype: float32
operations:
  - id: op-1
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
"""
    recipe = loads_recipe(yaml_text)
    with pytest.raises(StaticValidationError) as exc:
        validate_recipe_static(recipe)
    assert exc.value.error_code == E_W7_INVALID_OPERATION_PARAMS


def test_cat3_case4_unrecognized_param():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: extra-param
  version: 1.0.0
  description: "Unrecognized extra parameter"
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
      nonexistent_param: 1234
"""
    recipe = loads_recipe(yaml_text)
    with pytest.raises(StaticValidationError) as exc:
        validate_recipe_static(recipe)
    assert exc.value.error_code == E_W7_INVALID_OPERATION_PARAMS


def test_cat3_case5_duplicate_registration():
    from neuralops.operations.weights.svd_aasvd import WeightsCompressAASVDOperation
    with pytest.raises(OperationError) as exc:
        register_operation(override=False)(WeightsCompressAASVDOperation)
    assert exc.value.error_code == E_REGISTRY_DUPLICATE
