#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Category 5 Conformance Tests: Validation Config per NRL v1.0.0 §14.
"""

import pytest

from neuralops.recipe.errors import RecipeSchemaError
from neuralops.recipe.loader import loads_recipe
from neuralops.validation.profiles import get_gates_for_profile


def test_cat5_case1_none_profile_requires_production_ready_false():
    # W15 rule: If profile is 'none', production_ready must be explicitly false
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: bad-none-profile
  version: 1.0.0
  description: "Profile none with production_ready true"
  production_ready: true
model:
  source: distilbert-base-uncased
  dtype: float32
validation:
  profile: "none"
operations:
  - id: op-1
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.5
"""
    with pytest.raises(RecipeSchemaError) as exc:
        loads_recipe(yaml_text)
    assert "production_ready" in str(exc.value).lower()


def test_cat5_case2_none_profile_with_explicit_false():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: valid-none-profile
  version: 1.0.0
  description: "Profile none with production_ready explicitly false"
  production_ready: false
model:
  source: distilbert-base-uncased
  dtype: float32
validation:
  profile: "none"
operations:
  - id: op-1
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.5
"""
    recipe = loads_recipe(yaml_text)
    assert recipe.validation.profile == "none"
    assert recipe.metadata.production_ready is False


def test_cat5_case3_quick_profile_gates():
    gates = get_gates_for_profile("quick")
    names = [g.name for g in gates]
    assert "shapes_match" in names
    assert "dtypes_match" in names
    assert "forward_pass_runs" in names
    assert "parameter_count" in names


def test_cat5_case4_standard_profile_gates():
    gates = get_gates_for_profile("standard")
    names = [g.name for g in gates]
    assert "no_nan_inf" in names
    assert "output_similarity" in names


def test_cat5_case5_thorough_profile_gates():
    gates = get_gates_for_profile("thorough")
    names = [g.name for g in gates]
    assert "perplexity_delta" in names
    assert len(gates) >= 7
