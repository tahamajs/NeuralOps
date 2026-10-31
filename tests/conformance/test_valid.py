#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Category 1 Conformance Tests: Valid Recipes per NRL v1.0.0 §14.
"""

from neuralops.recipe.loader import loads_recipe
from neuralops.validator.static import validate_recipe_static


def test_cat1_case1_minimal_valid():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: minimal-valid
  version: 1.0.0
  description: "Minimal valid recipe"
model:
  source: distilbert-base-uncased
  family: distilbert
  dtype: float32
operations:
  - id: op-1
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target: "*.intermediate.dense"
      target_ratio: 0.5
"""
    recipe = loads_recipe(yaml_text)
    assert recipe.apiVersion == "neuralops/v1"
    validate_recipe_static(recipe)


def test_cat1_case2_multi_stage():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: multi-stage-valid
  version: 1.0.0
  description: "Multi-stage transformation"
model:
  source: meta-llama/Llama-3-8B
  family: llama
  dtype: bfloat16
operations:
  - id: step-1-replace
    type: attention.replace
    version: 1.0.0
    config:
      target: chunk_gla
  - id: step-2-compress
    type: weights.compress
    version: 1.0.0
    depends_on: ["step-1-replace"]
    config:
      method: aasvd
      target: "*.mlp.*"
      target_ratio: 0.6
"""
    recipe = loads_recipe(yaml_text)
    assert len(recipe.operations) == 2
    validate_recipe_static(recipe)


def test_cat1_case3_full_metadata():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: full-metadata-valid
  version: 1.2.3
  description: "Recipe with full metadata and references"
  author: research@neuralops.org
  license: Apache-2.0
  tags: ["pruning", "distilbert", "v0.1"]
  production_ready: true
  references: ["https://arxiv.org/abs/2401.00000"]
model:
  source: distilbert-base-uncased
  family: distilbert
  dtype: float32
operations:
  - id: op-1
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.5
"""
    recipe = loads_recipe(yaml_text)
    assert recipe.metadata.tags == ["pruning", "distilbert", "v0.1"]
    assert recipe.metadata.production_ready is True
    validate_recipe_static(recipe)


def test_cat1_case4_rollback_execution_spec():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: rollback-spec-valid
  version: 1.0.0
  description: "Recipe with strict stage rollback"
model:
  source: gpt2
  family: gpt2
  dtype: float32
execution:
  seed: 42
  device: cpu
  rollback: true
operations:
  - id: op-1
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.7
"""
    recipe = loads_recipe(yaml_text)
    assert recipe.execution.rollback is True
    assert recipe.execution.seed == 42
    validate_recipe_static(recipe)


def test_cat1_case5_thorough_validation_spec():
    yaml_text = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: thorough-validation-valid
  version: 1.0.0
  description: "Recipe with thorough validation profile"
model:
  source: distilbert-base-uncased
  family: distilbert
  dtype: float32
validation:
  profile: thorough
  policy: fail_fast
  gates: ["shapes_match", "forward_pass_runs", "perplexity_delta"]
operations:
  - id: op-1
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.5
"""
    recipe = loads_recipe(yaml_text)
    assert recipe.validation.profile == "thorough"
    assert recipe.validation.policy == "fail_fast"
    validate_recipe_static(recipe)
