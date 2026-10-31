#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Category 6 Conformance Tests: Canonical Hashing and Normalization per NRL v1.0.0 §14.
"""

from neuralops.recipe.canonical import canonical_bytes, recipe_hash
from neuralops.recipe.loader import loads_recipe


def test_cat6_case1_key_reordering_invariance():
    yaml1 = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: order-test
  version: 1.0.0
  description: "Key ordering"
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
"""
    yaml2 = """
kind: Recipe
apiVersion: neuralops/v1
model:
  dtype: float32
  source: distilbert-base-uncased
metadata:
  description: "Key ordering"
  name: order-test
  version: 1.0.0
operations:
  - config:
      target_ratio: 0.5
      method: aasvd
    id: op-1
    type: weights.compress
    version: 1.0.0
"""
    r1 = loads_recipe(yaml1)
    r2 = loads_recipe(yaml2)
    assert recipe_hash(r1) == recipe_hash(r2)


def test_cat6_case2_nfc_unicode_normalization():
    # 'café': composed ('\u00e9') vs decomposed ('e\u0301')
    composed_desc = "caf\u00e9"
    decomposed_desc = "cafe\u0301"
    assert composed_desc != decomposed_desc  # raw strings differ in python

    yaml1 = f"""
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: nfc-test
  version: 1.0.0
  description: "{composed_desc}"
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
"""
    yaml2 = f"""
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: nfc-test
  version: 1.0.0
  description: "{decomposed_desc}"
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
"""
    r1 = loads_recipe(yaml1)
    r2 = loads_recipe(yaml2)
    assert recipe_hash(r1) == recipe_hash(r2)


def test_cat6_case3_whitespace_formatting_invariance():
    yaml1 = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: ws-test
  version: 1.0.0
  description: "Whitespace test"
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
"""
    yaml2 = """
apiVersion:   neuralops/v1
kind:     Recipe

metadata:
  name:    ws-test
  version:  1.0.0
  description:  "Whitespace test"

model:
  source:    distilbert-base-uncased
  dtype:    float32

operations:
  - id:  op-1
    type:  weights.compress
    version:  1.0.0
    config:
      method:  aasvd
      target_ratio:  0.5
"""
    r1 = loads_recipe(yaml1)
    r2 = loads_recipe(yaml2)
    assert recipe_hash(r1) == recipe_hash(r2)


def test_cat6_case4_parameter_sensitivity():
    yaml1 = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: param-sens
  version: 1.0.0
  description: "Sensitivity test"
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
"""
    yaml2 = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: param-sens
  version: 1.0.0
  description: "Sensitivity test"
model:
  source: distilbert-base-uncased
  dtype: float32
operations:
  - id: op-1
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target_ratio: 0.6
"""
    r1 = loads_recipe(yaml1)
    r2 = loads_recipe(yaml2)
    assert recipe_hash(r1) != recipe_hash(r2)


def test_cat6_case5_name_sensitivity():
    yaml1 = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: name-a
  version: 1.0.0
  description: "Name sensitivity"
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
"""
    yaml2 = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: name-b
  version: 1.0.0
  description: "Name sensitivity"
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
"""
    r1 = loads_recipe(yaml1)
    r2 = loads_recipe(yaml2)
    assert recipe_hash(r1) != recipe_hash(r2)
