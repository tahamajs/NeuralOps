#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Static Recipe Validator and Runtime Validation Gates.
"""

import pytest
import torch
import torch.nn as nn

from neuralops.recipe.loader import loads_recipe
from neuralops.recipe.schema import Operation
from neuralops.validation.gates.behavioral import OutputSimilarityGate, PerplexityDeltaGate
from neuralops.validation.gates.numerical import ForwardPassRunsGate, NoNanInfGate
from neuralops.validation.gates.structural import ParameterCountGate, ShapesMatchGate
from neuralops.validation.runner import run_validation_gates
from neuralops.validator.errors import StaticValidationError
from neuralops.validator.static import validate_recipe_static

VALID_RECIPE_YAML = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: test-compress
  version: 1.0.0
  description: "Test recipe description"
  author: test@neuralops.org
model:
  source: distilbert-base-uncased
  family: distilbert
  dtype: float32
operations:
  - id: step-1
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target: "*.intermediate.dense"
      target_ratio: 0.5
"""


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = nn.Module()
        self.config.hidden_size = 32
        self.config.vocab_size = 100
        self.emb = nn.Embedding(100, 32)
        self.linear = nn.Linear(32, 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.int32, torch.int64):
            x = self.emb(x)
        return self.linear(x)


def test_static_validation_valid():
    recipe = loads_recipe(VALID_RECIPE_YAML)
    warnings = validate_recipe_static(recipe)
    assert isinstance(warnings, list)


def test_static_validation_w3_empty_operations():
    recipe = loads_recipe(VALID_RECIPE_YAML)
    recipe.operations = []
    with pytest.raises(StaticValidationError) as exc:
        validate_recipe_static(recipe)
    assert "operations" in str(exc.value).lower()


def test_static_validation_w15_duplicate_id():
    recipe = loads_recipe(VALID_RECIPE_YAML)
    dup = Operation(
        id="step-1",
        type="weights.compress",
        version="1.0.0",
        config={"method": "aasvd", "target_ratio": 0.5},
    )
    recipe.operations.append(dup)
    with pytest.raises(StaticValidationError) as exc:
        validate_recipe_static(recipe)
    assert "duplicate" in str(exc.value).lower()


def test_static_validation_w4_incompatible_version():
    recipe = loads_recipe(VALID_RECIPE_YAML)
    recipe.apiVersion = "neuralops/v2"
    with pytest.raises(StaticValidationError) as exc:
        validate_recipe_static(recipe)
    assert "unsupported recipe language version" in str(exc.value).lower()


def test_validation_gates_l1_and_l2():
    m1 = SmallModel()
    m2 = SmallModel()
    m2.load_state_dict(m1.state_dict())

    res_shapes = ShapesMatchGate().run(m1, m2)
    assert res_shapes.passed

    res_fwd = ForwardPassRunsGate().run(m1)
    assert res_fwd.passed

    res_nan = NoNanInfGate().run(m1)
    assert res_nan.passed

    res_param = ParameterCountGate().run(m1, m2, context={"target_ratio": 1.0})
    assert res_param.passed


def test_validation_gates_l3():
    m1 = SmallModel()
    m2 = SmallModel()
    m2.load_state_dict(m1.state_dict())

    res_sim = OutputSimilarityGate(min_similarity=0.99).run(m1, m2)
    assert res_sim.passed

    res_ppl = PerplexityDeltaGate(max_relative_increase=0.05).run(m1, m2)
    assert res_ppl.passed


def test_run_validation_gates_suite():
    m1 = SmallModel()
    m2 = SmallModel()
    m2.load_state_dict(m1.state_dict())

    v_res = run_validation_gates(model=m1, baseline_model=m2, profile="quick")
    assert v_res.passed
    assert v_res.total_gates >= 4
    assert v_res.failed_gates == 0
