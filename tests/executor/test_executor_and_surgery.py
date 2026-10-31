#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Recipe Executor, Transactional Rollback, and ModelSurgery API.
"""

import pytest
import torch
import torch.nn as nn

from neuralops.core.low_rank.svd_linear import SVDLinear
from neuralops.executor.executor import execute_recipe
from neuralops.executor.planner import plan_recipe
from neuralops.recipe.loader import loads_recipe
from neuralops.surgery import ModelSurgery


class SmallMLP(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.config = nn.Module()
        self.config.hidden_size = hidden
        self.dense1 = nn.Linear(hidden, hidden * 2)
        self.dense2 = nn.Linear(hidden * 2, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense2(torch.relu(self.dense1(x)))


SAMPLE_RECIPE_YAML = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: test-aasvd-exec
  version: 1.0.0
  description: "Test execution recipe"
  author: test@neuralops.org
model:
  source: test-model
  family: generic
  dtype: float32
operations:
  - id: compress-dense
    type: weights.compress
    version: 1.0.0
    config:
      method: aasvd
      target: "*dense*"
      target_ratio: 0.5
      rank: auto
validation:
  profile: quick
"""


def test_planner_creates_plan():
    recipe = loads_recipe(SAMPLE_RECIPE_YAML)
    plan = plan_recipe(recipe)
    assert len(plan.stages) == 1
    assert plan.stages[0].id == "compress-dense"
    assert plan.stages[0].operation_type == "weights.compress"


def test_execute_recipe_end_to_end():
    model = SmallMLP(32)
    x = torch.randn(2, 4, 32)
    out_before = model(x)

    recipe = loads_recipe(SAMPLE_RECIPE_YAML)
    result = execute_recipe(recipe, model)

    assert result.success
    assert not result.rolled_back
    assert len(result.stage_records) == 1
    assert result.stage_records[0].success

    out_after = result.model(x)
    assert out_after.shape == out_before.shape
    assert isinstance(result.model.dense1, SVDLinear)


def test_modelsurgery_fluent_api():
    model = SmallMLP(32)
    s = ModelSurgery(model)
    orig_params = sum(p.numel() for p in s.model.parameters())

    s.apply("weights.compress[method=aasvd]", target="*dense*", target_ratio=0.5, rank=8)
    new_params = sum(p.numel() for p in s.model.parameters())
    assert new_params < orig_params
    assert len(s.get_evidence()) == 1

    val = s.validate(profile="quick")
    assert val.passed

    # Rollback restores parameters
    s.rollback()
    restored_params = sum(p.numel() for p in s.model.parameters())
    assert restored_params == orig_params
    assert isinstance(s.model.dense1, nn.Linear)


def test_modelsurgery_from_recipe():
    model = SmallMLP(32)
    recipe = loads_recipe(SAMPLE_RECIPE_YAML)
    s = ModelSurgery.from_recipe(recipe, model=model)
    s.apply()
    assert len(s.get_evidence()) >= 1
    assert isinstance(s.model.dense1, SVDLinear)
