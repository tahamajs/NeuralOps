#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying execute_plan validation gates compare against the real pre-state,
rather than comparing self to self.
"""

from unittest.mock import patch
import torch
import torch.nn as nn

from neuralops.pipeline_transform import executor
from neuralops.pipeline_transform.executor import execute_plan
from neuralops.pipeline_transform.planner import plan_recipe
from neuralops.recipe.schema import Metadata, ModelSpec, Operation, Recipe


class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 64, "num_hidden_layers": 1})()
        self.intermediate = nn.ModuleDict(
            {
                "dense1": nn.Linear(64, 64),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.intermediate["dense1"](x)


def test_executor_validation_compares_against_real_pre_state():
    """Verify run_validation_gates receives a distinct pre-mutation copy of the model."""
    captured_before = []
    captured_after = []

    real_run_validation_gates = executor.run_validation_gates

    def spy_run_validation_gates(*args, **kwargs):
        mb = kwargs.get("model_before") if "model_before" in kwargs else (args[0] if len(args) > 0 else None)
        ma = kwargs.get("model_after") if "model_after" in kwargs else (args[1] if len(args) > 1 else None)
        captured_before.append(mb)
        captured_after.append(ma)
        return real_run_validation_gates(*args, **kwargs)

    recipe = Recipe(
        apiVersion="neuralops/v1",
        kind="Recipe",
        metadata=Metadata(name="val-test", version="1.0.0", description="Validation test"),
        model=ModelSpec(family="distilbert", source="distilbert-base-uncased", dtype="float32"),
        operations=[
            Operation(
                id="c1",
                type="weights.compress",
                version="1.0.0",
                config={"method": "svd", "target_ratio": 0.8, "target": "intermediate.dense1"},
            )
        ],
    )
    plan = plan_recipe(recipe)
    model = SimpleNet()

    with patch("neuralops.pipeline_transform.executor.run_validation_gates", side_effect=spy_run_validation_gates):
        execute_plan(plan, model, device="cpu")

    assert len(captured_before) == 1
    assert len(captured_after) == 1
    # model_before and model_after must be different objects, not self-to-self
    assert captured_before[0] is not captured_after[0]
