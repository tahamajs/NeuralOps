#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
End-to-End Integration Test for Phase 2: plan_recipe -> execute_plan.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
import torch.nn as nn

from neuralops.pipeline_transform.executor import execute_plan
from neuralops.pipeline_transform.planner import plan_recipe
from neuralops.pipeline_transform.result import PipelineError, PipelineResult
from neuralops.recipe.loader import loads_recipe


class E2ETransformerModel(nn.Module):
    """Minimal 2-layer transformer-style network for Phase 2 E2E testing."""

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "model_type": "distilbert",
        })()
        self.transformer = nn.ModuleDict({
            "layer": nn.ModuleList([
                nn.ModuleDict({
                    "attention": nn.Linear(64, 64),
                    "intermediate": nn.Linear(64, 64),
                })
                for _ in range(2)
            ])
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.transformer["layer"]:
            x = layer["attention"](x)
            x = layer["intermediate"](x)
        return x


def _hash_model_state(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k in sorted(model.state_dict().keys()):
        h.update(k.encode("utf-8"))
        h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_e2e_phase2_safe_transformation_pipeline():
    """Verify end-to-end plan_recipe -> execute_plan execution and rollback."""
    recipe_yaml = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: e2e-distilbert-safe-compression
  version: 1.0.0
  description: "End-to-end test recipe for Phase 2 safe transformation"
model:
  source: distilbert-base-uncased
  family: distilbert
  dtype: float32
validation:
  profile: structural
operations:
  - id: compress-layer-0
    type: weights.compress
    version: 1.0.0
    config:
      target: "transformer.layer.0.intermediate"
      method: "svd"
      target_ratio: 0.75
  - id: compress-layer-1
    type: weights.compress
    version: 1.0.0
    config:
      target: "transformer.layer.1.intermediate"
      method: "svd"
      target_ratio: 0.75
    depends_on:
      - compress-layer-0
"""
    recipe = loads_recipe(recipe_yaml)
    model = E2ETransformerModel()

    # 1. Plan recipe without CUDA
    is_init_before = torch.cuda.is_initialized()
    plan = plan_recipe(recipe, model_or_config=model)
    is_init_after = torch.cuda.is_initialized()
    assert is_init_before == is_init_after

    assert plan.recipe_name == "e2e-distilbert-safe-compression"
    assert len(plan) == 2
    assert plan.recipe_hash.startswith("sha256:")

    # 2. Execute plan transactionally
    result = execute_plan(plan, model=model, device="cpu")
    assert isinstance(result, PipelineResult)
    assert result.success is True
    assert len(result.stage_records) == 2
    assert result.stage_records[0].stage_id == "compress-layer-0"
    assert result.stage_records[1].stage_id == "compress-layer-1"
    assert result.stage_records[0].validation_result.passed is True
    assert result.stage_records[1].validation_result.passed is True

    # 3. JSON serialization round-trip
    res_dict = result.to_dict()
    json_str = json.dumps(res_dict)
    loaded_res = json.loads(json_str)
    assert loaded_res["success"] is True
    assert len(loaded_res["stage_records"]) == 2

    # 4. Transactional rollback test: run a failing stage on the resulting model
    hash_before_failing = _hash_model_state(model)

    failing_yaml = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: e2e-failing-stage
  version: 1.0.0
  description: "Failing stage for rollback verification"
model:
  source: distilbert-base-uncased
  family: distilbert
  dtype: float32
validation:
  profile: structural
operations:
  - id: impossible-compression
    type: weights.compress
    version: 1.0.0
    config:
      target: "*"
      method: "svd"
      target_ratio: 0.01
"""
    failing_recipe = loads_recipe(failing_yaml)
    failing_plan = plan_recipe(failing_recipe, model_or_config=model)

    with pytest.raises(PipelineError) as exc_info:
        execute_plan(failing_plan, model=model, device="cpu")

    err = exc_info.value
    assert err.failed_stage_index == 0
    assert len(err.suggestions) >= 3

    # Check that model was rolled back bit-identically
    hash_after_rollback = _hash_model_state(model)
    assert hash_after_rollback == hash_before_failing
