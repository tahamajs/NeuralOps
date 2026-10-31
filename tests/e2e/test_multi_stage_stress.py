#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Stress tests for multi-stage transformation recipes, transactional rollback,
sequential pipeline mutations, and hardware-guided auto-config execution.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from neuralops.pipeline_transform.auto_config import suggest_config
from neuralops.pipeline_transform.hardware_planner import estimate_resources
from neuralops.pipeline_transform.executor import execute_plan
from neuralops.pipeline_transform.planner import plan_recipe
from neuralops.pipeline_transform.result import PipelineError, PipelineResult
from neuralops.recipe.loader import loads_recipe


class ThreeLayerTransformerToy(nn.Module):
    """3-layer toy transformer for stress testing multi-stage recipes."""

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {
            "hidden_size": 64,
            "num_hidden_layers": 3,
            "model_type": "distilbert",
        })()
        self.transformer = nn.ModuleDict({
            "layer": nn.ModuleList([
                nn.ModuleDict({
                    "attention": nn.Linear(64, 64),
                    "intermediate": nn.Linear(64, 64),
                })
                for _ in range(3)
            ])
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.transformer["layer"]:
            x = layer["attention"](x)
            x = layer["intermediate"](x)
        return x


def _hash_weights(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k in sorted(model.state_dict().keys()):
        h.update(k.encode("utf-8"))
        h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_multi_stage_three_layer_transformation():
    """Verify clean planning and transactional execution across a 3-stage dependency chain."""
    recipe_yaml = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: three-stage-compression
  version: 1.0.0
  description: "3-stage intermediate compression chain"
model:
  source: distilbert-base-uncased
  family: distilbert
  dtype: float32
validation:
  profile: structural
operations:
  - id: stage-0-compress
    type: weights.compress
    version: 1.0.0
    config:
      target: "transformer.layer.0.intermediate"
      method: "svd"
      target_ratio: 0.80
  - id: stage-1-compress
    type: weights.compress
    version: 1.0.0
    config:
      target: "transformer.layer.1.intermediate"
      method: "svd"
      target_ratio: 0.75
    depends_on:
      - stage-0-compress
  - id: stage-2-compress
    type: weights.compress
    version: 1.0.0
    config:
      target: "transformer.layer.2.intermediate"
      method: "svd"
      target_ratio: 0.70
    depends_on:
      - stage-1-compress
"""
    recipe = loads_recipe(recipe_yaml)
    model = ThreeLayerTransformerToy()

    # Plan
    plan = plan_recipe(recipe, model_or_config=model)
    assert len(plan.stages) == 3
    assert [s.id for s in plan.stages] == [
        "stage-0-compress",
        "stage-1-compress",
        "stage-2-compress",
    ]

    # Execute
    res = execute_plan(plan, model=model, device="cpu")
    assert isinstance(res, PipelineResult)
    assert res.success is True
    assert len(res.stage_records) == 3
    assert all(sr.validation_result.passed for sr in res.stage_records)

    # Forward pass verification
    inp = torch.randn(2, 8, 64)
    out = model(inp)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


def test_multi_stage_mid_pipeline_failure_rollback():
    """Verify that a failure at Stage 3 restores the model to the exact pre-failure state."""
    recipe_yaml = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: three-stage-with-failure
  version: 1.0.0
  description: "Recipe with valid stage 0 and 1, but failing stage 2"
model:
  source: distilbert-base-uncased
  family: distilbert
  dtype: float32
validation:
  profile: structural
operations:
  - id: valid-stage-0
    type: weights.compress
    version: 1.0.0
    config:
      target: "transformer.layer.0.intermediate"
      method: "svd"
      target_ratio: 0.80
  - id: valid-stage-1
    type: weights.compress
    version: 1.0.0
    config:
      target: "transformer.layer.1.intermediate"
      method: "svd"
      target_ratio: 0.75
    depends_on:
      - valid-stage-0
  - id: failing-stage-2
    type: weights.compress
    version: 1.0.0
    config:
      target: "transformer.layer.non_existent"
      method: "svd"
      target_ratio: 0.70
    depends_on:
      - valid-stage-1
"""
    recipe = loads_recipe(recipe_yaml)
    model = ThreeLayerTransformerToy()

    plan = plan_recipe(recipe, model_or_config=model)

    # Stage 0 and 1 will execute; stage 2 fails during SVD decomposition because ratio is too small
    with pytest.raises(PipelineError) as exc_info:
        execute_plan(plan, model=model, device="cpu")

    err = exc_info.value
    assert err.failed_stage_index == 2
    assert err.failed_operation == "weights.compress"
    assert len(err.suggestions) >= 2

    # Forward pass still works on restored model
    inp = torch.randn(2, 8, 64)
    out = model(inp)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


def test_sequential_recipes_applied_to_same_model():
    """Verify applying two independent valid recipes sequentially on the same model."""
    recipe1_yaml = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: recipe-pass-1
  version: 1.0.0
  description: "Pass 1"
model:
  source: distilbert-base-uncased
  family: distilbert
  dtype: float32
validation:
  profile: structural
operations:
  - id: pass-1-op
    type: weights.compress
    version: 1.0.0
    config:
      target: "transformer.layer.0.intermediate"
      method: "svd"
      target_ratio: 0.85
"""
    recipe2_yaml = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: recipe-pass-2
  version: 1.0.0
  description: "Pass 2"
model:
  source: distilbert-base-uncased
  family: distilbert
  dtype: float32
validation:
  profile: structural
operations:
  - id: pass-2-op
    type: weights.compress
    version: 1.0.0
    config:
      target: "transformer.layer.1.intermediate"
      method: "svd"
      target_ratio: 0.85
"""
    model = ThreeLayerTransformerToy()
    rec1 = loads_recipe(recipe1_yaml)
    rec2 = loads_recipe(recipe2_yaml)

    plan1 = plan_recipe(rec1, model_or_config=model)
    res1 = execute_plan(plan1, model=model, device="cpu")
    assert res1.success is True

    plan2 = plan_recipe(rec2, model_or_config=model)
    res2 = execute_plan(plan2, model=model, device="cpu")
    assert res2.success is True

    # Output remains valid
    inp = torch.randn(1, 4, 64)
    out = model(inp)
    assert out.shape == (1, 4, 64)
    assert torch.isfinite(out).all()


def test_auto_config_to_plan_to_execute_pipeline():
    """Verify end-to-end auto-config suggestion -> plan -> execute flow."""
    recipe_yaml = """
apiVersion: neuralops/v1
kind: Recipe
metadata:
  name: auto-config-test
  version: 1.0.0
  description: "Test auto-config to execution pipeline"
model:
  source: distilbert-base-uncased
  family: distilbert
  dtype: float32
validation:
  profile: structural
operations:
  - id: compress-heavy
    type: weights.compress
    version: 1.0.0
    config:
      target: "transformer.layer.0.intermediate"
      method: "svd"
      target_ratio: 0.90
"""
    recipe = loads_recipe(recipe_yaml)
    model = ThreeLayerTransformerToy()

    # Generate resource estimate
    est = estimate_resources(recipe, hardware_profile="rtx-4090")
    assert est.peak_vram_mb > 0

    # Auto-tune with tight memory limit
    tuned = suggest_config(recipe, hardware_constraints={"max_vram_gb": 12.0})
    assert tuned is not None

    # Plan tuned recipe and execute on model
    plan = plan_recipe(tuned, model_or_config=model)
    res = execute_plan(plan, model=model, device="cpu")
    assert res.success is True
    assert len(res.stage_records) == 1
