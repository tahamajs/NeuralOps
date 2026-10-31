#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Sub-Agent B1: Planner Agent (Execution Planning & Precondition Checking).
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from neuralops.pipeline_transform.errors import PlanningError
from neuralops.pipeline_transform.plan import ExecutionPlan
from neuralops.pipeline_transform.planner import plan_recipe
from neuralops.recipe.schema import (
    Metadata,
    ModelSpec,
    Operation,
    Recipe,
)


class DummyLlamaModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 768, "num_hidden_layers": 12})()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.Linear(768, 768),
                "mlp": nn.Linear(768, 3072),
            })
            for _ in range(2)
        ])


def test_b1_valid_llama_recipe_planning():
    """Verify plan_recipe on a valid LLaMA recipe returns an ExecutionPlan with correct stages."""
    recipe = Recipe(
        apiVersion="neuralops/v1",
        kind="Recipe",
        metadata=Metadata(name="llama-compression", version="1.0.0", description="Llama test"),
        model=ModelSpec(family="llama", source="meta-llama/Llama-3-8B", dtype="float16"),
        operations=[
            Operation(
                id="compress-mlp",
                type="weights.compress",
                version="1.0.0",
                config={"target": "*.mlp.*", "method": "aasvd", "target_ratio": 0.5},
            ),
        ],
    )
    dummy_model = DummyLlamaModel()
    plan = plan_recipe(recipe, model_or_config=dummy_model)

    assert isinstance(plan, ExecutionPlan)
    assert len(plan) == 1
    assert plan.stages[0].id == "compress-mlp"
    assert plan.stages[0].operation_type == "weights.compress"
    assert plan.recipe_name == "llama-compression"
    assert plan.recipe_hash.startswith("sha256:")
    assert plan.estimated_parameters_delta < 0


def test_b1_incompatible_capability_raises_precondition_unsatisfied():
    """Verify plan_recipe on incompatible capability returns PlanningError with E_COMP_PRECONDITION_UNSATISFIED."""
    # DistilBERT marks weights.quantize[method=fp8] as unsupported_on_cpu
    recipe = Recipe(
        apiVersion="neuralops/v1",
        kind="Recipe",
        metadata=Metadata(name="distilbert-quantize-unsupported", version="1.0.0", description="Quant test"),
        model=ModelSpec(family="distilbert", source="distilbert-base-uncased", dtype="float32"),
        operations=[
            Operation(
                id="quant-fp8",
                type="weights.quantize",
                version="1.0.0",
                config={"bits": 8, "method": "fp8"},
            ),
        ],
    )
    with pytest.raises(PlanningError) as exc_info:
        plan_recipe(recipe)

    assert exc_info.value.error_code == "E_COMP_PRECONDITION_UNSATISFIED"


def test_b1_mamba_model_raises_no_adapter():
    """Verify plan_recipe on a recipe targeting a Mamba model returns PlanningError (no adapter)."""
    recipe = Recipe(
        apiVersion="neuralops/v1",
        kind="Recipe",
        metadata=Metadata(name="mamba-recipe", version="1.0.0", description="Mamba test"),
        model=ModelSpec(family="mamba", source="state-spaces/mamba-130m", dtype="float32"),
        operations=[
            Operation(
                id="compress-mamba",
                type="weights.compress",
                version="1.0.0",
                config={"method": "svd", "target_ratio": 0.5},
            ),
        ],
    )
    with pytest.raises(PlanningError) as exc_info:
        plan_recipe(recipe)

    assert exc_info.value.error_code == "E_ADAPTER_NOT_FOUND"


def test_b1_execution_plan_json_roundtrip():
    """Verify ExecutionPlan.to_dict() round-trips cleanly through json.dumps/loads."""
    recipe = Recipe(
        apiVersion="neuralops/v1",
        kind="Recipe",
        metadata=Metadata(name="llama-test", version="1.0.0", description="Roundtrip test"),
        model=ModelSpec(family="llama", source="meta-llama/Llama-3-8B", dtype="float16"),
        operations=[
            Operation(
                id="s1",
                type="weights.compress",
                version="1.0.0",
                config={"target": "layer.*", "method": "aasvd", "target_ratio": 0.5},
            ),
        ],
    )
    plan = plan_recipe(recipe)
    plan_dict = plan.to_dict()

    # Must be JSON serializable
    json_str = json.dumps(plan_dict)
    loaded_dict = json.loads(json_str)

    restored_plan = ExecutionPlan.from_dict(loaded_dict)
    assert restored_plan.recipe_name == plan.recipe_name
    assert len(restored_plan) == len(plan)
    assert restored_plan.stages[0].id == plan.stages[0].id
    assert restored_plan.recipe_hash == plan.recipe_hash


def test_b1_zero_accelerator_initialization():
    """Verify planning creates zero CUDA context."""
    is_init_before = torch.cuda.is_initialized()
    recipe = Recipe(
        apiVersion="neuralops/v1",
        kind="Recipe",
        metadata=Metadata(name="device-init-check", version="1.0.0", description="Device check"),
        model=ModelSpec(family="llama", source="meta-llama/Llama-3-8B", dtype="float16"),
        operations=[
            Operation(
                id="s1",
                type="weights.compress",
                version="1.0.0",
                config={"target": "*", "method": "aasvd", "target_ratio": 0.5},
            ),
        ],
    )
    _ = plan_recipe(recipe)
    is_init_after = torch.cuda.is_initialized()
    assert is_init_before == is_init_after
