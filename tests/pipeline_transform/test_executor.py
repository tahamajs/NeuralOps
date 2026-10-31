#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Sub-Agent B4: Executor Agent (Transactional Execution, Per-Stage Validation & Rollback).
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
import torch.nn as nn

from neuralops.pipeline_transform.executor import execute_plan
from neuralops.pipeline_transform.plan import ExecutionPlan
from neuralops.pipeline_transform.planner import plan_recipe
from neuralops.pipeline_transform.result import PipelineError, PipelineResult
from neuralops.recipe.schema import (
    Metadata,
    ModelSpec,
    Operation,
    Recipe,
)


def _hash_state_dict(sd: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode("utf-8"))
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


class TwoLayerTestNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 64, "num_hidden_layers": 2})()
        self.intermediate = nn.ModuleDict({
            "dense1": nn.Linear(64, 64),
            "dense2": nn.Linear(64, 64),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.intermediate["dense1"](x)
        return self.intermediate["dense2"](h)


def test_b4_successful_two_stage_recipe():
    """Verify successful 2-stage recipe produces PipelineResult with 2 StageRecords."""
    recipe = Recipe(
        apiVersion="neuralops/v1",
        kind="Recipe",
        metadata=Metadata(name="two-stage-test", version="1.0.0", description="Two stage test"),
        model=ModelSpec(family="distilbert", source="distilbert-base-uncased", dtype="float32"),
        operations=[
            Operation(
                id="stage-1",
                type="weights.compress",
                version="1.0.0",
                config={"target": "intermediate.dense1", "method": "svd", "target_ratio": 0.8},
            ),
            Operation(
                id="stage-2",
                type="weights.compress",
                version="1.0.0",
                config={"target": "intermediate.dense2", "method": "svd", "target_ratio": 0.8},
                depends_on=["stage-1"],
            ),
        ],
    )
    model = TwoLayerTestNet()
    plan = plan_recipe(recipe, model_or_config=model)
    assert len(plan) == 2

    result = execute_plan(plan, model=model, device="cpu")
    assert isinstance(result, PipelineResult)
    assert result.success is True
    assert len(result.stage_records) == 2
    assert result.stage_records[0].stage_id == "stage-1"
    assert result.stage_records[1].stage_id == "stage-2"
    assert result.stage_records[0].success is True
    assert result.stage_records[1].success is True


def test_b4_failing_stage_two_triggers_rollback_to_stage_one():
    """Verify failing stage 2 triggers rollback so model is bit-identical to model after stage 1."""
    recipe_stage1 = Recipe(
        apiVersion="neuralops/v1",
        kind="Recipe",
        metadata=Metadata(name="stage1-only", version="1.0.0", description="Stage 1 only"),
        model=ModelSpec(family="distilbert", source="distilbert-base-uncased", dtype="float32"),
        operations=[
            Operation(
                id="stage-1",
                type="weights.compress",
                version="1.0.0",
                config={"target": "intermediate.dense1", "method": "svd", "target_ratio": 0.8},
            ),
        ],
    )

    model = TwoLayerTestNet()
    plan1 = plan_recipe(recipe_stage1, model_or_config=model)
    res1 = execute_plan(plan1, model=model, device="cpu")
    hash_after_stage1 = _hash_state_dict(model.state_dict())

    # Now execute a plan with failing stage 2 on the SAME model
    recipe_failing = Recipe(
        apiVersion="neuralops/v1",
        kind="Recipe",
        metadata=Metadata(name="failing-stage2", version="1.0.0", description="Failing stage 2"),
        model=ModelSpec(family="distilbert", source="distilbert-base-uncased", dtype="float32"),
        operations=[
            Operation(
                id="stage-2",
                type="weights.compress",
                version="1.0.0",
                # target_ratio 0.1 on all (*) will fail ParameterCountGate because compressing one layer cannot achieve 0.1 ratio
                config={"target": "*", "method": "svd", "target_ratio": 0.1},
            ),
        ],
    )

    plan2 = plan_recipe(recipe_failing, model_or_config=model)

    with pytest.raises(PipelineError) as exc_info:
        execute_plan(plan2, model=model, device="cpu")

    err = exc_info.value
    assert err.failed_stage_index == 0
    assert err.failed_operation == "weights.compress"
    assert len(err.suggestions) >= 3

    # Error message format check
    err_str = str(err)
    assert "Transformation failed at stage 0" in err_str
    assert "Actionable Suggestions:" in err_str

    # Bit-identical state restoration check: model after failed stage 2 MUST match model after stage 1
    hash_after_rollback = _hash_state_dict(model.state_dict())
    assert hash_after_rollback == hash_after_stage1


def test_b4_pipeline_error_message_and_suggestions():
    """Verify PipelineError contains stage, gate, and at least 3 suggestions."""
    err = PipelineError(
        message="Gate shapes_match failed",
        failed_stage_index=0,
        failed_operation="weights.compress",
        failed_gate="shapes_match",
        suggestions=[
            "Check layer dimensions.",
            "Inspect forward graph.",
            "Verify tensor shapes.",
        ],
    )
    msg = str(err)
    assert "stage 0" in msg
    assert "weights.compress" in msg
    assert "shapes_match" in msg
    assert "1. Check layer dimensions." in msg
    assert "2. Inspect forward graph." in msg
    assert "3. Verify tensor shapes." in msg


def test_b4_pipeline_result_json_roundtrip():
    """Verify PipelineResult.to_dict() is JSON-serializable."""
    recipe = Recipe(
        apiVersion="neuralops/v1",
        kind="Recipe",
        metadata=Metadata(name="json-check", version="1.0.0", description="Json check"),
        model=ModelSpec(family="distilbert", source="distilbert-base-uncased", dtype="float32"),
        operations=[
            Operation(
                id="s1",
                type="weights.compress",
                version="1.0.0",
                config={"target": "intermediate.dense1", "method": "svd", "target_ratio": 0.8},
            ),
        ],
    )
    model = TwoLayerTestNet()
    plan = plan_recipe(recipe, model_or_config=model)
    result = execute_plan(plan, model=model, device="cpu")

    res_dict = result.to_dict()
    json_str = json.dumps(res_dict)
    loaded = json.loads(json_str)
    assert loaded["success"] is True
    assert len(loaded["stage_records"]) == 1
