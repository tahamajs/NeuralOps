#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess and functional tests for examples/17_recipe_dsl_and_snapshot_rollback.py.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import torch

from examples._shared import TinyClassifier
from neuralops.pipeline_transform import ModelSnapshot, plan_recipe
from neuralops.recipe import Metadata, ModelSpec, Operation, Recipe
from neuralops.surgery import ModelSurgery


def test_17_runs_clean():
    """Verify example 17 executes cleanly and returns exit code 0."""
    proc = subprocess.run(
        [sys.executable, "examples/17_recipe_dsl_and_snapshot_rollback.py"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 17 failed with stderr: {proc.stderr}\nstdout: {proc.stdout}"
    assert "[PASS] Overall Example 17 Status" in proc.stdout
    assert "All weights restored byte-for-byte" in proc.stdout


def test_17_no_hardcoded_pass():
    """Verify that example 17 has no hardcoded pass_fail(..., True, ...)."""
    src = Path("examples/17_recipe_dsl_and_snapshot_rollback.py").read_text(encoding="utf-8")
    assert 'pass_fail("Overall Example 17 Status", True' not in src
    assert 'pass_fail("Recipe Schema Object Construction", True' not in src
    assert "all_passed = " in src


def test_17_recipe_dsl_and_planning():
    """Verify Recipe construction and static planning generate stages accurately."""
    recipe = Recipe(
        apiVersion="neuralops/v1",
        kind="SurgeryRecipe",
        metadata=Metadata(
            name="test-recipe",
            version="1.0.0",
            description="Test recipe description",
        ),
        model=ModelSpec(source="distilbert", dtype="fp32"),
        operations=[
            Operation(
                id="stage_attention",
                type="attention.replace",
                version="1.0.0",
                config={"target": "cosformer"},
            ),
            Operation(
                id="stage_mlp",
                type="architecture.reduce_mlp",
                version="1.0.0",
                config={"ratio": 0.8},
                depends_on=["stage_attention"],
            ),
        ],
    )
    model = TinyClassifier(dim=64, hidden_dim=128)
    plan = plan_recipe(recipe, model_or_config=model)
    assert plan is not None
    assert len(plan.stages) == 2
    assert plan.stages[0].id == "stage_attention"
    assert plan.stages[1].id == "stage_mlp"
    assert plan.stages[1].depends_on == ["stage_attention"]


def test_17_rollback_restores_exact_weights():
    """Verify that ModelSnapshot restores exact byte-for-byte weights after surgery mutation."""
    model = TinyClassifier(dim=64, hidden_dim=128)
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    snapshot = ModelSnapshot.capture(model)

    surgery = ModelSurgery(model)
    surgery.apply("attention.replace", target="cosformer")

    # Verify mutation occurred
    mutated_weights = surgery.model.state_dict()
    assert any(
        k not in mutated_weights or not torch.equal(mutated_weights[k], initial_weights[k]) for k in initial_weights
    )

    # Roll back
    restored = snapshot.restore(surgery.model)

    # Verify exact byte-for-byte restoration
    assert set(restored.state_dict().keys()) == set(initial_weights.keys())
    for k, v in initial_weights.items():
        assert torch.equal(restored.state_dict()[k], v), f"Weight mismatch for parameter {k} after snapshot restore"
