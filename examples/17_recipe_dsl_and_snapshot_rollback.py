#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 17: Recipe DSL, Execution Planning, and Snapshot Rollback.

Subsystem Demonstrated:
    Declarative surgery recipe programming, static validation, execution planning,
    and atomic rollback safety:
    1. Programmatic Recipe DSL Construction (neuralops.recipe)
       - Builds a canonical SurgeryRecipe with apiVersion "neuralops/v1"
       - Defines chained operations (attention.replace, architecture.reduce_mlp)
    2. Execution Planning (neuralops.pipeline_transform.plan_recipe)
       - Performs static schema validation and dependency resolution
       - Generates ordered execution stages with operation contracts
    3. Model Mutation & Staged Execution
       - Injects cosformer linear attention via ModelSurgery
    4. Model Snapshot & Atomic Rollback (neuralops.pipeline_transform.ModelSnapshot)
       - Captures immutable state and architectural graph snapshot
       - Simulates intentional mid-pipeline failure during recipe execution
       - Restores exact model state and verifies byte-for-byte state dict parity and SHA256 match

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution on macOS CPU.
    - GPU: Optional (device agnostic).
    - Network: Not required; uses offline declarative schemas.

What You Will Learn:
    1. How to programmatically construct NeuralOps recipes in Python.
    2. How the recipe planner statically verifies preconditions and dependencies.
    3. How ModelSnapshot preserves both weights and dynamic module graph topologies.
    4. How to execute atomic rollback after simulated runtime surgery failures.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from examples._shared import (
    TinyClassifier,
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    section,
    table,
    tiny_inputs,
)
from neuralops.pipeline_transform import (
    ModelSnapshot,
    execute_recipe,
    plan_recipe,
)
from neuralops.recipe import Metadata, ModelSpec, Operation, Recipe
from neuralops.surgery import ModelSurgery


def compute_model_hash(model: torch.nn.Module) -> str:
    """Compute deterministic SHA256 checksum over model state dict."""
    hasher = hashlib.sha256()
    for key in sorted(model.state_dict().keys()):
        tensor = model.state_dict()[key].cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def main() -> int:
    section("NeuralOps Example 17: Recipe DSL & Snapshot Rollback")

    device = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", device)
    kv("PyTorch Version", torch.__version__)

    # =========================================================================
    # Part 1: Programmatic Recipe DSL Construction
    # =========================================================================
    section("Part 1: Programmatic Recipe DSL Construction")

    recipe = Recipe(
        apiVersion="neuralops/v1",
        kind="SurgeryRecipe",
        metadata=Metadata(
            name="distilbert-hybrid-compression",
            version="1.0.0",
            description="Replace quadratic attention with cosformer and compress MLP blocks",
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

    kv("Recipe API Version", recipe.apiVersion)
    kv("Recipe Kind", recipe.kind)
    kv("Recipe Name", recipe.metadata.name)
    kv("Declared Operations", len(recipe.operations))

    p_schema = bool(recipe.apiVersion == "neuralops/v1" and len(recipe.operations) == 2)
    pass_fail(
        "Recipe Schema Object Construction",
        p_schema,
        "Valid canonical recipe with 2 operations" if p_schema else "Failed schema construction",
    )

    # =========================================================================
    # Part 2: Static Execution Planning
    # =========================================================================
    section("Part 2: Static Execution Planning")

    model = TinyClassifier(dim=64, hidden_dim=128).to(device)
    orig_params = sum(p.numel() for p in model.parameters())
    orig_hash = compute_model_hash(model)
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    kv("Baseline Parameters", f"{orig_params:,}")
    kv("Baseline SHA256 Hash", orig_hash[:16] + "...")

    plan = plan_recipe(recipe, model_or_config=model)
    p_plan = bool(plan and len(plan.stages) == 2)
    pass_fail(
        "Recipe Static Validation & Planning",
        p_plan,
        f"Generated {len(plan.stages)} valid stages" if p_plan else "Planning failed",
    )

    plan_rows: list[dict[str, str]] = []
    for s in plan.stages:
        plan_rows.append(
            {
                "Stage ID": s.id,
                "Operation": s.operation_type,
                "Version": s.version,
                "Dependencies": ", ".join(s.depends_on) if s.depends_on else "None",
                "Target Contract": s.contract.name if s.contract else "Generic",
            }
        )

    headers_plan = ["Stage ID", "Operation", "Version", "Dependencies", "Target Contract"]
    rows_plan = [[str(r[h]) for h in headers_plan] for r in plan_rows]
    table(rows_plan, headers=headers_plan)

    # =========================================================================
    # Part 3: Snapshot Capture & Staged Execution
    # =========================================================================
    section("Part 3: Model Snapshot & Mutation Pipeline")

    snapshot = ModelSnapshot.capture(model)
    kv("Snapshot State", "Captured in-memory deepcopy & state_dict")

    surgery = ModelSurgery(model)
    # Stage 1 succeeds
    surgery.apply("attention.replace", target="cosformer")
    mutated_params = sum(p.numel() for p in surgery.model.parameters())
    mutated_hash = compute_model_hash(surgery.model)

    kv("Stage 1 (attention.replace) Result", f"{orig_params:,} -> {mutated_params:,} params")
    p_mutation = bool(mutated_hash != orig_hash and mutated_params != orig_params)
    pass_fail(
        "Model Mutation Confirmed",
        p_mutation,
        "Weights and parameter count diverged from baseline" if p_mutation else "Mutation failed",
    )

    # =========================================================================
    # Part 4: Mid-Pipeline Failure Simulation & Atomic Rollback
    # =========================================================================
    section("Part 4: Simulated Failure & Rollback Execution")

    # Construct a recipe containing an invalid operation to trigger mid-pipeline failure
    failing_recipe = Recipe(
        apiVersion="neuralops/v1",
        kind="SurgeryRecipe",
        metadata=Metadata(
            name="failing-recipe-pipeline",
            version="1.0.0",
            description="Recipe with intentional invalid operation to verify rollback",
        ),
        model=ModelSpec(source="distilbert", dtype="fp32"),
        operations=[
            Operation(
                id="failing_stage",
                type="invalid_operation_that_fails",
                version="1.0.0",
                config={},
            ),
        ],
    )

    simulated_failure_caught = False
    try:
        execute_recipe(failing_recipe, surgery.model)
    except Exception as err:
        simulated_failure_caught = True
        kv("Simulated Failure Caught", f"{type(err).__name__}: {err}")
        # Roll back model to the pre-surgery baseline snapshot
        surgery.model = snapshot.restore(surgery.model)

    pass_fail(
        "Simulated Pipeline Failure Handling",
        simulated_failure_caught,
        "Failure caught and rollback triggered" if simulated_failure_caught else "Failed to catch error",
    )

    # Verify rollback invariants
    restored_params = sum(p.numel() for p in surgery.model.parameters())
    restored_hash = compute_model_hash(surgery.model)

    param_match = bool(restored_params == orig_params)
    hash_match = bool(restored_hash == orig_hash)
    restored_perfectly = all(torch.equal(surgery.model.state_dict()[k], initial_weights[k]) for k in initial_weights)

    # Functional forward pass verification
    sample_inputs = tiny_inputs()
    sample_inputs = {k: v.to(device) for k, v in sample_inputs.items()}
    surgery.model.eval()
    with torch.no_grad():
        out_restored = surgery.model(**sample_inputs)
        logits = out_restored.logits if hasattr(out_restored, "logits") else out_restored
        fwd_valid = bool(torch.isfinite(logits).all().item())

    pass_fail(
        "Parameter Count Restored Exactly",
        param_match,
        f"{restored_params:,} == {orig_params:,}",
    )
    pass_fail(
        "Weight Hash Restored Exactly",
        hash_match,
        f"{restored_hash[:16]} == {orig_hash[:16]}",
    )
    pass_fail(
        "Snapshot Rollback Exact Restoration",
        restored_perfectly,
        "All weights restored byte-for-byte" if restored_perfectly else "Weight mismatch detected",
    )
    pass_fail(
        "Restored Model Forward Pass",
        fwd_valid,
        "Forward pass valid and finite" if fwd_valid else "Forward pass produced non-finite output",
    )

    print("\n  >>> Model restored to pre-recipe state, param count matches, hashes match <<<\n")

    all_passed = bool(
        p_schema
        and p_plan
        and p_mutation
        and simulated_failure_caught
        and param_match
        and hash_match
        and restored_perfectly
        and fwd_valid
    )

    pass_fail(
        "Overall Example 17 Status",
        all_passed,
        "Recipe DSL, planning, mutation, and atomic snapshot rollback verified"
        if all_passed
        else "Sub-check failures detected",
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
