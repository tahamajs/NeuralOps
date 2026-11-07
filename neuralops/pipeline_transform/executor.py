#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Transactional Pipeline Execution Engine with Per-Stage Validation and Rollback.
"""

from __future__ import annotations

import logging
import platform
import time
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn

from neuralops.operations.registry import get_operation
from neuralops.pipeline_transform.plan import ExecutionPlan
from neuralops.pipeline_transform.result import PipelineError, PipelineResult, StageRecord
from neuralops.pipeline_transform.snapshot import capture, restore
from neuralops.runtime.context import get_event_sink
from neuralops.validation.runner import run_validation_gates

logger = logging.getLogger("NeuralOps.PipelineTransform.Executor")


def _get_memory_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def execute_plan(
    plan: ExecutionPlan,
    model: Optional[nn.Module] = None,
    device: str = "auto",
) -> PipelineResult:
    """Execute an ExecutionPlan transactionally with per-stage validation and rollback.

    Parameters:
        plan: The compiled ExecutionPlan to execute.
        model: Optional in-memory model to mutate. If None, loaded from plan target.
        device: Device to place the model on ("auto", "cpu", "cuda", etc.).

    Returns:
        PipelineResult with final model, telemetry stage records, and evidence.

    Raises:
        PipelineError: If any stage or validation gate fails (model is rolled back).
    """
    t_start = time.perf_counter()

    if model is None:
        raise ValueError(
            f"No model provided to execute_plan. Automatic downloading for '{plan.target_model}' "
            "requires passing an explicit model instance in offline execution mode."
        )

    # Device placement
    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    model.to(target_device)

    env_info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(target_device),
        "cuda_available": torch.cuda.is_available(),
    }

    stage_records: List[StageRecord] = []
    completed_stages: Set[str] = set()
    event_sink = get_event_sink()
    event_sink.emit("pipeline.start", plan_id=getattr(plan, "name", "plan"), total_stages=len(plan.stages))

    for stage_idx, stage in enumerate(plan.stages):
        logger.info("Starting stage %d: '%s' (%s)", stage_idx + 1, stage.id, stage.operation_type)
        event_sink.emit(
            "stage.start",
            stage_idx=stage_idx,
            stage_id=stage.id,
            operation_type=stage.operation_type,
        )

        # 1. Verify dependencies
        for dep in stage.depends_on:
            if dep not in completed_stages:
                raise PipelineError(
                    message=f"Stage '{stage.id}' depends on uncompleted stage '{dep}'",
                    failed_stage_index=stage_idx,
                    failed_operation=stage.operation_type,
                    failed_gate="dependency_check",
                    error_code="E_STAGE_DEPENDENCY_UNMET",
                    suggestions=[
                        f"Ensure prior stage '{dep}' executes successfully before '{stage.id}'.",
                        "Check stage reordering in the recipe.",
                    ],
                )

        # 2. Resolve operation implementation
        op_impl = get_operation(stage.operation_type, stage.version)

        # 3. Capture snapshot for transactional safety
        # Target specific paths if declared or provided in config
        state_paths = None
        if "target" in stage.config:
            state_paths = [str(stage.config["target"])]
        elif op_impl.contract and op_impl.contract.declares_mutations:
            state_paths = list(op_impl.contract.declares_mutations)

        snapshot = capture(model, state_paths=state_paths)
        mem_before = _get_memory_bytes(model)
        params_before = sum(p.numel() for p in model.parameters())

        # 4. Apply operation
        t_stage_start = time.perf_counter()
        try:
            new_model, op_evidence = op_impl.apply(model, stage.config)
        except Exception as exc:
            # Rollback and raise PipelineError
            logger.error("Operation application failed at stage '%s': %s", stage.id, exc)
            restore(model, snapshot)
            raise PipelineError(
                message=f"Operation application failed during stage '{stage.id}': {exc}",
                failed_stage_index=stage_idx,
                failed_operation=stage.operation_type,
                failed_gate="operation_apply",
                error_code="E_OPERATION_APPLY_FAILED",
                suggestions=[
                    f"Check configuration parameter types for '{stage.operation_type}'.",
                    "Verify model structure is compatible with the selected transformation.",
                    "Inspect operation trace for root cause details.",
                ],
            ) from exc

        # 5. Run validation gates (model before vs model after)
        val_profile = (
            plan.validation_spec.get("profile", "structural")
            if isinstance(plan.validation_spec, dict)
            else getattr(plan.validation_spec, "profile", "structural")
        )
        stage_val_ctx = dict(stage.config)
        stage_val_ctx.setdefault("baseline_parameters", params_before)
        val_result = run_validation_gates(
            model_before=model,
            model_after=new_model,
            profile=val_profile,
            context=stage_val_ctx,
        )

        # 6. Verify gate results
        if not val_result.passed:
            logger.warning("Stage '%s' validation FAILED. Initiating transactional rollback.", stage.id)
            restore(model, snapshot)

            failed_gate = next(
                (g.gate_name for g in val_result.gate_results if not g.passed and not g.skipped),
                "unknown_gate",
            )
            raise PipelineError(
                message=f"Validation failed after stage '{stage.id}' on gate '{failed_gate}'",
                failed_stage_index=stage_idx,
                failed_operation=stage.operation_type,
                failed_gate=failed_gate,
                validation_result=val_result,
                suggestions=[
                    f"Review '{failed_gate}' gate requirements and thresholds in recipe validation spec.",
                    "Verify transformation config does not corrupt numerical stability or architecture shapes.",
                    "Check if a less aggressive transformation ratio allows validation to pass.",
                ],
            )

        # 7. Commit stage success
        model = new_model
        mem_after = _get_memory_bytes(model)
        duration = time.perf_counter() - t_stage_start

        record = StageRecord(
            operation_name=stage.operation_type,
            duration=duration,
            memory_before=mem_before,
            memory_after=mem_after,
            validation_result=val_result,
            evidence=op_evidence,
            stage_id=stage.id,
            success=True,
        )
        stage_records.append(record)
        completed_stages.add(stage.id)
        event_sink.emit(
            "stage.success",
            stage_idx=stage_idx,
            stage_id=stage.id,
            duration=duration,
            memory_bytes=mem_after,
        )
        logger.info("Stage '%s' successfully committed (%.3fs)", stage.id, duration)

    total_time = time.perf_counter() - t_start
    event_sink.emit(
        "pipeline.success",
        total_time=total_time,
        total_stages=len(stage_records),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return PipelineResult(
        final_model=model,
        stage_records=stage_records,
        recipe_hash=plan.recipe_hash,
        environment=env_info,
        total_time=total_time,
        success=True,
    )


__all__ = ["execute_plan"]
