#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recipe Execution Engine with Transactional Rollback and Per-Stage Validation.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional

import torch.nn as nn

from neuralops.executor.plan import ExecutionPlan, Stage
from neuralops.executor.planner import plan_recipe
from neuralops.executor.result import PipelineResult, StageRecord
from neuralops.executor.snapshot import ModelSnapshot
from neuralops.operations.registry import get_operation
from neuralops.recipe.schema import Recipe
from neuralops.validation.runner import run_validation_gates

logger = logging.getLogger("NeuralOps.Executor.Engine")


def execute_plan(
    plan: ExecutionPlan,
    model: nn.Module,
    baseline_model: Optional[nn.Module] = None,
    enable_rollback: Optional[bool] = None,
) -> PipelineResult:
    """Execute all stages in an ExecutionPlan transactionally."""
    t_start = time.perf_counter()

    should_rollback = (
        enable_rollback
        if enable_rollback is not None
        else (plan.execution_spec.rollback == "stage" or plan.execution_spec.rollback == "full" or plan.execution_spec.rollback is True)
    )

    # Initial baseline reference
    if baseline_model is None:
        try:
            baseline_model = copy.deepcopy(model).cpu()
        except Exception:
            baseline_model = None

    initial_snapshot = ModelSnapshot.capture(model) if should_rollback else None
    current_model = model
    stage_records: List[StageRecord] = []

    for stage in plan.stages:
        t_stage = time.perf_counter()
        stage_snapshot = ModelSnapshot.capture(current_model) if should_rollback else None

        logger.info("Executing stage '%s' (%s@%s)...", stage.id, stage.operation_type, stage.version)
        try:
            op_impl = get_operation(stage.operation_type, stage.version)
            current_model, evidence = op_impl.apply(current_model, stage.config)
        except Exception as e:
            logger.error("Stage '%s' execution raised exception: %s", stage.id, e)
            if should_rollback and stage_snapshot is not None:
                logger.warning("Rolling back model after stage '%s' failure...", stage.id)
                current_model = stage_snapshot.restore(current_model)

            rec = StageRecord(
                stage_id=stage.id,
                operation_type=stage.operation_type,
                success=False,
                error=str(e),
                duration_seconds=time.perf_counter() - t_stage,
            )
            stage_records.append(rec)
            return PipelineResult(
                success=False,
                model=current_model,
                stage_records=stage_records,
                duration_seconds=time.perf_counter() - t_start,
                rolled_back=should_rollback,
                error=f"Stage '{stage.id}' failed: {e}",
            )

        # Per-stage validation if configured
        val_res = None
        profile = plan.validation_spec.profile
        if profile != "none":
            val_res = run_validation_gates(
                model=current_model,
                baseline_model=baseline_model,
                profile=profile,
                context=stage.config,
            )
            if not val_res.passed:
                logger.warning("Stage '%s' failed validation profile '%s'", stage.id, profile)
                if should_rollback and stage_snapshot is not None:
                    logger.warning("Rolling back stage '%s' due to validation failure...", stage.id)
                    current_model = stage_snapshot.restore(current_model)

                rec = StageRecord(
                    stage_id=stage.id,
                    operation_type=stage.operation_type,
                    success=False,
                    evidence=evidence,
                    validation=val_res,
                    error=f"Validation failed with {val_res.failed_gates} gate error(s)",
                    duration_seconds=time.perf_counter() - t_stage,
                )
                stage_records.append(rec)
                return PipelineResult(
                    success=False,
                    model=current_model,
                    stage_records=stage_records,
                    validation=val_res,
                    duration_seconds=time.perf_counter() - t_start,
                    rolled_back=should_rollback,
                    error=f"Stage '{stage.id}' failed validation",
                )

        rec = StageRecord(
            stage_id=stage.id,
            operation_type=stage.operation_type,
            success=True,
            evidence=evidence,
            validation=val_res,
            duration_seconds=time.perf_counter() - t_stage,
        )
        stage_records.append(rec)

    # Final overall validation
    final_val = None
    profile = plan.validation_spec.profile
    if profile != "none":
        final_val = run_validation_gates(
            model=current_model,
            baseline_model=baseline_model,
            profile=profile,
        )
        if not final_val.passed and should_rollback and initial_snapshot is not None:
            logger.warning("Final validation failed; rolling back entire transformation...")
            current_model = initial_snapshot.restore(current_model)
            return PipelineResult(
                success=False,
                model=current_model,
                stage_records=stage_records,
                validation=final_val,
                duration_seconds=time.perf_counter() - t_start,
                rolled_back=True,
                error="Final pipeline validation failed",
            )

    return PipelineResult(
        success=True,
        model=current_model,
        stage_records=stage_records,
        validation=final_val,
        duration_seconds=time.perf_counter() - t_start,
        rolled_back=False,
    )


def execute_recipe(
    recipe: Recipe,
    model: nn.Module,
    baseline_model: Optional[nn.Module] = None,
    enable_rollback: Optional[bool] = None,
) -> PipelineResult:
    """Compile and execute a Recipe against a model."""
    plan = plan_recipe(recipe, model=model)
    return execute_plan(
        plan=plan,
        model=model,
        baseline_model=baseline_model,
        enable_rollback=enable_rollback,
    )


__all__ = ["execute_plan", "execute_recipe"]
