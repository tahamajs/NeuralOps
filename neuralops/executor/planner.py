#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recipe Planner resolving declarative recipes into staged Execution Plans.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import torch.nn as nn

from neuralops.executor.plan import ExecutionPlan, Stage
from neuralops.operations.registry import get_operation
from neuralops.recipe.schema import Recipe
from neuralops.validation.errors import StaticValidationError
from neuralops.validation.static import validate_recipe_static

logger = logging.getLogger("NeuralOps.Executor.Planner")


def plan_recipe(
    recipe: Recipe,
    model: Optional[nn.Module] = None,
) -> ExecutionPlan:
    """Validate recipe and compile it into an ExecutionPlan."""
    # 1. Static validation
    validate_recipe_static(recipe)

    # 2. Compile stages
    stages: List[Stage] = []
    stage_ids: Set[str] = set()

    for op in recipe.operations:
        op_id = op.id or f"stage-{len(stages) + 1}"
        stage_ids.add(op_id)

        # Check dependencies exist
        for dep in op.depends_on:
            if dep not in stage_ids:
                raise StaticValidationError(
                    f"Stage '{op_id}' depends on undefined prior stage '{dep}'",
                    error_code="E_PLAN_UNRESOLVED_DEPENDENCY",
                    stage=op_id,
                )

        op_impl = get_operation(op.type, op.version)
        stage = Stage(
            id=op_id,
            operation_type=op.type,
            version=op.version,
            config=dict(op.config),
            contract=op_impl.contract,
            depends_on=list(op.depends_on),
        )
        stages.append(stage)

    plan = ExecutionPlan(
        recipe_name=recipe.metadata.name,
        target_model=recipe.model.source or recipe.model.family or "unknown",
        stages=stages,
        validation_spec=recipe.validation,
        execution_spec=recipe.execution,
        metadata={
            "recipe_version": recipe.metadata.version,
            "author": recipe.metadata.author,
            "created_at": recipe.metadata.created_at,
        },
    )

    logger.info("Compiled recipe '%s' into %d execution stages.", plan.recipe_name, len(plan.stages))
    return plan


__all__ = ["plan_recipe"]
