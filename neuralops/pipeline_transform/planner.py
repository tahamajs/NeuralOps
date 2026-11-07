#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recipe Planner resolving declarative recipes into validated Execution Plans.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import torch
import torch.nn as nn

from neuralops.adapters.base import ArchitectureAdapter
from neuralops.adapters.registry import find_adapter, find_adapter_by_family
from neuralops.operations.registry import get_operation
from neuralops.pipeline_transform.errors import PlanningError
from neuralops.pipeline_transform.plan import ExecutionPlan, Stage
from neuralops.recipe.canonical import recipe_hash
from neuralops.recipe.loader import load_recipe
from neuralops.recipe.schema import Recipe
from neuralops.validation.errors import StaticValidationError
from neuralops.validation.static import validate_recipe_static

logger = logging.getLogger("NeuralOps.PipelineTransform.Planner")


def plan_recipe(
    recipe: Union[Recipe, str, Path, Dict[str, Any]],
    model_or_config: Optional[Union[nn.Module, Any]] = None,
) -> ExecutionPlan:
    """Check and compile a recipe into an ExecutionPlan without GPU execution.

    Parameters:
        recipe: Recipe object, YAML/JSON string or path, or raw dict.
        model_or_config: Optional nn.Module or model config object for precondition checks.

    Returns:
        Validated ExecutionPlan ready for transactional execution.

    Raises:
        PlanningError: If adapter is missing or operation preconditions fail.
    """
    cuda_before = torch.cuda.is_initialized()

    # 1. Parse and statically validate recipe
    if isinstance(recipe, (str, Path)):
        recipe_obj = load_recipe(recipe)
    elif isinstance(recipe, dict):
        recipe_obj = Recipe.from_dict(recipe)
    else:
        recipe_obj = recipe

    try:
        validate_recipe_static(recipe_obj)
    except StaticValidationError as e:
        raise PlanningError(str(e), error_code=e.error_code, stage=e.stage) from e

    # 2. Determine target model & architecture adapter
    adapter: Optional[ArchitectureAdapter] = None
    target_name = recipe_obj.model.source or recipe_obj.model.family or "unknown"

    if isinstance(model_or_config, nn.Module):
        adapter = find_adapter(model_or_config)
    elif recipe_obj.model.family:
        adapter = find_adapter_by_family(recipe_obj.model.family)
    elif recipe_obj.model.source:
        # Infer family from source identifier
        source_lower = recipe_obj.model.source.lower()
        if "llama" in source_lower:
            adapter = find_adapter_by_family("llama")
        elif "distilbert" in source_lower:
            adapter = find_adapter_by_family("distilbert")
        elif "gpt2" in source_lower:
            adapter = find_adapter_by_family("gpt2")
        elif "mamba" in source_lower:
            adapter = None  # Explicitly no adapter for Mamba

    if adapter is None:
        family_name = recipe_obj.model.family or (recipe_obj.model.source if recipe_obj.model.source else "unknown")
        raise PlanningError(
            f"No architecture adapter found for model or family '{family_name}'",
            error_code="E_ADAPTER_NOT_FOUND",
        )

    # 3. Compile stages and verify adapter capabilities & preconditions
    stages: List[Stage] = []
    stage_ids: Set[str] = set()
    warnings: List[str] = []
    total_param_delta: int = 0
    declared_caps = adapter.declared_capabilities()

    # Base parameter count estimation if model is available
    base_params = (
        sum(p.numel() for p in model_or_config.parameters()) if isinstance(model_or_config, nn.Module) else 100_000_000
    )

    for idx, op in enumerate(recipe_obj.operations):
        op_id = op.id or f"stage-{idx + 1}"
        stage_ids.add(op_id)

        # Dependency resolution
        for dep in op.depends_on:
            if dep not in stage_ids:
                raise PlanningError(
                    f"Stage '{op_id}' depends on unresolved stage '{dep}'",
                    error_code="E_PLAN_UNRESOLVED_DEPENDENCY",
                    stage=op_id,
                )

        # Operation resolution from registry
        try:
            op_impl = get_operation(op.type, op.version)
        except Exception as e:
            raise PlanningError(
                f"Failed to resolve operation '{op.type}' v{op.version}: {e}",
                error_code="E_OP_NOT_FOUND",
                stage=op_id,
            ) from e

        contract = op_impl.contract

        # Adapter capability check
        # Match op.type or op.type[key=val]
        op_keys = [op.type]
        for k, v in op.config.items():
            op_keys.append(f"{op.type}[{k}={v}]")

        # Check if any matching capability is defined and allowed
        capability_found = False
        for key in op_keys:
            if key in declared_caps:
                capability_found = True
                status = declared_caps[key].lower()
                if "unsupported" in status:
                    raise PlanningError(
                        f"Adapter '{adapter.family}' marks operation '{key}' as unsupported ({declared_caps[key]})",
                        error_code="E_COMP_PRECONDITION_UNSATISFIED",
                        stage=op_id,
                    )
                break

        if not capability_found:
            # Check family compatibility: if adapter has any capability starting with op family
            op_family = op.type.split(".")[0]
            if not any(k.startswith(op_family) for k in declared_caps.keys()):
                raise PlanningError(
                    f"Adapter '{adapter.family}' does not declare capability for operation family '{op.type}'",
                    error_code="E_COMP_PRECONDITION_UNSATISFIED",
                    stage=op_id,
                )

        # Precondition check
        if contract and contract.preconditions:
            for prec in contract.preconditions:
                if isinstance(model_or_config, nn.Module):
                    satisfied, reason = adapter.check_precondition(model_or_config, prec)
                    if not satisfied:
                        raise PlanningError(
                            f"Precondition '{prec}' unsatisfied for operation '{op.type}': {reason}",
                            error_code="E_COMP_PRECONDITION_UNSATISFIED",
                            stage=op_id,
                        )

        # Naive resource delta estimation
        stage_param_delta = 0
        if "compress" in op.type:
            comp_ratio = float(op.config.get("compression_ratio", 0.5))
            stage_param_delta = -int(base_params * (1.0 - comp_ratio) * 0.25)
        elif "drop_layer" in op.type:
            stage_param_delta = -int(base_params * 0.1)

        total_param_delta += stage_param_delta

        stage = Stage(
            id=op_id,
            operation_type=op.type,
            version=op.version,
            config=dict(op.config),
            contract=contract,
            depends_on=list(op.depends_on),
            estimated_parameters_delta=stage_param_delta,
        )
        stages.append(stage)

    # Compute canonical recipe hash
    r_hash = recipe_hash(recipe_obj)

    # Build ExecutionPlan
    plan = ExecutionPlan(
        recipe_name=recipe_obj.metadata.name,
        target_model=target_name,
        stages=stages,
        warnings=warnings,
        estimated_parameters_delta=total_param_delta,
        recipe_hash=r_hash,
        validation_spec=(
            recipe_obj.validation.to_dict()
            if hasattr(recipe_obj.validation, "to_dict")
            else asdict_safe(recipe_obj.validation)
        ),
        execution_spec=(
            recipe_obj.execution.to_dict()
            if hasattr(recipe_obj.execution, "to_dict")
            else asdict_safe(recipe_obj.execution)
        ),
        metadata={
            "recipe_version": recipe_obj.metadata.version,
            "author": recipe_obj.metadata.author,
            "created_at": recipe_obj.metadata.created_at,
        },
    )

    cuda_after = torch.cuda.is_initialized()
    if not cuda_before and cuda_after:
        warnings.append("CUDA context was unexpectedly initialized during planning.")

    logger.info("Compiled recipe '%s' into %d execution stages.", plan.recipe_name, len(plan.stages))
    return plan


def asdict_safe(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {}


__all__ = ["plan_recipe"]
