#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Static Recipe Validator implementing Rules W1 through W15 per NRL v1.0.0 §5.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from neuralops.operations.registry import get_operation
from neuralops.recipe.schema import Recipe
from neuralops.validator.errors import (
    E_W2_MISSING_REQUIRED_FIELD,
    E_W3_EMPTY_OPERATIONS,
    E_W4_INCOMPATIBLE_VERSION,
    E_W5_MISSING_TARGET_MODEL,
    E_W6_INVALID_TARGET_PATTERN,
    E_W7_INVALID_OPERATION_PARAMS,
    E_W10_TYPE_CHAIN_MISMATCH,
    E_W12_COMPOSABILITY_VIOLATION,
    E_W15_DUPLICATE_OPERATION_ID,
    StaticValidationError,
)

logger = logging.getLogger("NeuralOps.Validator.Static")


def validate_recipe_static(recipe: Recipe) -> List[str]:
    """Validate recipe against NRL v1.0.0 rules W1 through W15.

    Returns:
        List of warning messages (if any non-fatal warnings exist).

    Raises:
        StaticValidationError on any rule violation.
    """
    warnings: List[str] = []

    # W1 / W2: Required top-level fields
    if not recipe.apiVersion:
        raise StaticValidationError("Recipe is missing apiVersion header", E_W4_INCOMPATIBLE_VERSION)
    if not recipe.metadata or not recipe.metadata.name:
        raise StaticValidationError("Recipe is missing metadata.name", E_W2_MISSING_REQUIRED_FIELD)
    if not recipe.model:
        raise StaticValidationError("Recipe is missing model specification", E_W5_MISSING_TARGET_MODEL)

    # W4: Version compatibility
    if recipe.apiVersion != "neuralops/v1":
        raise StaticValidationError(
            f"Unsupported recipe language version: '{recipe.apiVersion}'. Must be 'neuralops/v1'.",
            E_W4_INCOMPATIBLE_VERSION,
        )

    # W5: Target model
    if not recipe.model.family and not recipe.model.source:
        raise StaticValidationError(
            "Recipe model specification must define family or source",
            E_W5_MISSING_TARGET_MODEL,
        )

    # W3: Operations list non-empty
    if not recipe.operations:
        raise StaticValidationError(
            "Recipe operations list must contain at least one operation",
            E_W3_EMPTY_OPERATIONS,
        )

    # W15: Unique operation IDs
    seen_ids: Set[str] = set()
    for op in recipe.operations:
        if op.id in seen_ids:
            raise StaticValidationError(
                f"Duplicate operation ID '{op.id}' detected in recipe",
                E_W15_DUPLICATE_OPERATION_ID,
            )
        seen_ids.add(op.id)

    # Type chain tracking (W10) and composability tracking (W12)
    current_state: str = "base"
    seen_op_names: Dict[str, int] = {}

    for op in recipe.operations:
        # Resolve operation contract
        op_impl = get_operation(op.type, op.version)
        contract = op_impl.contract

        # W6: Check target pattern
        target_param = op.config.get("target")
        if target_param is not None and not str(target_param).strip():
            raise StaticValidationError(
                f"Operation '{op.id}' has empty target pattern",
                E_W6_INVALID_TARGET_PATTERN,
                stage=op.id,
            )

        # W7: Check required parameters from schema
        schema = contract.params_schema or {}
        required_fields = schema.get("required", [])
        for req in required_fields:
            if req not in op.config:
                raise StaticValidationError(
                    f"Operation '{op.id}' ({op.type}@{op.version}) missing required parameter '{req}'",
                    E_W7_INVALID_OPERATION_PARAMS,
                    stage=op.id,
                )

        # Check additionalProperties
        if schema.get("additionalProperties") is False:
            allowed = set(schema.get("properties", {}).keys())
            for param_key in op.config:
                if param_key not in allowed:
                    raise StaticValidationError(
                        f"Operation '{op.id}' received unrecognized parameter '{param_key}'",
                        E_W7_INVALID_OPERATION_PARAMS,
                        stage=op.id,
                    )

        # W10: Type chain consistency
        expected_in = contract.input_state
        if expected_in != "*" and current_state != expected_in:
            raise StaticValidationError(
                f"Operation '{op.id}' ({op.type}) expects input state '{expected_in}' "
                f"but current model state is '{current_state}'",
                E_W10_TYPE_CHAIN_MISMATCH,
                stage=op.id,
            )
        if contract.output_state != "*":
            current_state = contract.output_state

        # W12: Composability check
        seen_op_names[op.type] = seen_op_names.get(op.type, 0) + 1
        if seen_op_names[op.type] > 1 and not contract.composable_with_self:
            raise StaticValidationError(
                f"Operation '{op.type}' cannot be composed multiple times in the same recipe",
                E_W12_COMPOSABILITY_VIOLATION,
                stage=op.id,
            )

    return warnings


__all__ = ["validate_recipe_static"]
