#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware-Constrained Automatic Recipe Configuration and Optimization.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Union

from neuralops.pipeline_transform.errors import ConfigurationError
from neuralops.pipeline_transform.hardware_planner import HardwareProfile, estimate_resources
from neuralops.recipe.loader import load_recipe
from neuralops.recipe.schema import Recipe


def suggest_config(
    recipe: Union[str, Recipe, Dict[str, Any]],
    hardware_constraints: Dict[str, Any],
    hardware_profile: Optional[Union[str, HardwareProfile, Dict[str, Any]]] = None,
) -> Recipe:
    """Optimize operation parameters to satisfy physical hardware constraints.

    Constraints:
    - max_vram_gb: float (maximum allowable peak VRAM in gigabytes)
    - max_time_seconds: float (maximum allowable transformation wall-clock duration)
    - max_parameters: int (maximum parameter count budget)

    Returns:
        Modified Recipe instance with updated parameters and tuning rationale.

    Raises:
        ConfigurationError: if no feasible configuration satisfies the constraints.
    """
    if isinstance(recipe, str):
        recipe_obj = load_recipe(recipe)
    elif isinstance(recipe, dict):
        recipe_obj = Recipe.from_dict(recipe)
    else:
        recipe_obj = copy.deepcopy(recipe)

    max_vram_gb = hardware_constraints.get("max_vram_gb")
    max_time_sec = hardware_constraints.get("max_time_seconds")
    max_params = hardware_constraints.get("max_parameters")

    # If constraint is impossible from the start (e.g. <= 0 or tiny fraction of a MB)
    if max_vram_gb is not None and max_vram_gb < 0.1:
        raise ConfigurationError(
            f"VRAM constraint {max_vram_gb} GB is infeasible for model transformation.",
            constraints=hardware_constraints,
        )

    # 1. Initial estimate
    initial_estimate = estimate_resources(recipe_obj, hardware_profile=hardware_profile)
    max_vram_mb = (max_vram_gb * 1024.0) if max_vram_gb is not None else float("inf")

    # If initial config already satisfies constraints
    if initial_estimate.peak_vram_mb <= max_vram_mb:
        if max_time_sec is None or initial_estimate.total_time_seconds <= max_time_sec:
            if max_params is None or initial_estimate.parameters_after <= max_params:
                recipe_obj.metadata.tags.append("auto-configured:constraints-satisfied")
                return recipe_obj

    # 2. Search for feasible configuration by adjusting compression / rank
    candidate_ratios = [0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
    rationale: Dict[str, Any] = {}
    found_feasible = False

    for op in recipe_obj.operations:
        if op.type == "weights.compress":
            orig_ratio = float(op.config.get("target_ratio", 0.5))

            for cand_ratio in candidate_ratios:
                if cand_ratio >= orig_ratio and initial_estimate.peak_vram_mb > max_vram_mb:
                    continue

                op.config["target_ratio"] = cand_ratio
                test_est = estimate_resources(recipe_obj, hardware_profile=hardware_profile)

                if test_est.peak_vram_mb <= max_vram_mb:
                    if max_time_sec is None or test_est.total_time_seconds <= max_time_sec:
                        if max_params is None or test_est.parameters_after <= max_params:
                            found_feasible = True
                            rationale[op.id or "weights.compress"] = {
                                "original_target_ratio": orig_ratio,
                                "suggested_target_ratio": cand_ratio,
                                "estimated_peak_vram_mb": test_est.peak_vram_mb,
                                "estimated_time_seconds": test_est.total_time_seconds,
                                "reason": f"Reduced target_ratio from {orig_ratio} to {cand_ratio} to satisfy max_vram_gb={max_vram_gb}",
                            }
                            break

            if found_feasible:
                break

    if not found_feasible:
        # Re-check if any configuration met constraints
        test_est = estimate_resources(recipe_obj, hardware_profile=hardware_profile)
        if test_est.peak_vram_mb > max_vram_mb or (max_time_sec and test_est.total_time_seconds > max_time_sec):
            raise ConfigurationError(
                f"No parameter configuration could satisfy constraints: peak VRAM {test_est.peak_vram_mb:.1f} MB > limit {max_vram_mb:.1f} MB",
                constraints=hardware_constraints,
            )

    recipe_obj.metadata.tags.append("auto-configured:optimized")
    return recipe_obj


__all__ = ["suggest_config"]
