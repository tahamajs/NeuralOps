#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Multi-Objective Pareto Frontier Sweeper

Computes non-dominated trade-off frontiers across model accuracy, inference latency,
VRAM footprint, and compression ratio from hyperparameter sweep runs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["ParetoFrontierSweeper"]


class ParetoFrontierSweeper:
    """Computes Pareto optimal trade-off frontiers over benchmark sweep configurations."""

    def __init__(
        self,
        objectives: Sequence[str] = ("accuracy", "latency_ms", "compression_ratio"),
        directions: Sequence[str] = ("max", "min", "max"),
    ) -> None:
        if len(objectives) != len(directions):
            raise ValueError("objectives and directions must have identical lengths.")
        self.objectives = list(objectives)
        self.directions = [d.lower() for d in directions]

    def _dominates(
        self,
        candidate_a: Dict[str, Any],
        candidate_b: Dict[str, Any],
    ) -> bool:
        """Return True if candidate_a Pareto-dominates candidate_b."""
        better_in_any = False
        for obj, direction in zip(self.objectives, self.directions):
            val_a = candidate_a.get(obj)
            val_b = candidate_b.get(obj)
            if val_a is None or val_b is None:
                continue

            if direction == "max":
                if val_a < val_b:
                    return False
                if val_a > val_b:
                    better_in_any = True
            elif direction == "min":
                if val_a > val_b:
                    return False
                if val_a < val_b:
                    better_in_any = True
        return better_in_any

    def compute_frontier(
        self,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract the non-dominated Pareto frontier from candidate benchmark records.

        Args:
            candidates: List of dictionary records containing evaluation metrics.

        Returns:
            Subset of candidate dictionaries belonging to the Pareto frontier.
        """
        if not candidates:
            return []

        frontier: list[dict[str, Any]] = []

        for candidate in candidates:
            dominated = False
            for other in candidates:
                if candidate is other:
                    continue
                if self._dominates(other, candidate):
                    dominated = True
                    break
            if not dominated:
                frontier.append(candidate)

        return frontier
