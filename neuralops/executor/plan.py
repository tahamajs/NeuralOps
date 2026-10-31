#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Execution Plan and Stage Data Models for Recipe Execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from neuralops.operations.base import OperationContract
from neuralops.recipe.schema import ExecutionSpec, ValidationSpec


@dataclass
class Stage:
    """An executable transformation step within an execution plan."""

    id: str
    operation_type: str
    version: str
    config: Dict[str, Any]
    contract: OperationContract
    depends_on: List[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """A fully-validated, staged transformation plan."""

    recipe_name: str
    target_model: str
    stages: List[Stage]
    validation_spec: ValidationSpec = field(default_factory=lambda: ValidationSpec(profile="standard"))
    execution_spec: ExecutionSpec = field(default_factory=ExecutionSpec)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.stages)


__all__ = ["ExecutionPlan", "Stage"]
