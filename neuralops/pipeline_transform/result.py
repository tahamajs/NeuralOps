#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Transformation Pipeline Results and Diagnostic Error Models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch.nn as nn

from neuralops.core.errors import NeuralOpsError
from neuralops.validation.result import ValidationResult


class PipelineError(NeuralOpsError):
    """Raised when pipeline transformation fails, providing diagnostic rollback context."""

    def __init__(
        self,
        message: str,
        failed_stage_index: int,
        failed_operation: str,
        failed_gate: Optional[str] = None,
        validation_result: Optional[ValidationResult] = None,
        suggestions: Optional[List[str]] = None,
        error_code: str = "E_TRANSFORMATION_STAGE_FAILED",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.failed_stage_index = failed_stage_index
        self.failed_operation = failed_operation
        self.failed_gate = failed_gate or "unknown"
        self.validation_result = validation_result
        self.suggestions = suggestions or [
            "Verify operation configuration parameters in the recipe.",
            "Inspect model architecture compatibility and layer dimensions.",
            "Lower compression target ratio or switch to a less aggressive rank.",
        ]
        self.error_code = error_code

    def __str__(self) -> str:
        lines = [
            f"[{self.error_code}] Transformation failed at stage {self.failed_stage_index} ('{self.failed_operation}'):",
            f"  Failed Gate : {self.failed_gate}",
            f"  Message     : {self.message}",
        ]
        if self.validation_result and self.validation_result.gate_results:
            failed_gates = [g for g in self.validation_result.gate_results if not g.passed and not g.skipped]
            if failed_gates:
                lines.append("  Gate Details:")
                for fg in failed_gates:
                    lines.append(f"    - {fg.gate_name}: {fg.message}")
        lines.append("  Actionable Suggestions:")
        for idx, s in enumerate(self.suggestions, start=1):
            lines.append(f"    {idx}. {s}")
        return "\n".join(lines)


@dataclass
class StageRecord:
    """Detailed telemetry record for an executed transformation stage."""

    operation_name: str
    duration: float
    memory_before: int
    memory_after: int
    validation_result: Optional[ValidationResult] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    stage_id: str = ""
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "operation_name": self.operation_name,
            "success": self.success,
            "duration": self.duration,
            "memory_before": self.memory_before,
            "memory_after": self.memory_after,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "evidence": dict(self.evidence),
        }


@dataclass
class PipelineResult:
    """Outcome of transactional transformation plan execution."""

    final_model: nn.Module
    stage_records: List[StageRecord] = field(default_factory=list)
    recipe_hash: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)
    total_time: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "total_time": self.total_time,
            "recipe_hash": self.recipe_hash,
            "stage_records": [s.to_dict() for s in self.stage_records],
            "environment": dict(self.environment),
            "error": self.error,
        }


__all__ = ["PipelineError", "PipelineResult", "StageRecord"]
