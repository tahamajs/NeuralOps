#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Execution Result and Stage Recording Data Models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch.nn as nn

from neuralops.validation.result import ValidationResult


@dataclass
class StageRecord:
    """Record of a single executed transformation stage."""

    stage_id: str
    operation_type: str
    success: bool
    evidence: Dict[str, Any] = field(default_factory=dict)
    validation: Optional[ValidationResult] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "operation_type": self.operation_type,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "evidence": self.evidence,
            "validation": self.validation.to_dict() if self.validation else None,
        }


@dataclass
class PipelineResult:
    """Outcome of a recipe execution pipeline."""

    success: bool
    model: nn.Module
    stage_records: List[StageRecord] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    duration_seconds: float = 0.0
    rolled_back: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "rolled_back": self.rolled_back,
            "error": self.error,
            "stages": [s.to_dict() for s in self.stage_records],
            "validation": self.validation.to_dict() if self.validation else None,
        }


__all__ = ["PipelineResult", "StageRecord"]
