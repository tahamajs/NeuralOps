#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Base definitions and interfaces for Validation Gates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import torch.nn as nn


class GateSeverity(str, Enum):
    FATAL = "FATAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class GateResult:
    """Outcome of a single validation gate execution."""

    gate_name: str
    passed: bool
    severity: GateSeverity = GateSeverity.ERROR
    measured_value: Any = None
    threshold: Any = None
    message: str = ""
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "measured_value": self.measured_value,
            "threshold": self.threshold,
            "message": self.message,
            "duration_seconds": self.duration_seconds,
            "details": self.details,
        }


class Gate(ABC):
    """Abstract Base Class for a transformation validation gate."""

    name: str
    severity: GateSeverity = GateSeverity.ERROR
    level: str = "L1"

    @abstractmethod
    def run(
        self,
        model: nn.Module,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GateResult:
        """Execute validation gate on model."""
        raise NotImplementedError


__all__ = ["Gate", "GateResult", "GateSeverity"]
