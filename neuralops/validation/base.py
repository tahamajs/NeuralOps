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
    skipped: bool = False
    evidence_dict: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None

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
            "skipped": self.skipped,
            "evidence_dict": self.evidence_dict,
            "error_code": self.error_code,
        }


class Gate(ABC):
    """Abstract Base Class for a transformation validation gate."""

    name: str = "base_gate"
    severity: GateSeverity = GateSeverity.ERROR
    level: str = "L1"
    required: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: Optional[str] = None,
        level: Optional[str] = None,
        required: bool = True,
        config: Optional[Dict[str, Any]] = None,
        severity: GateSeverity = GateSeverity.ERROR,
    ) -> None:
        if name:
            self.name = name
        if level:
            self.level = level
        self.required = required
        self.config = config or {}
        self.severity = severity

    @abstractmethod
    def run(
        self,
        model: Optional[nn.Module] = None,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[Dict[str, Any]] = None,
        model_before: Optional[nn.Module] = None,
        model_after: Optional[nn.Module] = None,
    ) -> GateResult:
        """Execute validation gate on model."""
        raise NotImplementedError


__all__ = ["Gate", "GateResult", "GateSeverity"]
