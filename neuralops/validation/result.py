#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Validation Result reporting structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from neuralops.validation.base import GateResult, GateSeverity


@dataclass
class ValidationResult:
    """Aggregate result from executing a suite of validation gates."""

    passed: bool
    profile: str
    gate_results: List[GateResult] = field(default_factory=list)
    total_gates: int = 0
    passed_gates: int = 0
    failed_gates: int = 0
    warning_count: int = 0
    duration_seconds: float = 0.0

    @classmethod
    def from_results(
        cls,
        profile: str,
        results: List[GateResult],
        duration_seconds: float = 0.0,
    ) -> ValidationResult:
        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        failed_count = sum(
            1 for r in results if not r.passed and r.severity in (GateSeverity.FATAL, GateSeverity.ERROR)
        )
        warnings = sum(1 for r in results if not r.passed and r.severity == GateSeverity.WARNING)
        overall_passed = failed_count == 0

        return cls(
            passed=overall_passed,
            profile=profile,
            gate_results=results,
            total_gates=total,
            passed_gates=passed_count,
            failed_gates=failed_count,
            warning_count=warnings,
            duration_seconds=duration_seconds,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "profile": self.profile,
            "total_gates": self.total_gates,
            "passed_gates": self.passed_gates,
            "failed_gates": self.failed_gates,
            "warning_count": self.warning_count,
            "duration_seconds": self.duration_seconds,
            "gates": [r.to_dict() for r in self.gate_results],
        }


__all__ = ["ValidationResult"]
