#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L1 Structural Parameter Count Validation Gate.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity


class ParameterCountGate(Gate):
    """Checks parameter count delta and verifies compression budget."""

    name = "parameter_count"
    level = "L1"
    severity = GateSeverity.ERROR

    def run(
        self,
        model: Optional[nn.Module] = None,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[Dict[str, Any]] = None,
        model_before: Optional[nn.Module] = None,
        model_after: Optional[nn.Module] = None,
    ) -> GateResult:
        t0 = time.perf_counter()
        active_model = model_after or model
        base_model = model_before or baseline_model

        if active_model is None:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message="No model provided to parameter_count gate",
                duration_seconds=time.perf_counter() - t0,
            )

        test_params = sum(p.numel() for p in active_model.parameters())

        if base_model is not None:
            base_params = sum(p.numel() for p in base_model.parameters())
            ratio = test_params / max(1, base_params)
            target_ratio = float(context.get("target_ratio", 1.0)) if context else 1.0
            target_pattern = str(context.get("target", "*")) if context else "*"

            # If target pattern is selective (not entire model "*"), whole-model parameter ratio is bounded by <= 1.0
            if target_pattern not in ("*", "*.*", "all") and ratio <= 1.0:
                passed = True
            else:
                passed = ratio <= (target_ratio + 0.05)
            return GateResult(
                gate_name=self.name,
                passed=passed,
                severity=self.severity,
                measured_value=ratio,
                threshold=target_ratio,
                message=f"Compression ratio: {ratio:.4f} (target <= {target_ratio:.4f})",
                duration_seconds=time.perf_counter() - t0,
                details={"baseline_parameters": base_params, "final_parameters": test_params},
            )

        return GateResult(
            gate_name=self.name,
            passed=True,
            severity=self.severity,
            measured_value=test_params,
            message=f"Total parameters: {test_params}",
            duration_seconds=time.perf_counter() - t0,
        )


__all__ = ["ParameterCountGate"]
