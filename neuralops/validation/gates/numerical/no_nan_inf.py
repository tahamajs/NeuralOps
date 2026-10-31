#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L2 Numerical No-NaN/Inf Validation Gate.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity
from neuralops.validation.gates.numerical.forward_pass import _get_dummy_input


class NoNanInfGate(Gate):
    """Checks that model parameters and forward outputs contain no NaN or Inf."""

    name = "no_nan_inf"
    level = "L2"
    severity = GateSeverity.FATAL

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
        if active_model is None:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message="No model provided to no_nan_inf gate",
                duration_seconds=time.perf_counter() - t0,
            )

        # 1. Check all model parameters
        for name, p in active_model.named_parameters():
            if torch.isnan(p).any():
                return GateResult(
                    gate_name=self.name,
                    passed=False,
                    severity=self.severity,
                    error_code="E_NUMERICAL_NAN_DETECTED",
                    message=f"Parameter '{name}' contains NaN values",
                    duration_seconds=time.perf_counter() - t0,
                )
            if torch.isinf(p).any():
                return GateResult(
                    gate_name=self.name,
                    passed=False,
                    severity=self.severity,
                    error_code="E_NUMERICAL_INF_DETECTED",
                    message=f"Parameter '{name}' contains Inf values",
                    duration_seconds=time.perf_counter() - t0,
                )

        # 2. Check forward pass outputs
        active_model.eval()
        dummy_in = _get_dummy_input(active_model)
        try:
            with torch.no_grad():
                out = active_model(dummy_in)
            out_tensor = out[0] if isinstance(out, (tuple, list)) else getattr(out, "logits", out)
            if isinstance(out_tensor, torch.Tensor):
                if torch.isnan(out_tensor).any():
                    return GateResult(
                        gate_name=self.name,
                        passed=False,
                        severity=self.severity,
                        error_code="E_NUMERICAL_NAN_DETECTED",
                        message="Forward output contains NaN values",
                        duration_seconds=time.perf_counter() - t0,
                    )
                if torch.isinf(out_tensor).any():
                    return GateResult(
                        gate_name=self.name,
                        passed=False,
                        severity=self.severity,
                        error_code="E_NUMERICAL_INF_DETECTED",
                        message="Forward output contains Inf values",
                        duration_seconds=time.perf_counter() - t0,
                    )
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message=f"Forward pass exception during no_nan_inf check: {e}",
                duration_seconds=time.perf_counter() - t0,
            )

        return GateResult(
            gate_name=self.name,
            passed=True,
            severity=self.severity,
            message="No NaN or Inf found in parameters or outputs",
            duration_seconds=time.perf_counter() - t0,
        )


__all__ = ["NoNanInfGate"]
