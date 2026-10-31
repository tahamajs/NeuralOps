#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L2 Numerical Validation Gates.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity


def _get_dummy_input(model: nn.Module, batch_size: int = 2, seq_len: int = 16) -> Any:
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", 64)

    if vocab_size is not None:
        return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return torch.randn(batch_size, seq_len, hidden_size, device=device)


class ForwardPassRunsGate(Gate):
    """Verifies that the model forward pass executes without raising exceptions."""

    name = "forward_pass_runs"
    level = "L2"
    severity = GateSeverity.FATAL

    def run(
        self,
        model: nn.Module,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GateResult:
        t0 = time.perf_counter()
        model.eval()
        dummy_in = _get_dummy_input(model)

        try:
            with torch.no_grad():
                _ = model(dummy_in)
            return GateResult(
                gate_name=self.name,
                passed=True,
                severity=self.severity,
                message="Forward pass completed successfully",
                duration_seconds=time.perf_counter() - t0,
            )
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message=f"Forward pass failed with exception: {e}",
                duration_seconds=time.perf_counter() - t0,
            )


class NoNanInfGate(Gate):
    """Checks that model parameters and forward outputs contain no NaN or Inf."""

    name = "no_nan_inf"
    level = "L2"
    severity = GateSeverity.FATAL

    def run(
        self,
        model: nn.Module,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GateResult:
        t0 = time.perf_counter()

        # Check model parameters
        for name, p in model.named_parameters():
            if torch.isnan(p).any():
                return GateResult(
                    gate_name=self.name,
                    passed=False,
                    severity=self.severity,
                    message=f"Parameter '{name}' contains NaN values",
                    duration_seconds=time.perf_counter() - t0,
                )
            if torch.isinf(p).any():
                return GateResult(
                    gate_name=self.name,
                    passed=False,
                    severity=self.severity,
                    message=f"Parameter '{name}' contains Inf values",
                    duration_seconds=time.perf_counter() - t0,
                )

        # Check forward pass outputs
        model.eval()
        dummy_in = _get_dummy_input(model)
        try:
            with torch.no_grad():
                out = model(dummy_in)
            out_tensor = out[0] if isinstance(out, (tuple, list)) else getattr(out, "logits", out)
            if isinstance(out_tensor, torch.Tensor):
                if torch.isnan(out_tensor).any():
                    return GateResult(
                        gate_name=self.name,
                        passed=False,
                        severity=self.severity,
                        message="Forward output contains NaN values",
                        duration_seconds=time.perf_counter() - t0,
                    )
                if torch.isinf(out_tensor).any():
                    return GateResult(
                        gate_name=self.name,
                        passed=False,
                        severity=self.severity,
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


__all__ = ["ForwardPassRunsGate", "NoNanInfGate"]
