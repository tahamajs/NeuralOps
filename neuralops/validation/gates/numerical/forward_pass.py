#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L2 Numerical Forward Pass Execution Validation Gate.
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
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)

    if vocab_size is not None:
        return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    if hidden_size is None:
        # Check first linear layer for in_features if config is absent
        for m in model.modules():
            if isinstance(m, nn.Linear):
                hidden_size = m.in_features
                break
        if hidden_size is None:
            hidden_size = 64

    return torch.randn(batch_size, seq_len, hidden_size, device=device)


class ForwardPassRunsGate(Gate):
    """Verifies that the model forward pass executes without raising exceptions."""

    name = "forward_pass_runs"
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
                message="No model provided to forward_pass gate",
                duration_seconds=time.perf_counter() - t0,
            )

        active_model.eval()
        dummy_in = _get_dummy_input(active_model)

        try:
            with torch.no_grad():
                _ = active_model(dummy_in)
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


__all__ = ["ForwardPassRunsGate"]
