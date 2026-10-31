#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L1 Structural Shapes Validation Gate.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity


class ShapesMatchGate(Gate):
    """Verifies that model output shape matches baseline output shape on standard input."""

    name = "shapes_match"
    level = "L1"
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
        base_model = model_before or baseline_model

        if active_model is None:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message="No model provided to shapes_match gate",
                duration_seconds=time.perf_counter() - t0,
            )

        if base_model is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                severity=self.severity,
                message="Skipped shapes_match: no baseline model provided",
                duration_seconds=time.perf_counter() - t0,
            )

        active_model.eval()
        base_model.eval()
        device = next(active_model.parameters()).device if list(active_model.parameters()) else torch.device("cpu")

        # Create dummy batch
        vocab_size = getattr(getattr(active_model, "config", None), "vocab_size", None)
        hidden_size = getattr(getattr(active_model, "config", None), "hidden_size", 64)
        seq_len = 16
        batch_size = 2

        with torch.no_grad():
            try:
                if vocab_size is not None:
                    dummy_in = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                    out_base = base_model(dummy_in)
                    out_test = active_model(dummy_in)
                else:
                    dummy_in = torch.randn(batch_size, seq_len, hidden_size, device=device)
                    out_base = base_model(dummy_in)
                    out_test = active_model(dummy_in)
            except Exception as e:
                return GateResult(
                    gate_name=self.name,
                    passed=False,
                    severity=self.severity,
                    message=f"Forward pass raised exception during shapes_match: {e}",
                    duration_seconds=time.perf_counter() - t0,
                )

            base_tensor = out_base[0] if isinstance(out_base, (tuple, list)) else getattr(out_base, "logits", out_base)
            test_tensor = out_test[0] if isinstance(out_test, (tuple, list)) else getattr(out_test, "logits", out_test)

            if isinstance(base_tensor, torch.Tensor) and isinstance(test_tensor, torch.Tensor):
                matched = base_tensor.shape == test_tensor.shape
                return GateResult(
                    gate_name=self.name,
                    passed=matched,
                    severity=self.severity,
                    measured_value=list(test_tensor.shape),
                    threshold=list(base_tensor.shape),
                    message="Shapes match" if matched else f"Shape mismatch: {test_tensor.shape} vs {base_tensor.shape}",
                    duration_seconds=time.perf_counter() - t0,
                )

        return GateResult(
            gate_name=self.name,
            passed=True,
            severity=self.severity,
            message="Shapes match (non-tensor outputs)",
            duration_seconds=time.perf_counter() - t0,
        )


__all__ = ["ShapesMatchGate"]
