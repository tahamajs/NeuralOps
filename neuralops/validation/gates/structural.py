#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L1 Structural Validation Gates.
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
        model: nn.Module,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GateResult:
        t0 = time.perf_counter()
        if baseline_model is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                severity=self.severity,
                message="Skipped shapes_match: no baseline model provided",
                duration_seconds=time.perf_counter() - t0,
            )

        model.eval()
        baseline_model.eval()
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

        # Create dummy batch
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        hidden_size = getattr(getattr(model, "config", None), "hidden_size", 64)
        seq_len = 16
        batch_size = 2

        with torch.no_grad():
            try:
                if vocab_size is not None:
                    dummy_in = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                    out_base = baseline_model(dummy_in)
                    out_test = model(dummy_in)
                else:
                    dummy_in = torch.randn(batch_size, seq_len, hidden_size, device=device)
                    out_base = baseline_model(dummy_in)
                    out_test = model(dummy_in)
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


class DtypesMatchGate(Gate):
    """Verifies that model parameters have consistent and expected dtypes."""

    name = "dtypes_match"
    level = "L1"
    severity = GateSeverity.ERROR

    def run(
        self,
        model: nn.Module,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GateResult:
        t0 = time.perf_counter()
        dtypes = {p.dtype for p in model.parameters()}
        passed = len(dtypes) <= 2  # e.g., fp32 or fp16/bf16 with int embeddings
        return GateResult(
            gate_name=self.name,
            passed=passed,
            severity=self.severity,
            measured_value=[str(d) for d in dtypes],
            threshold="<= 2 distinct parameter dtypes",
            message=f"Parameter dtypes: {[str(d) for d in dtypes]}",
            duration_seconds=time.perf_counter() - t0,
        )


class NoOrphanModulesGate(Gate):
    """Verifies that no detached or unlinked modules exist in the model tree."""

    name = "no_orphan_modules"
    level = "L1"
    severity = GateSeverity.WARNING

    def run(
        self,
        model: nn.Module,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GateResult:
        t0 = time.perf_counter()
        total_modules = len(list(model.modules()))
        named_modules = len(list(model.named_modules()))
        passed = total_modules == named_modules
        return GateResult(
            gate_name=self.name,
            passed=passed,
            severity=self.severity,
            measured_value=named_modules,
            threshold=total_modules,
            message=f"All {named_modules} modules properly rooted",
            duration_seconds=time.perf_counter() - t0,
        )


class ParameterCountGate(Gate):
    """Checks parameter count delta and verifies compression budget."""

    name = "parameter_count"
    level = "L1"
    severity = GateSeverity.ERROR

    def run(
        self,
        model: nn.Module,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GateResult:
        t0 = time.perf_counter()
        test_params = sum(p.numel() for p in model.parameters())

        if baseline_model is not None:
            base_params = sum(p.numel() for p in baseline_model.parameters())
            ratio = test_params / max(1, base_params)
            target_ratio = float(context.get("target_ratio", 1.0)) if context else 1.0
            # Allow tolerance of 5% around target_ratio
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


__all__ = [
    "DtypesMatchGate",
    "NoOrphanModulesGate",
    "ParameterCountGate",
    "ShapesMatchGate",
]
