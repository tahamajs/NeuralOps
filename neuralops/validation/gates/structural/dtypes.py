#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L1 Structural Dtypes Validation Gate.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity


class DtypesMatchGate(Gate):
    """Verifies that model parameters have consistent and expected dtypes."""

    name = "dtypes_match"
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
        if active_model is None:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message="No model provided to dtypes_match gate",
                duration_seconds=time.perf_counter() - t0,
            )

        dtypes = {p.dtype for p in active_model.parameters()}
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


__all__ = ["DtypesMatchGate"]
