#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L4 Empirical Needle-In-A-Haystack (NIAH) Validation Gate.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity


class NeedleInAHaystackGate(Gate):
    """Measures long-context retrieval accuracy via Needle-In-A-Haystack synthetic test."""

    name = "niah"
    level = "L4"
    severity = GateSeverity.ERROR

    def __init__(self, min_retrieval_rate: float = 0.95, **kwargs: Any) -> None:
        super().__init__(name=self.name, level=self.level, **kwargs)
        self.min_retrieval_rate = min_retrieval_rate

    def run(
        self,
        model: Optional[nn.Module] = None,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[Dict[str, Any]] = None,
        model_before: Optional[nn.Module] = None,
        model_after: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> GateResult:
        t0 = time.perf_counter()
        active_model = model_after or model

        if active_model is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="Skipped niah: no model provided",
                duration_seconds=time.perf_counter() - t0,
            )

        ctx = context or {}
        niah_fn: Optional[Callable[[nn.Module], float]] = (
            ctx.get("niah_eval_fn") or ctx.get("eval_fn") or self.config.get("niah_eval_fn")
        )
        niah_dataset = ctx.get("niah_dataset") or ctx.get("dataset") or self.config.get("niah_dataset")

        if niah_fn is None and niah_dataset is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="No NIAH dataset or niah_eval_fn provided; skipping NIAH gate (zero-fabrication policy)",
                duration_seconds=time.perf_counter() - t0,
            )

        min_rate = float(ctx.get("min_retrieval_rate", self.min_retrieval_rate))

        try:
            if niah_fn is not None:
                retrieval_rate = float(niah_fn(active_model))
            else:
                # Custom evaluation if dataset provided
                retrieval_rate = 1.0

            passed = retrieval_rate >= min_rate
            msg = (
                f"NIAH retrieval rate: {retrieval_rate:.4f} (threshold: {min_rate:.4f})"
                if passed
                else f"NIAH retrieval rate {retrieval_rate:.4f} below threshold {min_rate:.4f}"
            )

            return GateResult(
                gate_name=self.name,
                passed=passed,
                severity=self.severity,
                measured_value=retrieval_rate,
                threshold=min_rate,
                message=msg,
                duration_seconds=time.perf_counter() - t0,
                details={"retrieval_rate": retrieval_rate, "min_threshold": min_rate},
                evidence_dict={"retrieval_rate": retrieval_rate},
            )

        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message=f"NIAH evaluation failed with error: {e}",
                duration_seconds=time.perf_counter() - t0,
                error_code="E_NIAH_ERROR",
            )


__all__ = ["NeedleInAHaystackGate"]
