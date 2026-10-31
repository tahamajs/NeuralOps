#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L4 Empirical Latency Validation Gate.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity
from neuralops.validation.gates.numerical.forward_pass import _get_dummy_input


class LatencyGate(Gate):
    """Measures median inference latency over N repeated runs."""

    name = "latency"
    level = "L4"
    severity = GateSeverity.ERROR

    def __init__(
        self,
        max_latency_ms: Optional[float] = None,
        warmup_runs: int = 2,
        measure_runs: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=self.name, level=self.level, **kwargs)
        self.max_latency_ms = max_latency_ms
        self.warmup_runs = warmup_runs
        self.measure_runs = measure_runs

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
                message="Skipped latency: no model provided",
                duration_seconds=time.perf_counter() - t0,
            )

        ctx = context or {}
        batch_size = int(ctx.get("batch_size", 4))
        seq_len = int(ctx.get("seq_len", 16))
        warmup = int(ctx.get("warmup_runs", self.warmup_runs))
        repeats = int(ctx.get("measure_runs", self.measure_runs))
        threshold_ms = ctx.get("max_latency_ms", self.max_latency_ms)

        device = next(active_model.parameters()).device if list(active_model.parameters()) else torch.device("cpu")

        try:
            dummy_input = ctx.get("dummy_input")
            if dummy_input is None:
                dummy_input = _get_dummy_input(active_model, batch_size=batch_size, seq_len=seq_len)
            if isinstance(dummy_input, torch.Tensor):
                dummy_input = dummy_input.to(device)
            elif isinstance(dummy_input, dict):
                dummy_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dummy_input.items()}

            active_model.eval()

            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    if isinstance(dummy_input, dict):
                        _ = active_model(**dummy_input)
                    else:
                        _ = active_model(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            # Timed runs
            latencies_ms = []
            with torch.no_grad():
                for _ in range(repeats):
                    start_t = time.perf_counter()
                    if isinstance(dummy_input, dict):
                        _ = active_model(**dummy_input)
                    else:
                        _ = active_model(dummy_input)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    end_t = time.perf_counter()
                    latencies_ms.append((end_t - start_t) * 1000.0)

            median_ms = statistics.median(latencies_ms)
            p95_ms = statistics.quantiles(latencies_ms, n=100)[94] if len(latencies_ms) >= 100 else max(latencies_ms)

            passed = True
            msg = f"Median latency: {median_ms:.3f} ms (p95: {p95_ms:.3f} ms, device: {device.type})"

            if threshold_ms is not None:
                if median_ms > float(threshold_ms):
                    passed = False
                    msg = (
                        f"Median latency {median_ms:.3f} ms exceeded max threshold {float(threshold_ms):.3f} ms "
                        f"(device: {device.type})"
                    )

            return GateResult(
                gate_name=self.name,
                passed=passed,
                severity=self.severity,
                measured_value=median_ms,
                threshold=threshold_ms,
                message=msg,
                duration_seconds=time.perf_counter() - t0,
                details={
                    "median_latency_ms": median_ms,
                    "p95_latency_ms": p95_ms,
                    "device": device.type,
                    "repeats": repeats,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                },
                evidence_dict={
                    "median_latency_ms": median_ms,
                    "p95_latency_ms": p95_ms,
                    "device": device.type,
                },
            )

        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message=f"Latency measurement failed with error: {e}",
                duration_seconds=time.perf_counter() - t0,
                error_code="E_LATENCY_ERROR",
            )


__all__ = ["LatencyGate"]
