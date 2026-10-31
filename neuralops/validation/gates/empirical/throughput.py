#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L4 Empirical Throughput Validation Gate.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity
from neuralops.validation.gates.numerical.forward_pass import _get_dummy_input


class ThroughputGate(Gate):
    """Measures inference throughput (tokens/sec or samples/sec)."""

    name = "throughput"
    level = "L4"
    severity = GateSeverity.ERROR

    def __init__(
        self,
        min_throughput: Optional[float] = None,
        warmup_runs: int = 2,
        measure_runs: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=self.name, level=self.level, **kwargs)
        self.min_throughput = min_throughput
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
                message="Skipped throughput: no model provided",
                duration_seconds=time.perf_counter() - t0,
            )

        ctx = context or {}
        batch_size = int(ctx.get("batch_size", 4))
        seq_len = int(ctx.get("seq_len", 16))
        warmup = int(ctx.get("warmup_runs", self.warmup_runs))
        repeats = int(ctx.get("measure_runs", self.measure_runs))
        threshold = ctx.get("min_throughput", self.min_throughput)

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

            # Timed batch runs
            t_start = time.perf_counter()
            with torch.no_grad():
                for _ in range(repeats):
                    if isinstance(dummy_input, dict):
                        _ = active_model(**dummy_input)
                    else:
                        _ = active_model(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            t_end = time.perf_counter()

            total_elapsed = max(t_end - t_start, 1e-9)
            total_tokens = batch_size * seq_len * repeats
            tokens_per_sec = float(total_tokens / total_elapsed)
            samples_per_sec = float((batch_size * repeats) / total_elapsed)

            passed = True
            msg = f"Throughput: {tokens_per_sec:.1f} tokens/sec ({samples_per_sec:.1f} samples/sec, device: {device.type})"

            if threshold is not None:
                if tokens_per_sec < float(threshold):
                    passed = False
                    msg = (
                        f"Throughput {tokens_per_sec:.1f} tokens/sec below minimum threshold "
                        f"{float(threshold):.1f} tokens/sec (device: {device.type})"
                    )

            return GateResult(
                gate_name=self.name,
                passed=passed,
                severity=self.severity,
                measured_value=tokens_per_sec,
                threshold=threshold,
                message=msg,
                duration_seconds=time.perf_counter() - t0,
                details={
                    "tokens_per_second": tokens_per_sec,
                    "samples_per_second": samples_per_sec,
                    "total_tokens": total_tokens,
                    "elapsed_seconds": total_elapsed,
                    "device": device.type,
                },
                evidence_dict={
                    "tokens_per_second": tokens_per_sec,
                    "samples_per_second": samples_per_sec,
                    "device": device.type,
                },
            )

        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message=f"Throughput measurement failed with error: {e}",
                duration_seconds=time.perf_counter() - t0,
                error_code="E_THROUGHPUT_ERROR",
            )


__all__ = ["ThroughputGate"]
