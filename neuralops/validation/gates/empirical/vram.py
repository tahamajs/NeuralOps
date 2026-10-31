#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L4 Empirical Peak VRAM Validation Gate.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity
from neuralops.validation.gates.numerical.forward_pass import _get_dummy_input


class VRAMGate(Gate):
    """Measures peak GPU VRAM allocation via torch.cuda.max_memory_allocated.

    Adheres strictly to the zero-fabrication invariant (AGENTS.md §6.2):
    When executed on CPU or when CUDA is unavailable, returns skipped=True.
    """

    name = "vram"
    level = "L4"
    severity = GateSeverity.ERROR

    def __init__(self, max_vram_mb: Optional[float] = None, **kwargs: Any) -> None:
        super().__init__(name=self.name, level=self.level, **kwargs)
        self.max_vram_mb = max_vram_mb

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

        if not torch.cuda.is_available():
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="CUDA is not available; peak VRAM measurement skipped (zero-fabrication policy)",
                duration_seconds=time.perf_counter() - t0,
            )

        if active_model is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="Skipped vram: no model provided",
                duration_seconds=time.perf_counter() - t0,
            )

        ctx = context or {}
        batch_size = int(ctx.get("batch_size", 4))
        seq_len = int(ctx.get("seq_len", 16))
        threshold_mb = ctx.get("max_vram_mb", self.max_vram_mb)

        try:
            device = torch.device("cuda:0")
            active_model = active_model.to(device)
            active_model.eval()

            dummy_input = _get_dummy_input(active_model, batch_size=batch_size, seq_len=seq_len)
            if isinstance(dummy_input, torch.Tensor):
                dummy_input = dummy_input.to(device)
            elif isinstance(dummy_input, dict):
                dummy_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dummy_input.items()}

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            with torch.no_grad():
                if isinstance(dummy_input, dict):
                    _ = active_model(**dummy_input)
                else:
                    _ = active_model(dummy_input)
                torch.cuda.synchronize(device)

            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_mb = float(peak_bytes / (1024 * 1024))

            passed = True
            msg = f"Peak VRAM: {peak_mb:.2f} MB ({peak_bytes} bytes)"

            if threshold_mb is not None:
                if peak_mb > float(threshold_mb):
                    passed = False
                    msg = (
                        f"Peak VRAM {peak_mb:.2f} MB exceeded maximum threshold "
                        f"{float(threshold_mb):.2f} MB"
                    )

            return GateResult(
                gate_name=self.name,
                passed=passed,
                severity=self.severity,
                measured_value=peak_mb,
                threshold=threshold_mb,
                message=msg,
                duration_seconds=time.perf_counter() - t0,
                details={
                    "peak_vram_mb": peak_mb,
                    "peak_vram_bytes": peak_bytes,
                    "device": str(device),
                },
                evidence_dict={"peak_vram_mb": peak_mb, "device": str(device)},
            )

        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message=f"VRAM measurement failed with error: {e}",
                duration_seconds=time.perf_counter() - t0,
                error_code="E_VRAM_ERROR",
            )


__all__ = ["VRAMGate"]
