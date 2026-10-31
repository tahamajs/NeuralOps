#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L3 Behavioral Output Cosine Similarity Validation Gate.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.validation.base import Gate, GateResult, GateSeverity
from neuralops.validation.gates.numerical.forward_pass import _get_dummy_input


class OutputSimilarityGate(Gate):
    """Measures cosine similarity between transformed model and baseline model outputs."""

    name = "output_similarity"
    level = "L3"
    severity = GateSeverity.ERROR

    def __init__(self, min_similarity: float = 0.85, **kwargs: Any) -> None:
        super().__init__(name=self.name, level=self.level, **kwargs)
        self.min_similarity = min_similarity

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
        base_model = model_before or baseline_model

        if active_model is None or base_model is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="Skipped output_similarity: baseline_model not provided",
                duration_seconds=time.perf_counter() - t0,
            )

        threshold = float(context.get("min_similarity", self.min_similarity)) if context else self.min_similarity

        active_model.eval()
        base_model.eval()
        dummy_in = _get_dummy_input(active_model, batch_size=4, seq_len=32)

        try:
            with torch.no_grad():
                out_base = base_model(dummy_in)
                out_test = active_model(dummy_in)

            tb = out_base[0] if isinstance(out_base, (tuple, list)) else getattr(out_base, "logits", out_base)
            tt = out_test[0] if isinstance(out_test, (tuple, list)) else getattr(out_test, "logits", out_test)

            if not isinstance(tb, torch.Tensor) or not isinstance(tt, torch.Tensor):
                return GateResult(
                    gate_name=self.name,
                    passed=True,
                    severity=self.severity,
                    message="Skipped output_similarity: non-tensor output",
                    duration_seconds=time.perf_counter() - t0,
                )

            flat_b = tb.view(-1).float()
            flat_t = tt.view(-1).float()
            sim = F.cosine_similarity(flat_b.unsqueeze(0), flat_t.unsqueeze(0)).item()
            passed = sim >= threshold

            return GateResult(
                gate_name=self.name,
                passed=passed,
                severity=self.severity,
                measured_value=sim,
                threshold=threshold,
                message=f"Cosine similarity: {sim:.4f} (threshold >= {threshold:.4f})",
                duration_seconds=time.perf_counter() - t0,
            )
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message=f"Exception computing output similarity: {e}",
                duration_seconds=time.perf_counter() - t0,
            )


__all__ = ["OutputSimilarityGate"]
