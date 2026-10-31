#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L3 Behavioral Logit KL Divergence Validation Gate.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.validation.base import Gate, GateResult, GateSeverity
from neuralops.validation.gates.numerical.forward_pass import _get_dummy_input


class LogitKLGate(Gate):
    """Measures Kullback-Leibler divergence between baseline logits and transformed logits."""

    name = "logit_kl"
    level = "L3"
    severity = GateSeverity.ERROR

    def __init__(self, max_kl: float = 0.01, **kwargs: Any) -> None:
        super().__init__(name=self.name, level=self.level, **kwargs)
        self.max_kl = max_kl

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
                message="Skipped logit_kl: requires both model_before and model_after",
                duration_seconds=time.perf_counter() - t0,
            )

        threshold = float(context.get("max_kl", self.max_kl)) if context else self.max_kl
        batch_size = int(context.get("batch_size", 4)) if context else 4
        seq_len = int(context.get("seq_len", 16)) if context else 16

        active_model.eval()
        base_model.eval()

        dummy_in = _get_dummy_input(active_model, batch_size=batch_size, seq_len=seq_len)

        try:
            with torch.no_grad():
                out_base = base_model(dummy_in)
                out_test = active_model(dummy_in)

            logits_b = out_base[0] if isinstance(out_base, (tuple, list)) else getattr(out_base, "logits", out_base)
            logits_t = out_test[0] if isinstance(out_test, (tuple, list)) else getattr(out_test, "logits", out_test)

            if not isinstance(logits_b, torch.Tensor) or not isinstance(logits_t, torch.Tensor):
                return GateResult(
                    gate_name=self.name,
                    passed=True,
                    skipped=True,
                    severity=self.severity,
                    message="Skipped logit_kl: non-tensor output",
                    duration_seconds=time.perf_counter() - t0,
                )

            # Compute KL divergence: KL(P_base || Q_test)
            # F.kl_div expects input to be log-probabilities and target to be probabilities
            p_base = F.softmax(logits_b.float(), dim=-1)
            log_q_test = F.log_softmax(logits_t.float(), dim=-1)

            kl_div = F.kl_div(log_q_test, p_base, reduction="batchmean").item()
            passed = kl_div <= threshold

            return GateResult(
                gate_name=self.name,
                passed=passed,
                severity=self.severity,
                measured_value=kl_div,
                threshold=threshold,
                message=f"Logit KL divergence: {kl_div:.6f} (max allowed: {threshold:.6f})",
                duration_seconds=time.perf_counter() - t0,
                details={"kl_divergence": kl_div, "threshold": threshold},
            )
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message=f"Exception computing logit KL divergence: {e}",
                duration_seconds=time.perf_counter() - t0,
            )


__all__ = ["LogitKLGate"]
