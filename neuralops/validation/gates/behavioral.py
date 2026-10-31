#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L3-lite Behavioral Validation Gates.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.validation.base import Gate, GateResult, GateSeverity
from neuralops.validation.gates.numerical import _get_dummy_input


class OutputSimilarityGate(Gate):
    """Measures cosine similarity between transformed model and baseline model outputs."""

    name = "output_similarity"
    level = "L3"
    severity = GateSeverity.ERROR

    def __init__(self, min_similarity: float = 0.85):
        self.min_similarity = min_similarity

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
                message="Skipped output_similarity: baseline_model not provided",
                duration_seconds=time.perf_counter() - t0,
            )

        threshold = float(context.get("min_similarity", self.min_similarity)) if context else self.min_similarity

        model.eval()
        baseline_model.eval()
        dummy_in = _get_dummy_input(model, batch_size=4, seq_len=32)

        try:
            with torch.no_grad():
                out_base = baseline_model(dummy_in)
                out_test = model(dummy_in)

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


class PerplexityDeltaGate(Gate):
    """Measures cross-entropy loss / perplexity delta vs baseline on evaluation tokens."""

    name = "perplexity_delta"
    level = "L3"
    severity = GateSeverity.ERROR

    def __init__(self, max_relative_increase: float = 0.05):
        self.max_relative_increase = max_relative_increase

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
                message="Skipped perplexity_delta: baseline_model not provided",
                duration_seconds=time.perf_counter() - t0,
            )

        threshold = float(context.get("max_perplexity_increase", self.max_relative_increase)) if context else self.max_relative_increase

        model.eval()
        baseline_model.eval()
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab_size is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                severity=self.severity,
                message="Skipped perplexity_delta: non-generative/classification architecture without vocab_size",
                duration_seconds=time.perf_counter() - t0,
            )

        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        tokens = torch.randint(0, vocab_size, (2, 32), device=device)
        targets = tokens[:, 1:].contiguous()
        inputs = tokens[:, :-1].contiguous()

        try:
            with torch.no_grad():
                out_base = baseline_model(inputs)
                out_test = model(inputs)

            logits_b = out_base[0] if isinstance(out_base, (tuple, list)) else getattr(out_base, "logits", out_base)
            logits_t = out_test[0] if isinstance(out_test, (tuple, list)) else getattr(out_test, "logits", out_test)

            loss_b = F.cross_entropy(logits_b.view(-1, vocab_size), targets.view(-1)).item()
            loss_t = F.cross_entropy(logits_t.view(-1, vocab_size), targets.view(-1)).item()

            ppl_b = torch.exp(torch.tensor(loss_b)).item()
            ppl_t = torch.exp(torch.tensor(loss_t)).item()

            rel_increase = (ppl_t - ppl_b) / max(1e-5, ppl_b)
            passed = rel_increase <= threshold

            return GateResult(
                gate_name=self.name,
                passed=passed,
                severity=self.severity,
                measured_value=rel_increase,
                threshold=threshold,
                message=f"Perplexity delta: +{rel_increase * 100:.2f}% (ppl {ppl_b:.2f} -> {ppl_t:.2f}, max allowed: +{threshold * 100:.1f}%)",
                duration_seconds=time.perf_counter() - t0,
                details={"baseline_ppl": ppl_b, "final_ppl": ppl_t},
            )
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message=f"Exception computing perplexity delta: {e}",
                duration_seconds=time.perf_counter() - t0,
            )


__all__ = ["OutputSimilarityGate", "PerplexityDeltaGate"]
