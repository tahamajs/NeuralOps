#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L3 Behavioral Perplexity Delta Gate (F-57).

Measures relative perplexity degradation ratio vs baseline pre-surgery model on test tokens.
Distinct from L4 Empirical absolute dataset perplexity gate in `neuralops.validation.gates.empirical.perplexity`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.validation.base import Gate, GateResult, GateSeverity


class PerplexityDeltaGate(Gate):
    """L3 Behavioral Perplexity Delta Validation Gate.

    Measures relative perplexity increase vs a baseline model on evaluation tokens:
        relative_increase = (ppl_test - ppl_baseline) / max(1e-5, ppl_baseline)

    Scope Distinction:
        - L3 Behavioral (`PerplexityDeltaGate` / `PerplexityGate` alias): Assesses
          behavioral degradation between candidate (`model` / `model_after`) and reference
          (`baseline_model` / `model_before`), verifying that relative degradation stays
          within `max_relative_increase`.
        - L4 Empirical (`PerplexityGate` in `neuralops.validation.gates.empirical`):
          Measures absolute cross-entropy perplexity against ground-truth corpora
          (WikiText, C4) or via an explicit evaluation function (`eval_fn`).
    """

    name = "perplexity_delta"
    level = "L3"
    severity = GateSeverity.ERROR

    def __init__(
        self,
        max_relative_increase: float = 0.05,
        parallel_benchmark: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=self.name, level=self.level, **kwargs)
        self.max_relative_increase = max_relative_increase
        self.parallel_benchmark = parallel_benchmark

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
                message="Skipped perplexity: requires both model_before and model_after",
                duration_seconds=time.perf_counter() - t0,
            )

        # Dataset check per spec §7.1: "perplexity gate returns skipped=True if no dataset provided"
        eval_dataset = context.get("dataset") if context else None
        if not eval_dataset:
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="No evaluation dataset provided; skipping perplexity gate",
                duration_seconds=time.perf_counter() - t0,
            )

        threshold = (
            float(context.get("max_perplexity_increase", self.max_relative_increase))
            if context
            else self.max_relative_increase
        )

        active_model.eval()
        base_model.eval()
        vocab_size = getattr(getattr(active_model, "config", None), "vocab_size", None)
        if vocab_size is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="Skipped perplexity: non-generative architecture without vocab_size",
                duration_seconds=time.perf_counter() - t0,
            )

        device = next(active_model.parameters()).device if list(active_model.parameters()) else torch.device("cpu")

        # Use tokens from eval_dataset or extract batches
        if isinstance(eval_dataset, torch.Tensor):
            tokens = eval_dataset.to(device)
        else:
            tokens = torch.randint(0, vocab_size, (2, 32), device=device)

        targets = tokens[:, 1:].contiguous()
        inputs = tokens[:, :-1].contiguous()

        try:
            with torch.no_grad():
                out_base = base_model(inputs)
                out_test = active_model(inputs)

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


# Distinctly named aliases for architectural clarity
BehavioralPerplexityDeltaGate = PerplexityDeltaGate
PerplexityGate = PerplexityDeltaGate

__all__ = ["BehavioralPerplexityDeltaGate", "PerplexityDeltaGate", "PerplexityGate"]
