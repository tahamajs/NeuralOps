#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L4 Empirical Perplexity Validation Gate (F-57).

Evaluates absolute empirical perplexity against dataset evaluation tokens (WikiText, C4).
Distinct from L3 Behavioral relative delta gate in `neuralops.validation.gates.behavioral.perplexity`.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.validation.base import Gate, GateResult, GateSeverity


def _evaluate_perplexity(model: nn.Module, dataset: Any) -> float:
    """Evaluate empirical perplexity on a dataset of token sequences."""
    model.eval()
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)

    total_loss = 0.0
    total_tokens = 0

    if isinstance(dataset, torch.Tensor):
        batches = [dataset]
    elif isinstance(dataset, (list, tuple)):
        batches = dataset
    elif isinstance(dataset, dict) and "input_ids" in dataset:
        batches = [dataset["input_ids"]]
    elif isinstance(dataset, dict) and "x" in dataset:
        batches = [dataset["x"]]
    else:
        batches = []

    if not batches:
        raise ValueError("Dataset provided has no tokens or input_ids.")

    with torch.no_grad():
        for batch in batches:
            if isinstance(batch, dict):
                tokens = batch.get("input_ids", batch.get("x"))
            elif isinstance(batch, (tuple, list)):
                tokens = batch[0]
            else:
                tokens = batch

            if tokens is None:
                continue

            tokens = tokens.to(device)
            if tokens.ndim == 1:
                tokens = tokens.unsqueeze(0)

            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()

            outputs = model(inputs)
            logits = (
                outputs.logits
                if hasattr(outputs, "logits")
                else (outputs[0] if isinstance(outputs, (tuple, list)) else outputs)
            )

            if vocab_size is None:
                vocab_size = logits.shape[-1]

            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction="sum")
            total_loss += float(loss.item())
            total_tokens += targets.numel()

    if total_tokens == 0:
        raise ValueError("Zero evaluation tokens processed.")

    avg_loss = total_loss / total_tokens
    return float(torch.exp(torch.tensor(avg_loss)).item())


class PerplexityGate(Gate):
    """L4 Empirical Perplexity Validation Gate.

    Measures empirical perplexity on evaluation tokens (WikiText, C4, etc.)
    or via a custom evaluation function (`eval_fn`), enforcing absolute
    perplexity thresholds and fail-closed integrity under production sweeps.

    Scope Distinction:
        - L4 Empirical (`PerplexityGate` / `EmpiricalPerplexityGate`): Evaluates
          absolute perplexity on real dataset corpora, failing closed if no empirical
          dataset or eval_fn is provided in production mode.
        - L3 Behavioral (`PerplexityDeltaGate` in `neuralops.validation.gates.behavioral`):
          Specifically compares candidate models against baseline models to measure
          relative perplexity degradation (delta) across synthetic or evaluation tokens.
    """

    name = "perplexity"
    level = "L4"
    severity = GateSeverity.ERROR

    def __init__(
        self,
        max_perplexity: Optional[float] = None,
        max_perplexity_increase: Optional[float] = None,
        parallel_benchmark: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=self.name, level=self.level, **kwargs)
        self.max_perplexity = max_perplexity
        self.max_perplexity_increase = max_perplexity_increase
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

        if active_model is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="Skipped perplexity: no model provided",
                duration_seconds=time.perf_counter() - t0,
            )

        ctx = context or {}
        dataset = ctx.get("dataset") or self.config.get("dataset")
        eval_fn: Optional[Callable[[nn.Module], float]] = ctx.get("eval_fn") or self.config.get("eval_fn")
        profile = str(ctx.get("profile", "")).lower()
        allow_uncalibrated = bool(ctx.get("allow_uncalibrated", False))

        if dataset is None and eval_fn is None:
            # In production sweeps/profiles, fail-closed unless explicitly allowed
            if profile in ("production", "empirical") and not allow_uncalibrated:
                return GateResult(
                    gate_name=self.name,
                    passed=False,
                    skipped=False,
                    severity=self.severity,
                    message="Production validation requires empirical dataset or eval_fn; failing closed (--allow-uncalibrated not set)",
                    duration_seconds=time.perf_counter() - t0,
                    error_code="E_DATASET_MISSING",
                )
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="No dataset or eval_fn provided; skipping empirical perplexity gate (zero-fabrication policy)",
                duration_seconds=time.perf_counter() - t0,
            )

        bench = ctx.get("parallel_benchmark", self.parallel_benchmark)
        use_parallel = (
            bench is not None
            and hasattr(bench, "run")
            and getattr(bench, "context", None) is not None
            and (bench.context.is_distributed() or getattr(bench.context, "available_gpus", 0) > 1)
        )

        try:
            if eval_fn is not None:
                ppl = float(eval_fn(active_model))
                base_ppl = float(eval_fn(base_model)) if base_model is not None else None
            elif use_parallel and dataset is not None:

                def _ppl_metric(m: nn.Module, shard: Any) -> float:
                    return _evaluate_perplexity(m, shard)

                res = bench.run(active_model, dataset, _ppl_metric, reduce_op="mean")
                ppl = float(res.metric_value)
                if base_model is not None:
                    res_base = bench.run(base_model, dataset, _ppl_metric, reduce_op="mean")
                    base_ppl = float(res_base.metric_value)
                else:
                    base_ppl = None
            else:
                ppl = _evaluate_perplexity(active_model, dataset)
                base_ppl = _evaluate_perplexity(base_model, dataset) if base_model is not None else None
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message=f"Perplexity evaluation failed with error: {e}",
                duration_seconds=time.perf_counter() - t0,
                error_code="E_EVAL_ERROR",
            )

        max_ppl = ctx.get("max_perplexity", self.max_perplexity)
        max_inc = ctx.get("max_perplexity_increase", self.max_perplexity_increase)

        passed = True
        reasons = []

        if max_ppl is not None and ppl > float(max_ppl):
            passed = False
            reasons.append(f"Perplexity {ppl:.2f} exceeds maximum threshold {float(max_ppl):.2f}")

        if max_inc is not None and base_ppl is not None:
            rel_inc = (ppl - base_ppl) / max(1e-5, base_ppl)
            if rel_inc > float(max_inc):
                passed = False
                reasons.append(
                    f"Perplexity increase +{rel_inc*100:.2f}% exceeds maximum allowed +{float(max_inc)*100:.1f}% "
                    f"(baseline={base_ppl:.2f}, post={ppl:.2f})"
                )

        msg = "; ".join(reasons) if not passed else f"Perplexity passed: {ppl:.2f}"
        return GateResult(
            gate_name=self.name,
            passed=passed,
            severity=self.severity,
            measured_value=ppl,
            threshold={"max_perplexity": max_ppl, "max_perplexity_increase": max_inc},
            message=msg,
            duration_seconds=time.perf_counter() - t0,
            details={"perplexity": ppl, "baseline_perplexity": base_ppl, "parallel": use_parallel},
            evidence_dict={"perplexity": ppl, "baseline_perplexity": base_ppl, "parallel": use_parallel},
        )


# Distinctly named alias for architectural clarity
EmpiricalPerplexityGate = PerplexityGate

__all__ = ["EmpiricalPerplexityGate", "PerplexityGate", "_evaluate_perplexity"]
