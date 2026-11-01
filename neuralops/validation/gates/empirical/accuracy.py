#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
L4 Empirical Accuracy Validation Gate.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity


def _evaluate_accuracy(model: nn.Module, dataset: Iterable[Any]) -> float:
    """Evaluate classification accuracy on an iterable dataset of (inputs, targets)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataset:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            elif isinstance(batch, dict) and "input_ids" in batch and "labels" in batch:
                inputs, targets = batch["input_ids"], batch["labels"]
            else:
                continue

            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == targets).sum().item())
            total += int(targets.numel())

    if total == 0:
        raise ValueError("Dataset produced zero evaluatable targets.")
    return float(correct / total)


class AccuracyGate(Gate):
    """Measures task accuracy on a labeled dataset (GLUE, SST-2, or synthetic mini-dataset)."""

    name = "accuracy"
    level = "L4"
    severity = GateSeverity.ERROR

    def __init__(
        self,
        min_accuracy: Optional[float] = None,
        max_accuracy_drop: Optional[float] = None,
        parallel_benchmark: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=self.name, level=self.level, **kwargs)
        self.min_accuracy = min_accuracy
        self.max_accuracy_drop = max_accuracy_drop
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
                message="Skipped accuracy: no model provided",
                duration_seconds=time.perf_counter() - t0,
            )

        ctx = context or {}
        dataset = ctx.get("dataset") or self.config.get("dataset")
        eval_fn: Optional[Callable[[nn.Module], float]] = ctx.get("eval_fn") or self.config.get("eval_fn")

        if dataset is None and eval_fn is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="No dataset or eval_fn provided; skipping empirical accuracy gate (zero-fabrication policy)",
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
                acc = float(eval_fn(active_model))
                base_acc = float(eval_fn(base_model)) if base_model is not None else None
            elif use_parallel and dataset is not None:
                def _acc_metric(m: nn.Module, d: Any) -> float:
                    return _evaluate_accuracy(m, d)

                res = bench.run(active_model, dataset, _acc_metric)
                acc = float(res.metric_value)
                base_acc = float(bench.run(base_model, dataset, _acc_metric).metric_value) if base_model is not None else None
            else:
                acc = _evaluate_accuracy(active_model, dataset)
                base_acc = _evaluate_accuracy(base_model, dataset) if base_model is not None else None
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=self.severity,
                message=f"Accuracy evaluation failed with error: {e}",
                duration_seconds=time.perf_counter() - t0,
                error_code="E_EVAL_ERROR",
            )

        min_acc = ctx.get("min_accuracy", self.min_accuracy)
        max_drop = ctx.get("max_accuracy_drop", self.max_accuracy_drop)

        passed = True
        reasons = []

        if min_acc is not None and acc < float(min_acc):
            passed = False
            reasons.append(f"Accuracy {acc:.4f} is below minimum threshold {float(min_acc):.4f}")

        if max_drop is not None and base_acc is not None:
            drop = base_acc - acc
            if drop > float(max_drop):
                passed = False
                reasons.append(
                    f"Accuracy drop {drop:.4f} exceeds maximum allowed drop {float(max_drop):.4f} "
                    f"(baseline={base_acc:.4f}, post={acc:.4f})"
                )

        msg = "; ".join(reasons) if not passed else f"Accuracy passed: {acc:.4f}"
        return GateResult(
            gate_name=self.name,
            passed=passed,
            severity=self.severity,
            measured_value=acc,
            threshold={"min_accuracy": min_acc, "max_accuracy_drop": max_drop},
            message=msg,
            duration_seconds=time.perf_counter() - t0,
            details={"accuracy": acc, "baseline_accuracy": base_acc},
            evidence_dict={"accuracy": acc, "baseline_accuracy": base_acc},
        )


__all__ = ["AccuracyGate"]
