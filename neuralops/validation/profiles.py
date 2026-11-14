#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Validation Profiles mapping rigor levels to gate suites.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from neuralops.validation.base import Gate, GateResult, GateSeverity
from neuralops.validation.gates.behavioral import OutputSimilarityGate, PerplexityDeltaGate
from neuralops.validation.gates.numerical import ForwardPassRunsGate, NoNanInfGate
from neuralops.validation.gates.structural import ConnectivityGate, DtypesMatchGate, ParameterCountGate, ShapesMatchGate


class ValidationProfile(str, Enum):
    """Standardized validation rigor profiles."""

    NONE = "NONE"
    STRUCTURAL = "STRUCTURAL"
    NUMERICAL = "NUMERICAL"
    BEHAVIORAL = "BEHAVIORAL"
    EMPIRICAL = "EMPIRICAL"
    PRODUCTION = "PRODUCTION"


class EmpiricalBenchmarkGate(Gate):
    """Gate verifying empirical task accuracy on downstream benchmarks."""

    name = "empirical_benchmark"
    level = "L4"
    severity = GateSeverity.WARNING

    def __init__(
        self,
        min_score: Optional[float] = None,
        max_score_drop: Optional[float] = None,
        name: Optional[str] = None,
        level: Optional[str] = None,
        required: bool = True,
        config: Optional[Dict[str, Any]] = None,
        severity: GateSeverity = GateSeverity.WARNING,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name or self.name,
            level=level or self.level,
            required=required,
            config=config,
            severity=severity,
            **kwargs,
        )
        self.min_score = min_score
        self.max_score_drop = max_score_drop

    @staticmethod
    def _evaluate_on_dataset(model: nn.Module, dataset: Any) -> float:
        """Evaluate accuracy on iterable dataset of batches."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataset:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict) and "input_ids" in batch and "labels" in batch:
                    inputs, targets = batch["input_ids"], batch["labels"]
                elif isinstance(batch, dict) and "inputs" in batch and "targets" in batch:
                    inputs, targets = batch["inputs"], batch["targets"]
                else:
                    continue

                if isinstance(inputs, dict):
                    outputs = model(**inputs)
                else:
                    outputs = model(inputs)

                logits = getattr(outputs, "logits", outputs)
                if isinstance(logits, (tuple, list)) and len(logits) > 0:
                    logits = logits[0]
                if isinstance(logits, torch.Tensor) and logits.ndim > 1:
                    preds = torch.argmax(logits, dim=-1)
                else:
                    preds = logits

                if isinstance(targets, torch.Tensor) and isinstance(preds, torch.Tensor):
                    correct += int((preds == targets).sum().item())
                    total += int(targets.numel())

        if total == 0:
            raise ValueError("Dataset produced zero evaluatable targets.")
        return float(correct / total)

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
        ctx = context or {}
        eval_dataset = ctx.get("eval_dataset") or ctx.get("dataset") or self.config.get("eval_dataset")
        eval_fn: Optional[Callable[[nn.Module], float]] = ctx.get("eval_fn") or self.config.get("eval_fn")

        # If dataset or evaluation environment is unavailable, mark skipped=True
        if eval_dataset is None and eval_fn is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="Empirical evaluation dataset not configured; gate skipped",
                duration_seconds=time.perf_counter() - t0,
            )

        active_model = model_after if model_after is not None else model
        base_model = model_before if model_before is not None else baseline_model

        if active_model is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                skipped=True,
                severity=self.severity,
                message="Skipped empirical benchmark: no model provided",
                duration_seconds=time.perf_counter() - t0,
            )

        min_score = ctx.get("min_score", ctx.get("min_accuracy", self.min_score))
        max_score_drop = ctx.get("max_score_drop", ctx.get("max_accuracy_drop", self.max_score_drop))

        try:
            if eval_fn is not None:
                score = float(eval_fn(active_model))
                base_score = float(eval_fn(base_model)) if base_model is not None else None
            else:
                score = self._evaluate_on_dataset(active_model, eval_dataset)
                base_score = self._evaluate_on_dataset(base_model, eval_dataset) if base_model is not None else None
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                skipped=False,
                severity=self.severity,
                message=f"Empirical benchmark evaluation failed with error: {e}",
                duration_seconds=time.perf_counter() - t0,
                error_code="E_EVAL_ERROR",
            )

        passed = True
        reasons: List[str] = []

        if min_score is not None and score < float(min_score):
            passed = False
            reasons.append(f"Score {score:.4f} is below minimum threshold {float(min_score):.4f}")

        if max_score_drop is not None and base_score is not None:
            drop = base_score - score
            if drop > float(max_score_drop):
                passed = False
                reasons.append(
                    f"Score drop {drop:.4f} exceeds maximum allowed drop {float(max_score_drop):.4f} "
                    f"(baseline={base_score:.4f}, post={score:.4f})"
                )

        msg = "; ".join(reasons) if not passed else f"Empirical benchmark passed: score {score:.4f}"
        return GateResult(
            gate_name=self.name,
            passed=passed,
            skipped=False,
            severity=self.severity,
            measured_value=score,
            threshold={"min_score": min_score, "max_score_drop": max_score_drop},
            message=msg,
            duration_seconds=time.perf_counter() - t0,
            details={"score": score, "baseline_score": base_score},
            evidence_dict={"score": score, "baseline_score": base_score},
        )


def get_gates_for_profile(profile: Union[str, ValidationProfile]) -> List[Gate]:
    """Retrieve list of validation gates associated with given profile."""
    norm = profile.value.lower() if isinstance(profile, ValidationProfile) else str(profile).lower()

    if norm == "none":
        return []

    # STRUCTURAL: L1 only (NO L2 gates) per Phase 2 spec:
    # "run_validation_gates with profile=STRUCTURAL runs only L1 gates (not L2)"
    if norm == "structural":
        return [
            ShapesMatchGate(),
            DtypesMatchGate(),
            ConnectivityGate(),
            ParameterCountGate(),
        ]

    # QUICK: L1 + ForwardPassRunsGate per Category 5 conformance
    if norm == "quick":
        return [
            ShapesMatchGate(),
            DtypesMatchGate(),
            ConnectivityGate(),
            ParameterCountGate(),
            ForwardPassRunsGate(),
        ]

    # NUMERICAL: L1 + L2 per Phase 2 spec
    if norm == "numerical":
        return [
            ShapesMatchGate(),
            DtypesMatchGate(),
            ConnectivityGate(),
            ParameterCountGate(),
            ForwardPassRunsGate(),
            NoNanInfGate(),
        ]

    # STANDARD: L1 + L2 + OutputSimilarity per Category 5 conformance
    if norm == "standard":
        return [
            ShapesMatchGate(),
            DtypesMatchGate(),
            ConnectivityGate(),
            ParameterCountGate(),
            ForwardPassRunsGate(),
            NoNanInfGate(),
            OutputSimilarityGate(min_similarity=0.80),
        ]

    # BEHAVIORAL: L1 + L2 + L3
    if norm in ("behavioral", "thorough"):
        return [
            ShapesMatchGate(),
            DtypesMatchGate(),
            ConnectivityGate(),
            ParameterCountGate(),
            ForwardPassRunsGate(),
            NoNanInfGate(),
            OutputSimilarityGate(min_similarity=0.80),
            PerplexityDeltaGate(max_relative_increase=0.08),
        ]

    # PRODUCTION / EMPIRICAL: L1 + L2 + L3 + L4 (empirical)
    if norm in ("production", "empirical"):
        return [
            ShapesMatchGate(),
            DtypesMatchGate(),
            ConnectivityGate(),
            ParameterCountGate(),
            ForwardPassRunsGate(),
            NoNanInfGate(),
            OutputSimilarityGate(min_similarity=0.85),
            PerplexityDeltaGate(max_relative_increase=0.05),
            EmpiricalBenchmarkGate(),
        ]

    # Default fallback: structural
    return [
        ShapesMatchGate(),
        DtypesMatchGate(),
        ConnectivityGate(),
        ParameterCountGate(),
    ]


__all__ = ["EmpiricalBenchmarkGate", "ValidationProfile", "get_gates_for_profile"]
