"""Empirical benchmark validation gate module.

Provides EmpiricalBenchmarkGate for downstream accuracy and benchmark verification.
"""

from __future__ import annotations

from typing import Any, Optional
import torch.nn as nn

from neuralops.validation.profiles import EmpiricalBenchmarkGate as _BaseGate, GateResult


class EmpiricalBenchmarkGate(_BaseGate):
    """Empirical benchmark gate verifying task accuracy on downstream benchmarks."""

    def evaluate(
        self,
        model: nn.Module,
        dataset: Any = None,
        eval_fn: Optional[Any] = None,
        baseline_model: Optional[nn.Module] = None,
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GateResult:
        """Evaluate task accuracy without hardcoded fake metrics."""
        ctx = dict(context or {})
        if dataset is not None:
            ctx["eval_dataset"] = dataset
        if eval_fn is not None:
            ctx["eval_fn"] = eval_fn
        return self.run(
            model=model,
            baseline_model=baseline_model,
            context=ctx,
            **kwargs,
        )


__all__ = ["EmpiricalBenchmarkGate"]
