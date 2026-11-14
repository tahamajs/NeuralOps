#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Numerical Stability Validator.

Checks models and layers for numerical stability, NaN/Inf values, and gradient overflows.
Returns structured ValidationStatus without stdout printing (F-57, F-97).
"""

from __future__ import annotations

import torch

from neuralops.validation.training_validator import ValidationStatus


class NumericalValidator:
    """Numerical stability and gradient norm validator returning structured status."""

    @staticmethod
    def check_finite(tensor: torch.Tensor, name: str) -> ValidationStatus:
        if not isinstance(tensor, torch.Tensor):
            return ValidationStatus({
                "status": "failed",
                "passed": False,
                "name": name,
                "error": f"Expected torch.Tensor, got {type(tensor).__name__}",
                "has_nan": False,
                "has_inf": False,
            })
        has_nan = bool(torch.isnan(tensor).any().item())
        has_inf = bool(torch.isinf(tensor).any().item())
        passed = not (has_nan or has_inf)
        error = None
        if has_nan:
            error = f"NaN in {name}"
        elif has_inf:
            error = f"Inf in {name}"

        status = "passed" if passed else "failed"
        data = {
            "status": status,
            "passed": passed,
            "name": name,
            "has_nan": has_nan,
            "has_inf": has_inf,
        }
        if error:
            data["error"] = error
        return ValidationStatus(data)

    @staticmethod
    def check_range(
        tensor: torch.Tensor,
        name: str,
        min_val: float = -1e3,
        max_val: float = 1e3,
    ) -> ValidationStatus:
        max_abs = float(tensor.abs().max().item()) if isinstance(tensor, torch.Tensor) else float(abs(tensor))
        within_range = max_abs <= max_val
        return ValidationStatus({
            "status": "passed" if within_range else "warning",
            "passed": within_range,
            "name": name,
            "max_abs": max_abs,
            "min_val": min_val,
            "max_val": max_val,
        })

    @staticmethod
    def gradient_norm(model: torch.nn.Module) -> float:
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total**0.5


__all__ = ["NumericalValidator", "ValidationStatus"]
