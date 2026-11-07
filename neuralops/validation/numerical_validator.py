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
        if torch.isnan(tensor).any():
            raise ValueError(f"NaN in {name}")
        if torch.isinf(tensor).any():
            raise ValueError(f"Inf in {name}")
        return ValidationStatus({"status": "passed", "passed": True, "name": name})

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
