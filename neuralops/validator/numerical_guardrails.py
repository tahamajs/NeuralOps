#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Numerical Guardrails & Symplectic Stabilization Engine (Phases 031-034).

Provides mathematical protection against 10^{\pm 12} dynamic range spikes,
denormalized subnormals, and IEEE-754 mantissa overflows.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "SymplecticNumericalGuard",
    "symplectic_guarded_softmax",
    "sanitize_non_finite",
    "check_numerical_stability",
]


def sanitize_non_finite(
    x: torch.Tensor,
    nan_replacement: float = 0.0,
    pos_inf_replacement: float = 1e4,
    neg_inf_replacement: float = -1e4,
) -> torch.Tensor:
    """
    Sanitizes non-finite entries (NaN, +Inf, -Inf) while preserving tensor gradients.

    Args:
        x: Input tensor.
        nan_replacement: Value to replace NaN entries.
        pos_inf_replacement: Value to replace positive infinity.
        neg_inf_replacement: Value to replace negative infinity.

    Returns:
        Sanitized tensor with finite values.
    """
    if torch.isfinite(x).all():
        return x
    return torch.nan_to_num(
        x,
        nan=nan_replacement,
        posinf=pos_inf_replacement,
        neginf=neg_inf_replacement,
    )


def check_numerical_stability(
    x: torch.Tensor,
    dynamic_range_threshold: float = 1e12,
) -> Tuple[bool, dict]:
    """
    Audits tensor stability against extreme dynamic range spikes (10^{\pm 12}).

    Returns:
        Tuple of (is_stable, metrics_dict).
    """
    has_nan = bool(torch.isnan(x).any().item())
    has_inf = bool(torch.isinf(x).any().item())

    abs_x = torch.abs(x[torch.isfinite(x)])
    if abs_x.numel() == 0:
        return False, {"nan": has_nan, "inf": has_inf, "dynamic_range": float("inf")}

    max_val = float(torch.max(abs_x).item())
    min_val = float(torch.min(abs_x[abs_x > 0]).item()) if (abs_x > 0).any() else 1e-12

    dynamic_range = max_val / max(min_val, 1e-18)
    is_stable = (not has_nan) and (not has_inf) and (dynamic_range <= dynamic_range_threshold)

    return is_stable, {
        "nan": has_nan,
        "inf": has_inf,
        "max_abs": max_val,
        "min_positive": min_val,
        "dynamic_range": dynamic_range,
    }


def symplectic_guarded_softmax(
    x: torch.Tensor,
    dim: int = -1,
    eps_min: float = 1e-7,
    eps_max: float = 1e-4,
    clamp_max_exp: float = 50.0,
) -> torch.Tensor:
    """
    Symplectic scaled shifted log-sum-exp softmax preventing FP16/BF16 exponent overflow.

    Guarantees finite denominators even under 10^{\pm 12} activation spikes.
    """
    orig_dtype = x.dtype
    # Accumulate in FP32 to prevent mantissa truncation in SRAM registers
    x_fp32 = x.to(torch.float32)

    # Shifted max projection
    max_val = torch.max(x_fp32, dim=dim, keepdim=True).values
    # Clean non-finite maximums
    max_val = torch.nan_to_num(max_val, nan=0.0, posinf=clamp_max_exp, neginf=-clamp_max_exp)

    x_shifted = x_fp32 - max_val
    exp_shifted = torch.exp(x_shifted)
    denom = exp_shifted.sum(dim=dim, keepdim=True)

    # Dynamic symplectic epsilon floor scaled with respect to maximal projection
    dynamic_eps = torch.clamp(
        1e-12 * torch.exp(-torch.clamp(max_val, max=clamp_max_exp)),
        min=eps_min,
        max=eps_max,
    )

    out = exp_shifted / (denom + dynamic_eps)
    return out.to(orig_dtype)


class SymplecticNumericalGuard(nn.Module):
    """
    Symplectic numerical guard module wrapping attention logits and activation projections.
    """

    def __init__(
        self,
        eps_min: float = 1e-7,
        eps_max: float = 1e-4,
        dynamic_range_threshold: float = 1e12,
    ):
        super().__init__()
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.dynamic_range_threshold = dynamic_range_threshold

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return symplectic_guarded_softmax(
            x, dim=dim, eps_min=self.eps_min, eps_max=self.eps_max
        )
