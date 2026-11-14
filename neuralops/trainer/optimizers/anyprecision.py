#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
AnyPrecisionAdamW Optimizer

Stores first moment, second moment, and parameter updates in arbitrary target
precisions (float32, bfloat16, float16) with optional Kahan summation compensation,
drastically reducing optimizer state VRAM footprints by 50-75%.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


class AnyPrecisionAdamW(Optimizer):
    """AdamW optimizer with customizable precision for momentum and variance states.

    Allows memory-efficient fine-tuning and pretraining by casting optimizer states
    (exp_avg, exp_avg_sq) to reduced precision formats while maintaining numerical stability
    via optional Kahan summation error compensation.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running gradient moments (default: (0.9, 0.999)).
        eps: Term added to the denominator for numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 1e-2).
        use_kahan_summation: Whether to track compensation buffers for low-precision updates.
        momentum_dtype: Target dtype for the first moment exp_avg (default: torch.float32).
        variance_dtype: Target dtype for the second moment exp_avg_sq (default: torch.bfloat16).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        use_kahan_summation: bool = False,
        momentum_dtype: torch.dtype = torch.float32,
        variance_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_kahan_summation=use_kahan_summation,
            momentum_dtype=momentum_dtype,
            variance_dtype=variance_dtype,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            use_kahan = group["use_kahan_summation"]
            mom_dtype = group["momentum_dtype"]
            var_dtype = group["variance_dtype"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AnyPrecisionAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=mom_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=var_dtype)
                    if use_kahan:
                        state["compensation"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                step_t = state["step"]

                # Apply weight decay
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Compute updates in float32 for stability, then store in target dtypes
                g_f32 = grad.float()
                m_f32 = exp_avg.float().mul(beta1).add(g_f32, alpha=1.0 - beta1)
                v_f32 = exp_avg_sq.float().mul(beta2).addcmul(g_f32, g_f32, value=1.0 - beta2)

                state["exp_avg"] = m_f32.to(dtype=mom_dtype)
                state["exp_avg_sq"] = v_f32.clamp_min(0.0).to(dtype=var_dtype)

                # Bias correction terms
                bias_correction1 = 1.0 - beta1**step_t
                bias_correction2 = 1.0 - beta2**step_t

                denom = (v_f32.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                update = (m_f32 / denom) * step_size

                if use_kahan and p.dtype != torch.float32:
                    comp = state["compensation"]
                    y = update - comp
                    t = p.float() - y
                    state["compensation"] = (t - p.float()) + y
                    p.copy_(t.to(p.dtype))
                else:
                    p.data.add_(update.to(p.dtype), alpha=-1.0)

        return loss


def create_anyprecision_optimizer(
    params: Any,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    use_kahan_summation: bool = False,
    momentum_dtype: torch.dtype = torch.float32,
    variance_dtype: torch.dtype = torch.bfloat16,
    **kwargs: Any,
) -> AnyPrecisionAdamW:
    """Build and return AnyPrecisionAdamW optimizer."""
    return AnyPrecisionAdamW(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        use_kahan_summation=use_kahan_summation,
        momentum_dtype=momentum_dtype,
        variance_dtype=variance_dtype,
        **kwargs,
    )


__all__ = [
    "AnyPrecisionAdamW",
    "create_anyprecision_optimizer",
]
