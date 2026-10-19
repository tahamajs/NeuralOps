#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Lion Optimizer.

EvoLved Sign Momentum (Lion) optimizer implementation for high-throughput NeuralOps training.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """Lion (EvoLved Sign Momentum) optimizer.

    Formulated using sign operations for uniform gradient scaling and accelerated convergence.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate (default: 1e-4).
        betas: Coefficients for computing running gradient moments (default: (0.9, 0.99)).
        weight_decay: Weight decay coefficient (default: 0.0).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
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
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Step 1: update = sign(beta1 * exp_avg + (1 - beta1) * grad)
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1).sign()
                p.data.add_(update, alpha=-lr)

                # Step 2: exp_avg = beta2 * exp_avg + (1 - beta2) * grad
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


def create_lion_optimizer(
    params: Any,
    lr: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.99),
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> Lion:
    """Build and return Lion optimizer instance."""
    return Lion(params, lr=lr, betas=betas, weight_decay=weight_decay, **kwargs)
