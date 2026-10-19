#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""


Learning Rate Schedulers.

Provides custom learning rate scheduling policies, Warmup-Stable-Decay (WSD),
and warmup decay routines for NeuralOps pretraining and fine-tuning pipelines.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.optim import lr_scheduler

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # type: ignore[no-redef]

from neuralops.core.registry import Registry

SCHEDULER_REGISTRY = Registry("scheduler")


class WSDScheduler(LRScheduler):
    """Warmup-Stable-Decay (WSD) Learning Rate Scheduler.

    Features three distinct pretraining phases:
      1. Warmup: Linear increase from 0 (or initial lr) to peak learning rate.
      2. Stable: Constant peak learning rate across main compute budget.
      3. Decay: Smooth cosine or linear decay down to min_lr / min_lr_ratio
         over the final decay_steps or decay_ratio portion of training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1000,
        stable_steps: Optional[int] = None,
        decay_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        num_training_steps: Optional[int] = None,
        num_warmup_steps: Optional[int] = None,
        decay_ratio: float = 0.1,
        min_lr: float = 0.0,
        min_lr_ratio: float = 0.0,
        decay_type: str = "cosine",
        last_epoch: int = -1,
        **kwargs: Any,
    ) -> None:
        self.warmup_steps = int(num_warmup_steps if num_warmup_steps is not None else warmup_steps)
        effective_total = total_steps if total_steps is not None else num_training_steps

        if decay_steps is not None:
            self.decay_steps = int(decay_steps)
        elif effective_total is not None:
            self.decay_steps = max(1, int(effective_total * decay_ratio))
        else:
            self.decay_steps = 1000

        if stable_steps is not None:
            self.stable_steps = int(stable_steps)
        elif effective_total is not None:
            self.stable_steps = max(0, effective_total - self.warmup_steps - self.decay_steps)
        else:
            self.stable_steps = 0

        self.total_steps = self.warmup_steps + self.stable_steps + self.decay_steps
        self.decay_ratio = float(decay_ratio)
        self.min_lr = float(min_lr)
        self.min_lr_ratio = float(min_lr_ratio)
        self.decay_type = str(decay_type).lower()

        super().__init__(optimizer, last_epoch=last_epoch)

    @property
    def current_phase(self) -> str:
        """Return the current scheduler phase ('warmup', 'stable', or 'decay')."""
        step = self.last_epoch
        if step < self.warmup_steps:
            return "warmup"
        if step < self.warmup_steps + self.stable_steps:
            return "stable"
        return "decay"

    def get_factor(self, step: Optional[int] = None) -> float:
        """Compute the learning rate scaling factor at a given step."""
        if step is None:
            step = self.last_epoch

        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))

        if step < self.warmup_steps + self.stable_steps:
            return 1.0

        decay_step = step - (self.warmup_steps + self.stable_steps)
        progress = float(decay_step) / float(max(1, self.decay_steps))
        progress = min(1.0, max(0.0, progress))

        if self.decay_type == "linear":
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * (1.0 - progress)

        # Smooth cosine decay
        return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    def get_lr(self) -> List[float]:
        """Compute scheduled learning rates for all parameter groups."""
        factor = self.get_factor()
        return [base_lr * factor for base_lr in self.base_lrs]


@SCHEDULER_REGISTRY.register("wsd")
def create_wsd(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 1000,
    stable_steps: Optional[int] = None,
    decay_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_warmup_steps: Optional[int] = None,
    decay_ratio: float = 0.1,
    min_lr: float = 0.0,
    min_lr_ratio: float = 0.0,
    decay_type: str = "cosine",
    **kwargs: Any,
) -> WSDScheduler:
    """Build and return Warmup-Stable-Decay (WSD) scheduler."""
    return WSDScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        total_steps=total_steps,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        decay_ratio=decay_ratio,
        min_lr=min_lr,
        min_lr_ratio=min_lr_ratio,
        decay_type=decay_type,
        **kwargs,
    )


@SCHEDULER_REGISTRY.register("linear")
def create_linear(optimizer: torch.optim.Optimizer, num_warmup_steps: int = 0, num_training_steps: int = 10000, **kwargs: Any) -> Any:
    """Build and return linear."""
    return lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps
    )


@SCHEDULER_REGISTRY.register("cosine")
def create_cosine(optimizer: torch.optim.Optimizer, T_max: int = 10000, **kwargs: Any) -> Any:
    """Build and return cosine."""
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)


@SCHEDULER_REGISTRY.register("reduce_on_plateau")
def create_rop(optimizer: torch.optim.Optimizer, factor: float = 0.1, patience: int = 10, **kwargs: Any) -> Any:
    """Build and return rop."""
    return lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)


@SCHEDULER_REGISTRY.register("one_cycle")
def create_onecycle(optimizer: torch.optim.Optimizer, max_lr: float = 0.01, steps_per_epoch: int = 1000, epochs: int = 10, **kwargs: Any) -> Any:
    """Build and return onecycle."""
    return lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs
    )


@SCHEDULER_REGISTRY.register("cosine_warmup")
def create_cosine_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int = 500, num_training_steps: int = 10000, **kwargs: Any
) -> Any:
    """Build and return cosine warmup."""

    def lr_lambda(current_step: int) -> float:
        """Lr lambda."""
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)


@SCHEDULER_REGISTRY.register("polynomial")
def create_poly(optimizer: torch.optim.Optimizer, total_iters: int = 10000, power: float = 1.0, **kwargs: Any) -> Any:
    """Build and return poly."""
    return lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=power)


get_cosine_schedule_with_warmup = create_cosine_warmup


def get_lr_scheduler(name: str, optimizer: torch.optim.Optimizer, **kwargs: Any) -> Any:
    """Retrieve and construct a learning rate scheduler by registered name."""
    fn = SCHEDULER_REGISTRY.get(name.lower())
    if fn is None:
        raise KeyError(
            f"Scheduler '{name}' not found in SCHEDULER_REGISTRY. Available: {list(SCHEDULER_REGISTRY.keys())}"
        )
    return fn(optimizer, **kwargs)


def get_scheduler(
    name: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs: Any,
) -> Any:
    """Factory function for learning rate schedulers supporting decay_type and CLI flags."""
    decay_type_arg = kwargs.pop("decay_type", None)
    sched_name = name or decay_type_arg or "linear"
    if str(sched_name).lower() == "wsd" or str(decay_type_arg).lower() == "wsd":
        if optimizer is None:
            raise ValueError("Optimizer must be specified to construct WSDScheduler.")
        return create_wsd(optimizer=optimizer, **kwargs)

    if optimizer is None:
        raise ValueError("Optimizer must be specified to construct a learning rate scheduler.")
    return get_lr_scheduler(str(sched_name), optimizer=optimizer, **kwargs)


__all__ = [
    "WSDScheduler",
    "create_wsd",
    "create_linear",
    "create_cosine",
    "create_cosine_warmup",
    "create_onecycle",
    "create_poly",
    "create_rop",
    "get_lr_scheduler",
    "get_scheduler",
    "get_cosine_schedule_with_warmup",
    "SCHEDULER_REGISTRY",
]
