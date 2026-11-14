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
        """Initialize WSDScheduler.

        Args:
            optimizer: Wrapped PyTorch optimizer.
            warmup_steps: Number of initial linear warmup iterations.
            stable_steps: Number of constant peak learning rate iterations.
            decay_steps: Number of final decay iterations.
            total_steps: Total training steps across all 3 phases.
            num_training_steps: Alias for total_steps.
            num_warmup_steps: Alias for warmup_steps.
            decay_ratio: Fraction of total steps used for decay when decay_steps is None.
            min_lr: Minimum target learning rate at end of decay.
            min_lr_ratio: Ratio of peak learning rate to preserve at end of decay.
            decay_type: Decay profile ('cosine', 'linear', or 'exponential').
            last_epoch: The index of last epoch. Default: -1.
            **kwargs: Extra unused arguments.
        """
        self.warmup_steps = int(num_warmup_steps if num_warmup_steps is not None else warmup_steps)
        effective_total = total_steps if total_steps is not None else num_training_steps

        if effective_total is not None:
            effective_total = int(effective_total)
            if decay_steps is not None and stable_steps is not None:
                self.decay_steps = int(decay_steps)
                self.stable_steps = int(stable_steps)
            elif stable_steps is not None:
                self.stable_steps = int(stable_steps)
                self.decay_steps = max(1, effective_total - self.warmup_steps - self.stable_steps)
            elif decay_steps is not None:
                self.decay_steps = int(decay_steps)
                self.stable_steps = max(0, effective_total - self.warmup_steps - self.decay_steps)
            else:
                self.decay_steps = max(1, int(effective_total * decay_ratio))
                self.stable_steps = max(0, effective_total - self.warmup_steps - self.decay_steps)
            self.total_steps = effective_total
        else:
            self.decay_steps = int(decay_steps) if decay_steps is not None else 1000
            self.stable_steps = int(stable_steps) if stable_steps is not None else 0
            self.total_steps = self.warmup_steps + self.stable_steps + self.decay_steps

        self.decay_ratio = float(decay_ratio)
        self.min_lr = float(min_lr)
        self.min_lr_ratio = float(min_lr_ratio)
        self.decay_type = str(decay_type).lower()

        super().__init__(optimizer, last_epoch=last_epoch)

    @property
    def current_phase(self) -> str:
        """Return the current scheduler phase ('warmup', 'stable', or 'decay')."""
        step = max(0, self.last_epoch)
        if step < self.warmup_steps:
            return "warmup"
        if step < self.warmup_steps + self.stable_steps:
            return "stable"
        return "decay"

    def get_factor(self, step: Optional[int] = None) -> float:
        """Compute the learning rate scaling factor at a given step.

        Args:
            step: Step index to query (defaults to self.last_epoch).

        Returns:
            Multiplicative factor in [min_ratio, 1.0].
        """
        if step is None:
            step = self.last_epoch

        step = max(0, step)

        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))

        if step < self.warmup_steps + self.stable_steps:
            return 1.0

        decay_step = step - (self.warmup_steps + self.stable_steps)
        progress = float(decay_step) / float(max(1, self.decay_steps))
        progress = min(1.0, max(0.0, progress))

        min_ratio = self.min_lr_ratio
        if min_ratio == 0.0 and self.min_lr > 0.0 and hasattr(self, "base_lrs") and self.base_lrs:
            max_base = max(self.base_lrs)
            if max_base > 0:
                min_ratio = self.min_lr / max_base

        if self.decay_type == "linear":
            return min_ratio + (1.0 - min_ratio) * (1.0 - progress)

        if self.decay_type in ("exp", "exponential"):
            return min_ratio + (1.0 - min_ratio) * math.exp(-5.0 * progress)

        # Smooth cosine decay
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    def get_lr(self) -> List[float]:
        """Compute scheduled learning rates for all parameter groups.

        Returns:
            List of learning rates corresponding to optimizer parameter groups.
        """
        factor = self.get_factor()
        lrs: List[float] = []
        for base_lr in self.base_lrs:
            lr = base_lr * factor
            if self.min_lr > 0.0 and self.last_epoch >= self.warmup_steps and base_lr > 0.0:
                lr = max(self.min_lr, lr)
            lrs.append(lr)
        return lrs


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
    """Build and return Warmup-Stable-Decay (WSD) scheduler.

    Args:
        optimizer: Wrapped PyTorch optimizer.
        warmup_steps: Initial linear warmup steps.
        stable_steps: Steps spent at constant peak lr.
        decay_steps: Steps spent decaying to min lr.
        total_steps: Total training steps.
        num_training_steps: Total training steps alias.
        num_warmup_steps: Warmup steps alias.
        decay_ratio: Decay fraction when decay_steps is None.
        min_lr: Minimum target lr.
        min_lr_ratio: Minimum lr fraction.
        decay_type: Decay profile ('cosine', 'linear', or 'exponential').
        **kwargs: Extra unused arguments.

    Returns:
        WSDScheduler instance.
    """
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
def create_linear(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int = 0,
    num_training_steps: int = 10000,
    **kwargs: Any,
) -> Any:
    """Build and return linear learning rate schedule with optional warmup.

    Args:
        optimizer: Wrapped optimizer.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        **kwargs: Extra arguments (supports warmup_steps, total_steps).

    Returns:
        LRScheduler instance.
    """
    warmup = kwargs.pop("warmup_steps", num_warmup_steps)
    total = kwargs.pop("total_steps", num_training_steps)
    warmup = int(warmup)
    total = int(total)

    if warmup > 0:

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            progress = float(current_step - warmup) / float(max(1, total - warmup))
            return max(0.0, 1.0 - progress)

        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total)


get_linear_schedule_with_warmup = create_linear


@SCHEDULER_REGISTRY.register("cosine")
def create_cosine(
    optimizer: torch.optim.Optimizer,
    T_max: Optional[int] = None,
    eta_min: float = 0.0,
    last_epoch: int = -1,
    **kwargs: Any,
) -> lr_scheduler.CosineAnnealingLR:
    """Build and return PyTorch CosineAnnealingLR matching PyTorch exactly.

    Args:
        optimizer: Wrapped optimizer.
        T_max: Maximum number of iterations.
        eta_min: Minimum learning rate. Default: 0.0.
        last_epoch: The index of last epoch. Default: -1.
        **kwargs: Extra arguments (supports num_training_steps, total_steps, min_lr).

    Returns:
        torch.optim.lr_scheduler.CosineAnnealingLR instance.
    """
    if T_max is None:
        T_max = kwargs.pop("num_training_steps", None) or kwargs.pop("total_steps", None) or 10000
    if "min_lr" in kwargs and eta_min == 0.0:
        eta_min = kwargs.pop("min_lr")
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(T_max), eta_min=float(eta_min), last_epoch=last_epoch)


@SCHEDULER_REGISTRY.register("reduce_on_plateau")
@SCHEDULER_REGISTRY.register("rop")
def create_rop(
    optimizer: torch.optim.Optimizer,
    factor: float = 0.1,
    patience: int = 10,
    **kwargs: Any,
) -> lr_scheduler.ReduceLROnPlateau:
    """Build and return ReduceLROnPlateau scheduler.

    Args:
        optimizer: Wrapped optimizer.
        factor: Factor by which the learning rate will be reduced. Default: 0.1.
        patience: Number of epochs with no improvement after which learning rate is reduced.
        **kwargs: Extra arguments forwarded to ReduceLROnPlateau.

    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau instance.
    """
    return lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, **kwargs)


@SCHEDULER_REGISTRY.register("one_cycle")
@SCHEDULER_REGISTRY.register("onecycle")
def create_onecycle(
    optimizer: torch.optim.Optimizer,
    max_lr: float = 0.01,
    steps_per_epoch: int = 1000,
    epochs: int = 10,
    **kwargs: Any,
) -> lr_scheduler.OneCycleLR:
    """Build and return OneCycleLR scheduler.

    Args:
        optimizer: Wrapped optimizer.
        max_lr: Upper learning rate boundaries in the cycle.
        steps_per_epoch: The number of steps per epoch to train for.
        epochs: The number of epochs to train for.
        **kwargs: Extra arguments forwarded to OneCycleLR.

    Returns:
        torch.optim.lr_scheduler.OneCycleLR instance.
    """
    return lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs, **kwargs)


@SCHEDULER_REGISTRY.register("cosine_warmup")
@SCHEDULER_REGISTRY.register("cosine_with_warmup")
def create_cosine_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int = 500,
    num_training_steps: int = 10000,
    **kwargs: Any,
) -> lr_scheduler.LambdaLR:
    """Build and return cosine schedule with warmup.

    Args:
        optimizer: Wrapped optimizer.
        num_warmup_steps: Warmup step count.
        num_training_steps: Total training step count.
        **kwargs: Extra arguments (supports warmup_steps, total_steps).

    Returns:
        torch.optim.lr_scheduler.LambdaLR instance.
    """
    warmup = kwargs.pop("warmup_steps", num_warmup_steps)
    total = kwargs.pop("total_steps", num_training_steps)
    warmup = int(warmup)
    total = int(total)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        progress = float(current_step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)


get_cosine_schedule_with_warmup = create_cosine_warmup


@SCHEDULER_REGISTRY.register("polynomial")
@SCHEDULER_REGISTRY.register("poly")
def create_poly(
    optimizer: torch.optim.Optimizer,
    total_iters: int = 10000,
    power: float = 1.0,
    **kwargs: Any,
) -> lr_scheduler.PolynomialLR:
    """Build and return PolynomialLR scheduler.

    Args:
        optimizer: Wrapped optimizer.
        total_iters: Number of iterations to decay learning rate over.
        power: The power of the polynomial.
        **kwargs: Extra arguments forwarded to PolynomialLR.

    Returns:
        torch.optim.lr_scheduler.PolynomialLR instance.
    """
    total = kwargs.pop("total_steps", total_iters)
    return lr_scheduler.PolynomialLR(optimizer, total_iters=int(total), power=power, **kwargs)


def get_lr_scheduler(name: str, optimizer: torch.optim.Optimizer, **kwargs: Any) -> Any:
    """Retrieve and construct a learning rate scheduler by registered name.

    Args:
        name: Registered scheduler identifier.
        optimizer: Wrapped PyTorch optimizer.
        **kwargs: Constructor arguments for the target scheduler.

    Returns:
        Constructed learning rate scheduler.

    Raises:
        KeyError: If scheduler name is not registered.
    """
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
    """Factory function for learning rate schedulers supporting decay_type and CLI flags.

    Args:
        name: Optional scheduler name.
        optimizer: Wrapped PyTorch optimizer.
        **kwargs: Scheduler parameters (including decay_type, total_steps, etc.).

    Returns:
        Constructed learning rate scheduler.

    Raises:
        ValueError: If optimizer is None.
    """
    decay_type_arg = kwargs.get("decay_type", None)
    sched_name = name or decay_type_arg or "linear"
    sched_name_str = str(sched_name).lower()

    if optimizer is None:
        raise ValueError(f"Optimizer must be specified to construct '{sched_name_str}' scheduler.")

    if sched_name_str == "wsd":
        if decay_type_arg and str(decay_type_arg).lower() == "wsd":
            kwargs["decay_type"] = "cosine"
        return create_wsd(optimizer=optimizer, **kwargs)

    kwargs.pop("decay_type", None)
    return get_lr_scheduler(sched_name_str, optimizer=optimizer, **kwargs)


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
    "get_linear_schedule_with_warmup",
    "SCHEDULER_REGISTRY",
]
