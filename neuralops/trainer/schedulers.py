import math
from typing import Any

from torch.optim import lr_scheduler

from neuralops.core.registry import Registry

SCHEDULER_REGISTRY = Registry("scheduler")


@SCHEDULER_REGISTRY.register("linear")
def create_linear(optimizer, num_warmup_steps=0, num_training_steps=10000, **kwargs) -> Any:
    """Build and return linear."""
    return lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps
    )


@SCHEDULER_REGISTRY.register("cosine")
def create_cosine(optimizer, T_max=10000, **kwargs) -> Any:
    """Build and return cosine."""
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)


@SCHEDULER_REGISTRY.register("reduce_on_plateau")
def create_rop(optimizer, factor=0.1, patience=10, **kwargs) -> Any:
    """Build and return rop."""
    return lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)


@SCHEDULER_REGISTRY.register("one_cycle")
def create_onecycle(optimizer, max_lr=0.01, steps_per_epoch=1000, epochs=10, **kwargs) -> Any:
    """Build and return onecycle."""
    return lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs
    )


@SCHEDULER_REGISTRY.register("cosine_warmup")
def create_cosine_warmup(
    optimizer, num_warmup_steps=500, num_training_steps=10000, **kwargs
) -> Any:
    """Build and return cosine warmup."""

    def lr_lambda(current_step: int) -> float:
        """Lr lambda."""
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # FIX: Use math.cos and math.pi instead of torch.cos on Python float to avoid TypeError
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)


@SCHEDULER_REGISTRY.register("polynomial")
def create_poly(optimizer, total_iters=10000, power=1.0, **kwargs) -> Any:
    """Build and return poly."""
    return lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=power)


@SCHEDULER_REGISTRY.register("linear_warmup")
def create_linear_warmup(
    optimizer, num_warmup_steps=500, num_training_steps=10000, **kwargs
) -> Any:
    """Build linear schedule with warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return lr_scheduler.LambdaLR(optimizer, lr_lambda)


get_linear_schedule_with_warmup = create_linear_warmup


@SCHEDULER_REGISTRY.register("wsd")
def create_wsd(
    optimizer, num_warmup_steps=500, num_stable_steps=8000, num_decay_steps=1500, **kwargs
) -> Any:
    """Warmup-Stable-Decay (WSD) learning rate schedule (MiniMax / DeepSeek / Llama 3 style)."""
    total_steps = num_warmup_steps + num_stable_steps + num_decay_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + num_stable_steps:
            return 1.0
        else:
            decay_step = current_step - (num_warmup_steps + num_stable_steps)
            progress = float(decay_step) / float(max(1, num_decay_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)


get_wsd_schedule = create_wsd


# FIX: Public alias for standard cosine warmup scheduler
get_cosine_schedule_with_warmup = create_cosine_warmup


def get_lr_scheduler(name: str, optimizer, **kwargs) -> Any:
    """Retrieve and construct a learning rate scheduler by registered name."""
    fn = SCHEDULER_REGISTRY.get(name)
    if fn is None:
        raise KeyError(f"Scheduler '{name}' not found in SCHEDULER_REGISTRY. Available: {list(SCHEDULER_REGISTRY.keys())}")
    return fn(optimizer, **kwargs)


__all__ = [
    "SCHEDULER_REGISTRY",
    "create_linear",
    "create_cosine",
    "create_rop",
    "create_onecycle",
    "create_cosine_warmup",
    "create_linear_warmup",
    "create_poly",
    "create_wsd",
    "get_cosine_schedule_with_warmup",
    "get_linear_schedule_with_warmup",
    "get_wsd_schedule",
    "get_lr_scheduler",
]
