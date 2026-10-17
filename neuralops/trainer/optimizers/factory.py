"""Optimizer factories and registry bindings."""

from __future__ import annotations

from typing import Any

import torch.optim as optim

from neuralops.core.registry import Registry
from neuralops.trainer.optimizers.muon import (
    Muon,
    create_muon_adamw_optimizer,
)
from neuralops.trainer.optimizers.riemannian import (
    RiemannianStiefelOptimizer,
)

OPTIMIZER_REGISTRY = Registry("optimizer")


@OPTIMIZER_REGISTRY.register("muon")
def create_muon(
    params: Any,
    lr: float = 0.02,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> Any:
    """Build and return Muon optimizer."""
    return Muon(params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)


@OPTIMIZER_REGISTRY.register("muon_adamw")
def create_muon_adamw_reg(
    params: Any,
    lr: float = 0.02,
    lr_adam: float = 5e-5,
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> Any:
    """Build and return MuonAdamW hybrid optimizer."""
    return create_muon_adamw_optimizer(
        params, lr_muon=lr, lr_adam=lr_adam, weight_decay=weight_decay, **kwargs
    )


@OPTIMIZER_REGISTRY.register("riemannian_stiefel")
@OPTIMIZER_REGISTRY.register("stiefel")
@OPTIMIZER_REGISTRY.register("stiefel_optimizer")
def create_riemannian_stiefel(
    params: Any,
    lr: float = 1e-3,
    lr_stiefel: float | None = None,
    base_optimizer: str = "adamw",
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> RiemannianStiefelOptimizer:
    """Build and return RiemannianStiefelOptimizer."""
    return RiemannianStiefelOptimizer(
        params,
        lr=lr,
        lr_stiefel=lr_stiefel,
        base_optimizer=base_optimizer,
        weight_decay=weight_decay,
        **kwargs,
    )


@OPTIMIZER_REGISTRY.register("adamw")
def create_adamw(params: Any, lr: float = 1e-3, weight_decay: float = 0.01, **kwargs: Any) -> Any:
    """Build and return AdamW."""
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZER_REGISTRY.register("sgd")
def create_sgd(
    params: Any,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Build and return SGD."""
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)


@OPTIMIZER_REGISTRY.register("adam")
def create_adam(params: Any, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs: Any) -> Any:
    """Build and return Adam."""
    return optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZER_REGISTRY.register("adabelief")
def create_adabelief(
    params: Any, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs: Any
) -> Any:
    """Build and return AdaBelief."""
    from adabelief_pytorch import AdaBelief

    return AdaBelief(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZER_REGISTRY.register("ranger")
def create_ranger(params: Any, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs: Any) -> Any:
    """Build and return Ranger."""
    from ranger import Ranger

    return Ranger(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZER_REGISTRY.register("lamb")
def create_lamb(params: Any, lr: float = 1e-3, weight_decay: float = 0.01, **kwargs: Any) -> Any:
    """Build and return Lamb."""
    from torch_optimizer import Lamb

    return Lamb(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZER_REGISTRY.register("lars")
def create_lars(params: Any, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs: Any) -> Any:
    """Build and return Lars."""
    from torch_optimizer import Lars

    return Lars(params, lr=lr, weight_decay=weight_decay, **kwargs)


__all__ = [
    "OPTIMIZER_REGISTRY",
    "create_muon",
    "create_muon_adamw_reg",
    "create_riemannian_stiefel",
    "create_adamw",
    "create_sgd",
    "create_adam",
    "create_adabelief",
    "create_ranger",
    "create_lamb",
    "create_lars",
]
