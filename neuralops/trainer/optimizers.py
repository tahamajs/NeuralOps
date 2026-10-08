"""Backward-compatibility shim. Logic moved to neuralops.trainer.optimizers package."""

from __future__ import annotations

from neuralops.trainer.optimizers import (
    OPTIMIZER_REGISTRY,
    Muon,
    MuonAdamW,
    RiemannianStiefelOptimizer,
    create_adabelief,
    create_adam,
    create_adamw,
    create_lamb,
    create_lars,
    create_muon,
    create_muon_adamw_optimizer,
    create_muon_adamw_reg,
    create_ranger,
    create_riemannian_stiefel,
    create_sgd,
    is_stiefel_parameter,
    project_stiefel_gradient,
    stiefel_qr_retraction,
    zeropower_via_newtonschulz5,
)

__all__ = [
    "OPTIMIZER_REGISTRY",
    "RiemannianStiefelOptimizer",
    "Muon",
    "MuonAdamW",
    "create_muon_adamw_optimizer",
    "zeropower_via_newtonschulz5",
    "project_stiefel_gradient",
    "stiefel_qr_retraction",
    "is_stiefel_parameter",
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
