"""Optimizers package for NeuralOps."""

from __future__ import annotations

from neuralops.trainer.optimizers.factory import (
    OPTIMIZER_REGISTRY,
    create_adabelief,
    create_adam,
    create_adamw,
    create_lamb,
    create_lars,
    create_muon,
    create_muon_adamw_reg,
    create_ranger,
    create_riemannian_stiefel,
    create_sgd,
)
from neuralops.trainer.optimizers.muon import (
    Muon,
    MuonAdamW,
    create_muon_adamw_optimizer,
    zeropower_via_newtonschulz5,
)
from neuralops.trainer.optimizers.riemannian import (
    RiemannianStiefelOptimizer,
)
from neuralops.trainer.optimizers.stiefel import (
    is_stiefel_parameter,
    project_stiefel_gradient,
    stiefel_qr_retraction,
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
