"""Optimizers package for NeuralOps."""

from __future__ import annotations

from neuralops.trainer.optimizers.factory import (
    OPTIMIZER_REGISTRY,
    create_adabelief,
    create_adafactor,
    create_adam,
    create_adamw,
    create_adamw_8bit,
    create_lamb,
    create_lars,
    create_lion,
    create_lion_8bit,
    create_muon,
    create_muon_adamw_reg,
    create_ranger,
    create_riemannian_stiefel,
    create_sgd,
    create_sophia,
)
from neuralops.trainer.optimizers.modern import (
    AdaFactor,
    Lion,
    SophiaG,
    create_8bit_optimizer_if_available,
    get_optimizer_memory_footprint,
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
StiefelOptimizer = RiemannianStiefelOptimizer
RiemannianAdam = RiemannianStiefelOptimizer
from neuralops.trainer.optimizers.stiefel import (
    is_stiefel_parameter,
    project_stiefel_gradient,
    stiefel_qr_retraction,
)

__all__ = [
    "OPTIMIZER_REGISTRY",
    "RiemannianStiefelOptimizer",
    "StiefelOptimizer",
    "RiemannianAdam",
    "Muon",
    "MuonAdamW",
    "create_muon_adamw_optimizer",
    "zeropower_via_newtonschulz5",
    "project_stiefel_gradient",
    "stiefel_qr_retraction",
    "is_stiefel_parameter",
    "Lion",
    "SophiaG",
    "AdaFactor",
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
    "create_lion",
    "create_adafactor",
    "create_sophia",
    "create_adamw_8bit",
    "create_lion_8bit",
    "get_optimizer_memory_footprint",
    "create_8bit_optimizer_if_available",
]
