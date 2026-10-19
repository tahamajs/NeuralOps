#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Trainer Optimizers Subpackage.
"""

from __future__ import annotations

from neuralops.trainer.optimizers.factory import (
    OPTIMIZER_REGISTRY,
    create_adabelief,
    create_adam,
    create_adamw,
    create_lamb,
    create_lars,
    create_lion,
    create_muon,
    create_muon_adamw_reg,
    create_ranger,
    create_riemannian_stiefel,
    create_sgd,
)
from neuralops.trainer.optimizers.lion import (
    Lion,
    create_lion_optimizer,
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
    "Muon",
    "MuonAdamW",
    "create_muon_adamw_optimizer",
    "zeropower_via_newtonschulz5",
    "project_stiefel_gradient",
    "stiefel_qr_retraction",
    "is_stiefel_parameter",
    "Lion",
    "create_lion",
    "create_lion_optimizer",
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

from neuralops.trainer.optimizers import factory
from neuralops.trainer.optimizers import muon
from neuralops.trainer.optimizers import riemannian
from neuralops.trainer.optimizers import stiefel

from neuralops.trainer.optimizers import factory
from neuralops.trainer.optimizers import muon
from neuralops.trainer.optimizers import riemannian
from neuralops.trainer.optimizers import stiefel
