#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Advanced Optimization and Manifold Algorithms Facade.

Canonical Owner: neuralops.trainer.optimizers
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from neuralops.trainer.optimizers.factory import OPTIMIZER_REGISTRY
from neuralops.trainer.optimizers.lion import Lion, create_lion_optimizer
from neuralops.trainer.optimizers.muon import Muon, MuonAdamW, create_muon_adamw_optimizer
from neuralops.trainer.optimizers.riemannian import RiemannianStiefelOptimizer
from . import kernels


def get_optimizer(
    model_or_params: Any,
    name: str = "muon",
    lr: float = 0.02,
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> Any:
    """Builds and configures an advanced optimizer for post-surgery healing or training.

    Canonical implementation forwarded to neuralops.trainer.optimizers.
    """
    params = model_or_params.parameters() if isinstance(model_or_params, nn.Module) else model_or_params

    clean_name = name.lower().strip()
    if clean_name not in OPTIMIZER_REGISTRY:
        supported = list(OPTIMIZER_REGISTRY.keys())
        raise ValueError(f"Unknown optimizer '{name}'. Supported optimizers: {supported}")

    builder = OPTIMIZER_REGISTRY[clean_name]
    return builder(params, lr=lr, weight_decay=weight_decay, **kwargs)


__all__ = [
    "Lion",
    "Muon",
    "MuonAdamW",
    "RiemannianStiefelOptimizer",
    "create_lion_optimizer",
    "create_muon_adamw_optimizer",
    "get_optimizer",
    "OPTIMIZER_REGISTRY",
    "kernels",
]
