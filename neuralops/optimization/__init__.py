#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Advanced Optimization and Manifold Algorithms (Muon, Stiefel, Lion, WSD).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from neuralops.trainer.optimizers.factory import OPTIMIZER_REGISTRY
from neuralops.trainer.optimizers.lion import Lion, create_lion_optimizer
from neuralops.trainer.optimizers.muon import (
    Muon,
    MuonAdamW,
    create_muon_adamw_optimizer,
)
from neuralops.trainer.optimizers.riemannian import (
    RiemannianStiefelOptimizer,
)


def get_optimizer(
    model_or_params: Any,
    name: str = "muon",
    lr: float = 0.02,
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> Any:
    """Builds and configures an advanced optimizer for post-surgery healing or training.

    Parameters:
        model_or_params: PyTorch nn.Module or iterable of parameters.
        name: Name of optimizer ('muon', 'muon_adamw', 'stiefel', 'lion', 'adamw').
        lr: Base learning rate.
        weight_decay: Weight decay coefficient.
        **kwargs: Additional parameters forwarded to the specific optimizer constructor.

    Returns:
        Configured torch.optim.Optimizer instance.
    """
    params = (
        model_or_params.parameters()
        if isinstance(model_or_params, nn.Module)
        else model_or_params
    )

    clean_name = name.lower().strip()
    if clean_name not in OPTIMIZER_REGISTRY:
        supported = list(OPTIMIZER_REGISTRY.keys())
        raise ValueError(
            f"Unknown optimizer '{name}'. Supported optimizers: {supported}"
        )

    builder = OPTIMIZER_REGISTRY[clean_name]
    return builder(params, lr=lr, weight_decay=weight_decay, **kwargs)


__all__ = [
    "Muon",
    "MuonAdamW",
    "RiemannianStiefelOptimizer",
    "Lion",
    "OPTIMIZER_REGISTRY",
    "get_optimizer",
    "create_lion_optimizer",
    "create_muon_adamw_optimizer",
]
