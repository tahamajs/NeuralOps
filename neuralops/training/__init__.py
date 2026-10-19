#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Author: Taha Majlesi


NeuralOps Training Subpackage.

Provides training orchestration, foundation pretraining engines, optimizers,
schedulers (including WSD), multi-dataset streaming, and document packing loaders.
"""

from __future__ import annotations

import sys
from neuralops.training.checkpointing import checkpoint_selective, SelectiveCheckpointModule
from neuralops.trainer.base_trainer import BaseTrainer
from neuralops.trainer.custom_loop_trainer import CustomLoopTrainer
from neuralops.trainer.hf_trainer import HuggingFaceTrainer
from neuralops.trainer.registry import TrainerRegistry
from neuralops.trainer.trainer import SVDAwareTrainer
from neuralops.trainer.utils import train_model
from neuralops.trainer.schedulers import WSDScheduler
from neuralops.data.multidataset import MultiDatasetLoader
from neuralops.data.packing import PackedDataset
from neuralops.benchmarks.core.training import (
    finetune_classification,
    train_kd,
    recovery_finetune,
    train_with_training_type,
)

import neuralops.trainer.trainer as trainer
sys.modules['neuralops.training.trainer'] = trainer

from neuralops.trainer.optimizers import (
    Muon,
    MuonAdamW,
    RiemannianStiefelOptimizer,
    create_muon,
    create_muon_adamw_optimizer,
    create_riemannian_stiefel,
    create_adamw,
    create_sgd,
)

try:
    from neuralops.training.layer_distill import LayerAttentionDistillLoss, LayerDistillScheduler
    _has_distill = True
except ImportError:
    _has_distill = False

from neuralops.core.triton_kernels.fused_ce import (
    FusedLinearCrossEntropy,
    fused_linear_cross_entropy,
)
from neuralops.core.triton_kernels.fused_kd import (
    FusedLinearKLDivergence,
    fused_linear_kl_divergence,
)

__all__ = [
    "checkpoint_selective",
    "SelectiveCheckpointModule",
    "BaseTrainer",
    "CustomLoopTrainer",
    "HuggingFaceTrainer",
    "TrainerRegistry",
    "SVDAwareTrainer",
    "train_model",
    "finetune_classification",
    "train_kd",
    "recovery_finetune",
    "train_with_training_type",
    "Muon",
    "MuonAdamW",
    "RiemannianStiefelOptimizer",
    "create_muon",
    "create_muon_adamw_optimizer",
    "create_riemannian_stiefel",
    "create_adamw",
    "create_sgd",
    "FusedLinearCrossEntropy",
    "fused_linear_cross_entropy",
    "FusedLinearKLDivergence",
    "fused_linear_kl_divergence",
    "MultiDatasetLoader",
    "PackedDataset",
    "WSDScheduler",
]

if _has_distill:
    __all__.extend(["LayerAttentionDistillLoss", "LayerDistillScheduler"])
