"""Training package initialization with shims for trainer and distributed execution."""
from __future__ import annotations

import sys
from neuralops.training.checkpointing import checkpoint_selective, SelectiveCheckpointModule
from neuralops.trainer.base_trainer import BaseTrainer
from neuralops.trainer.custom_loop_trainer import CustomLoopTrainer
from neuralops.trainer.hf_trainer import HuggingFaceTrainer
from neuralops.trainer.registry import TrainerRegistry
from neuralops.trainer.trainer import SVDAwareTrainer
from neuralops.trainer.utils import train_model
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
]

if _has_distill:
    __all__.extend(["LayerAttentionDistillLoss", "LayerDistillScheduler"])
