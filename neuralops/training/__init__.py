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

try:
    from neuralops.training.layer_distill import LayerAttentionDistillLoss, LayerDistillScheduler
    _has_distill = True
except ImportError:
    _has_distill = False

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
]

if _has_distill:
    __all__.extend(["LayerAttentionDistillLoss", "LayerDistillScheduler"])
