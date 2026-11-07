#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Trainer Subpackage.
"""

from .base_trainer import BaseTrainer, Trainer
from .callbacks import GradientNormCallback, MemoryCallback, PreemptionCallback, SVDAttentionCallback, TimeCallback
from .custom_loop_trainer import CustomLoopTrainer

try:
    from .hf_trainer import HuggingFaceTrainer  # requires datasets + transformers.Trainer
except (ImportError, RuntimeError):
    HuggingFaceTrainer = None  # type: ignore[assignment,misc]
from .hyperopt import HyperparameterOptimizer, PrunedHyperparameterOptimizer, quick_optimize
from .metrics import (
    aggregate_results,
    compute_accuracy,
    compute_f1,
    compute_matthews_corrcoef,
    compute_pearson_spearman,
    get_task_metrics,
)
from .neuralops_trainer import NeuralOpsTrainer
from .optimizers import OPTIMIZER_REGISTRY, RiemannianStiefelOptimizer, project_stiefel_gradient, stiefel_qr_retraction
from .registry import TrainerRegistry
from .results_manager import ResultsManager
from .smart_gpu_trainer import AsyncDataPrefetcher, AutoMaxGPUTrainer, GPUMemoryManager
from .trainer_factory import TRAINER_FACTORY_REGISTRY, get_trainer
from .utils import (
    benchmark_model,
    compute_metrics,
    count_parameters,
    evaluate_model,
    plot_training_curves,
    save_training_history,
    train_model,
)

__all__ = [
    "BaseTrainer",
    "Trainer",
    "TrainerRegistry",
    "get_trainer",
    "TRAINER_FACTORY_REGISTRY",
    "ResultsManager",
    "MemoryCallback",
    "TimeCallback",
    "GradientNormCallback",
    "SVDAttentionCallback",
    "PreemptionCallback",
    "HyperparameterOptimizer",
    "PrunedHyperparameterOptimizer",
    "quick_optimize",
    "get_task_metrics",
    "compute_accuracy",
    "compute_f1",
    "compute_matthews_corrcoef",
    "compute_pearson_spearman",
    "aggregate_results",
    "train_model",
    "evaluate_model",
    "benchmark_model",
    "count_parameters",
    "compute_metrics",
    "save_training_history",
    "plot_training_curves",
    "HuggingFaceTrainer",
    "NeuralOpsTrainer",
    "CustomLoopTrainer",
    "AutoMaxGPUTrainer",
    "GPUMemoryManager",
    "AsyncDataPrefetcher",
    "InfiniteContextFoundationPretrainer",
    "OPTIMIZER_REGISTRY",
    "RiemannianStiefelOptimizer",
    "project_stiefel_gradient",
    "stiefel_qr_retraction",
]

from .foundation_pretrainer import InfiniteContextFoundationPretrainer
from .optimizers import (  # noqa: E402
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
from .schedulers import create_cosine, create_linear, create_onecycle, create_poly, create_rop  # noqa: E402

__all__ += [
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
    "create_linear",
    "create_cosine",
    "create_rop",
    "create_onecycle",
    "create_poly",
]

from .losses import ContrastiveLoss, CrossEntropyLoss, DiceLoss, FocalLoss, LabelSmoothingLoss, PolyLoss  # noqa: E402

__all__ += [
    "CrossEntropyLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "DiceLoss",
    "PolyLoss",
    "ContrastiveLoss",
]

from .distillation.multi_level_kd import (
    BlockwiseOperatorCloningLoss,
    HedgehogFeatureMapKDLoss,
    LinearCKALoss,
    LogitDistillationLoss,
    MiniLMValueRelationKDLoss,
    SmoothMaxOverLayersCKASteeringLoss,
    SpikyFeatureMap,
)
from .peft_recovery.svd_adapters import DoRALinear, PiSSALinear, inject_pissa_adapters
from .regularizers.attention_stability import GateSaturationPenalty, LearnableAttentionSink, SpectralJacobianRegularizer
from .rl_alignment.preference_and_onpolicy import DynamicRoutingAndRankRL, ORPOLoss, SimPOLoss
from .surgery_schedules.mohawk import (
    AlphaBlendingAttentionWrapper,
    AttentionHomotopyScheduler,
    HNCContinuationTrainer,
    MOHAWKTrainer,
)
from .ttt.test_time_trainer import TTTLinearAttentionLayer

__all__ += [
    "LogitDistillationLoss",
    "SpikyFeatureMap",
    "HedgehogFeatureMapKDLoss",
    "MiniLMValueRelationKDLoss",
    "BlockwiseOperatorCloningLoss",
    "LinearCKALoss",
    "SmoothMaxOverLayersCKASteeringLoss",
    "AttentionHomotopyScheduler",
    "AlphaBlendingAttentionWrapper",
    "MOHAWKTrainer",
    "HNCContinuationTrainer",
    "PiSSALinear",
    "DoRALinear",
    "inject_pissa_adapters",
    "SimPOLoss",
    "ORPOLoss",
    "DynamicRoutingAndRankRL",
    "LearnableAttentionSink",
    "GateSaturationPenalty",
    "SpectralJacobianRegularizer",
    "TTTLinearAttentionLayer",
]

from neuralops.trainer import (
    base_trainer,
    callbacks,
    custom_loop_trainer,
    hf_trainer,
    hyperopt,
    losses,
    methods,
    metrics,
    optimizers,
    registry,
    results_manager,
    rl_trainer,
    schedulers,
    train_lra,
    trainer,
    trainer_factory,
    trainer_svd,
    unsloth_trainer,
    utils,
)
