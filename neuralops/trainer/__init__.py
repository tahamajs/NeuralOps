"""
Training module – unified trainers, results management, metrics, and callbacks.
"""

from .base_trainer import BaseTrainer
from .callbacks import (
    GradientNormCallback,
    MemoryCallback,
    PreemptionCallback,
    SVDAttentionCallback,
    TimeCallback,
)
from .custom_loop_trainer import CustomLoopTrainer

# Import trainers to register them
from .hf_trainer import HuggingFaceTrainer
from .hyperopt import (
    HyperparameterOptimizer,
    PrunedHyperparameterOptimizer,
    quick_optimize,
)
from .metrics import (
    aggregate_results,
    compute_accuracy,
    compute_f1,
    compute_matthews_corrcoef,
    compute_pearson_spearman,
    get_task_metrics,
)
from .optimizers import (
    OPTIMIZER_REGISTRY,
    RiemannianStiefelOptimizer,
    project_stiefel_gradient,
    stiefel_qr_retraction,
)
from .registry import TrainerRegistry
from .results_manager import ResultsManager
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
    "CustomLoopTrainer",
    "OPTIMIZER_REGISTRY",
    "RiemannianStiefelOptimizer",
    "project_stiefel_gradient",
    "stiefel_qr_retraction",
]

# Optimizer factories
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

# Scheduler factories
from .schedulers import (  # noqa: E402
    create_cosine,
    create_linear,
    create_onecycle,
    create_poly,
    create_rop,
)

__all__ += [
    # Optimizers
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
    # Schedulers
    "create_linear",
    "create_cosine",
    "create_rop",
    "create_onecycle",
    "create_poly",
]

# Loss functions
from .losses import (  # noqa: E402
    ContrastiveLoss,
    CrossEntropyLoss,
    DiceLoss,
    FocalLoss,
    LabelSmoothingLoss,
    PolyLoss,
)

__all__ += [
    "CrossEntropyLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "DiceLoss",
    "PolyLoss",
    "ContrastiveLoss",
]

# Pillar 1-6 exports
from .distillation.multi_level_kd import (
    LogitDistillationLoss,
    SpikyFeatureMap,
    HedgehogFeatureMapKDLoss,
    MiniLMValueRelationKDLoss,
    BlockwiseOperatorCloningLoss,
    LinearCKALoss,
)
from .surgery_schedules.mohawk import (
    AttentionHomotopyScheduler,
    AlphaBlendingAttentionWrapper,
    MOHAWKTrainer,
)
from .peft_recovery.svd_adapters import (
    PiSSALinear,
    DoRALinear,
    inject_pissa_adapters,
)
from .rl_alignment.preference_and_onpolicy import (
    SimPOLoss,
    ORPOLoss,
    DynamicRoutingAndRankRL,
)
from .regularizers.attention_stability import (
    LearnableAttentionSink,
    GateSaturationPenalty,
    SpectralJacobianRegularizer,
)
from .ttt.test_time_trainer import TTTLinearAttentionLayer

__all__ += [
    "LogitDistillationLoss",
    "SpikyFeatureMap",
    "HedgehogFeatureMapKDLoss",
    "MiniLMValueRelationKDLoss",
    "BlockwiseOperatorCloningLoss",
    "LinearCKALoss",
    "AttentionHomotopyScheduler",
    "AlphaBlendingAttentionWrapper",
    "MOHAWKTrainer",
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

# Auto-wired package members
from neuralops.trainer import methods
from neuralops.trainer import optimizers
from neuralops.trainer import base_trainer
from neuralops.trainer import callbacks
from neuralops.trainer import custom_loop_trainer
from neuralops.trainer import hf_trainer
from neuralops.trainer import hyperopt
from neuralops.trainer import losses
from neuralops.trainer import metrics
from neuralops.trainer import optimizers
from neuralops.trainer import registry
from neuralops.trainer import results_manager
from neuralops.trainer import rl_trainer
from neuralops.trainer import schedulers
from neuralops.trainer import train_lra
from neuralops.trainer import trainer
from neuralops.trainer import trainer_factory
from neuralops.trainer import trainer_svd
from neuralops.trainer import unsloth_trainer
from neuralops.trainer import utils
