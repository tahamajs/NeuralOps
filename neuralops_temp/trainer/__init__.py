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
