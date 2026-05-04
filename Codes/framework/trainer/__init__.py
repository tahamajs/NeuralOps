from .base_trainer import BaseTrainer
from .registry import TrainerRegistry
from .trainer_factory import get_trainer
from .results_manager import ResultsManager
from .callbacks import MemoryCallback, TimeCallback, GradientNormCallback, SVDAttentionCallback
from .hyperopt import HyperparameterOptimizer, PrunedHyperparameterOptimizer, quick_optimize
from .metrics import get_task_metrics, compute_accuracy, compute_f1, compute_matthews_corrcoef, compute_pearson_spearman, aggregate_results
from .utils import train_model, evaluate_model, benchmark_model, count_parameters, compute_metrics, save_training_history, plot_training_curves

# Import trainers to register them
from .hf_trainer import HuggingFaceTrainer
from .unsloth_trainer import UnslothTrainer
from .rl_trainer import RLTrainer
from .custom_loop_trainer import CustomLoopTrainer

__all__ = [
    "BaseTrainer",
    "TrainerRegistry",
    "get_trainer",
    "ResultsManager",
    "MemoryCallback",
    "TimeCallback",
    "GradientNormCallback",
    "SVDAttentionCallback",
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
]