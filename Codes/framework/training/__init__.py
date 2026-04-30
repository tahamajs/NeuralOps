# framework/training/__init__.py
"""
Training utilities for SVD-Lin framework.

This module provides custom trainers, training utilities, benchmark tools,
and hyperparameter optimization for transformer models.

Components:
    - SVDAwareTrainer: Custom trainer with SVD-aware callbacks
    - HyperparameterOptimizer: Optuna-based hyperparameter search
    - Training utilities: train_model, evaluate_model, benchmark_model
    - Metrics helpers: compute_metrics, save_metrics, plot_curves

Example:
    >>> from framework.training import SVDAwareTrainer, HyperparameterOptimizer
    >>> 
    >>> # Hyperparameter optimization
    >>> optimizer = HyperparameterOptimizer(train_fn, eval_fn, config)
    >>> best_params = optimizer.run(n_trials=20)
    >>> 
    >>> # Custom trainer
    >>> trainer = SVDAwareTrainer(
    ...     model=model,
    ...     args=training_args,
    ...     train_dataset=train_dataset,
    ...     eval_dataset=eval_dataset
    ... )
    >>> trainer.train()
"""

from .trainer import SVDAwareTrainer
from .hyperopt import HyperparameterOptimizer
from .utils import (
    train_model,
    evaluate_model,
    benchmark_model,
    count_parameters,
    compute_metrics,
    save_training_history,
    plot_training_curves
)
from .callbacks import (
    MemoryCallback,
    TimeCallback,
    GradientNormCallback,
    SVDAttentionCallback
)
from .metrics import (
    compute_accuracy,
    compute_f1,
    compute_matthews_corrcoef,
    compute_pearson_spearman,
    get_task_metrics
)

__all__ = [
    # Trainer
    "SVDAwareTrainer",
    
    # Hyperparameter optimization
    "HyperparameterOptimizer",
    
    # Training utilities
    "train_model",
    "evaluate_model",
    "benchmark_model",
    "count_parameters",
    "compute_metrics",
    "save_training_history",
    "plot_training_curves",
    
    # Callbacks
    "MemoryCallback",
    "TimeCallback",
    "GradientNormCallback",
    "SVDAttentionCallback",
    
    # Metrics
    "compute_accuracy",
    "compute_f1",
    "compute_matthews_corrcoef",
    "compute_pearson_spearman",
    "get_task_metrics",
]

__version__ = "1.0.0"