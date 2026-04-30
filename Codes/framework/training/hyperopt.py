# framework/training/hyperopt.py
"""
Hyperparameter optimization using Optuna.
"""

import optuna
from optuna.trial import TrialState
import json
import pandas as pd
from typing import Dict, Any, Callable, Optional, List
from pathlib import Path


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.
    
    Optimizes hyperparameters for transformer models using Bayesian optimization.
    
    Example:
        >>> def train_fn(config):
        ...     # Train model with given config
        ...     return {'accuracy': 0.92}
        >>> 
        >>> def eval_fn(config):
        ...     # Evaluate model
        ...     return {'accuracy': 0.91}
        >>> 
        >>> optimizer = HyperparameterOptimizer(train_fn, eval_fn, config_template)
        >>> best_params = optimizer.run(n_trials=50)
    """
    
    def __init__(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        config_template: Dict[str, Any],
        study_name: str = "svd_attention_opt",
        storage: Optional[str] = None,
        direction: str = "maximize"
    ):
        """
        Args:
            train_fn: Function that takes config and trains model
            eval_fn: Function that takes config and returns metrics
            config_template: Base configuration dictionary
            study_name: Name for the Optuna study
            storage: Storage URL for persistent studies
            direction: Optimization direction ('maximize' or 'minimize')
        """
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.config_template = config_template
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.study = None
        self.best_params = None
    
    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Metric value to optimize
        """
        config = self.config_template.copy()
        
        # Suggest hyperparameters
        config['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-6, 5e-5, log=True
        )
        config['batch_size'] = trial.suggest_categorical(
            'batch_size', [8, 16, 32]
        )
        config['rank'] = trial.suggest_int(
            'rank', 32, 256, step=32
        )
        config['weight_decay'] = trial.suggest_float(
            'weight_decay', 0.0, 0.1
        )
        config['warmup_ratio'] = trial.suggest_float(
            'warmup_ratio', 0.0, 0.3
        )
        
        # Additional SVD-specific parameters
        if 'compression_method' in config:
            config['compression_method'] = trial.suggest_categorical(
                'compression_method', 
                ['standard_svd', 'randomized_svd', 'quantized_svd']
            )
        
        if 'energy_threshold' in config:
            config['energy_threshold'] = trial.suggest_float(
                'energy_threshold', 0.85, 0.99
            )
        
        # Train model
        self.train_fn(config)
        
        # Evaluate
        metrics = self.eval_fn(config)
        
        # Return primary metric
        return metrics.get('accuracy', metrics.get('matthews_correlation', 0.0))
    
    def run(
        self,
        n_trials: int = 20,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Time limit in seconds
            n_jobs: Number of parallel jobs
            show_progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary of best hyperparameters
        """
        # Create study
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True
        )
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar
        )
        
        # Save results
        self.best_params = self.study.best_params
        
        # Save best parameters
        with open('best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save trials dataframe
        df = self.study.trials_dataframe()
        df.to_csv('optuna_trials.csv', index=False)
        
        # Print summary
        print(f"\n✅ Optimization complete!")
        print(f"   Best value: {self.study.best_value:.4f}")
        print(f"   Best params: {self.best_params}")
        
        return self.best_params
    
    def get_trial_dataframe(self) -> pd.DataFrame:
        """Get dataframe of all trials."""
        if self.study is None:
            raise ValueError("Run optimization first")
        return self.study.trials_dataframe()
    
    def get_importance(self) -> Dict[str, float]:
        """
        Get hyperparameter importance.
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if self.study is None:
            raise ValueError("Run optimization first")
        
        importance = optuna.importance.get_param_importances(self.study)
        return importance
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history.
        
        Args:
            save_path: Path to save the plot
        """
        if self.study is None:
            raise ValueError("Run optimization first")
        
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.plot_optimization_history(self.study)
        if save_path:
            fig.write_image(save_path)
        return fig
    
    def plot_parallel_coordinate(self, save_path: Optional[str] = None):
        """Plot parallel coordinate plot."""
        if self.study is None:
            raise ValueError("Run optimization first")
        
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        if save_path:
            fig.write_image(save_path)
        return fig
    
    def plot_param_importances(self, save_path: Optional[str] = None):
        """Plot parameter importances."""
        if self.study is None:
            raise ValueError("Run optimization first")
        
        fig = optuna.visualization.plot_param_importances(self.study)
        if save_path:
            fig.write_image(save_path)
        return fig


class PrunedHyperparameterOptimizer(HyperparameterOptimizer):
    """
    Hyperparameter optimizer with pruning (early stopping).
    """
    
    def __init__(self, *args, pruner: str = "median", **kwargs):
        """
        Args:
            pruner: Pruner type ('median', 'halving', 'hyperband')
        """
        super().__init__(*args, **kwargs)
        
        if pruner == "median":
            self.pruner = optuna.pruners.MedianPruner()
        elif pruner == "halving":
            self.pruner = optuna.pruners.HalvingPruner()
        elif pruner == "hyperband":
            self.pruner = optuna.pruners.HyperbandPruner()
        else:
            self.pruner = optuna.pruners.MedianPruner()
    
    def objective(self, trial: optuna.trial.Trial) -> float:
        """Objective with intermediate reporting for pruning."""
        config = super().objective(trial)
        
        # Report intermediate values
        for step, value in enumerate(self.intermediate_values):
            trial.report(value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return self.final_value


def quick_optimize(
    train_fn: Callable,
    eval_fn: Callable,
    n_trials: int = 10,
    study_name: str = "quick_opt"
) -> Dict[str, Any]:
    """
    Quick hyperparameter optimization with sensible defaults.
    
    Args:
        train_fn: Training function
        eval_fn: Evaluation function
        n_trials: Number of trials
        study_name: Study name
        
    Returns:
        Best hyperparameters
    """
    config_template = {
        'learning_rate': 2e-5,
        'batch_size': 16,
        'rank': 128,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1
    }
    
    optimizer = HyperparameterOptimizer(train_fn, eval_fn, config_template, study_name)
    return optimizer.run(n_trials=n_trials)