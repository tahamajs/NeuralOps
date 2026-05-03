# framework/training/hyperopt.py
"""
Hyperparameter optimization using Optuna.

Requires ``pip install optuna`` for actual optimization.
If optuna is not installed, the classes will raise an ImportError
when instantiated (not at import time).
"""

import json
import pandas as pd
from typing import Dict, Any, Callable, Optional, List
from pathlib import Path

# Try to import optuna once
try:
    import optuna  # noqa: F401
    from optuna.trial import TrialState  # noqa: F401
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.

    Example:
        >>> def train_fn(config):
        ...     return {'accuracy': 0.92}
        >>> def eval_fn(config):
        ...     return {'accuracy': 0.91}
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
        direction: str = "maximize",
    ):
        if not _HAS_OPTUNA:
            raise ImportError(
                "optuna is required for HyperparameterOptimizer. "
                "Install with: pip install optuna"
            )
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.config_template = config_template
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.study = None
        self.best_params = None

    def objective(self, trial) -> float:
        """
        Objective function for Optuna optimization.
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
                ['standard_svd', 'randomized_svd', 'quantized_svd'],
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
        show_progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        import optuna as optuna_mod  # local import so we can use the module

        self.study = optuna_mod.create_study(
            direction=self.direction,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
        )
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
        )

        self.best_params = self.study.best_params

        with open('best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)

        df = self.study.trials_dataframe()
        df.to_csv('optuna_trials.csv', index=False)

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
        """Get hyperparameter importance."""
        if self.study is None:
            raise ValueError("Run optimization first")
        import optuna as optuna_mod
        return optuna_mod.importance.get_param_importances(self.study)

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if self.study is None:
            raise ValueError("Run optimization first")
        import optuna as optuna_mod
        fig = optuna_mod.visualization.plot_optimization_history(self.study)
        if save_path:
            fig.write_image(save_path)
        return fig

    def plot_parallel_coordinate(self, save_path: Optional[str] = None):
        """Plot parallel coordinate plot."""
        if self.study is None:
            raise ValueError("Run optimization first")
        import optuna as optuna_mod
        fig = optuna_mod.visualization.plot_parallel_coordinate(self.study)
        if save_path:
            fig.write_image(save_path)
        return fig

    def plot_param_importances(self, save_path: Optional[str] = None):
        """Plot parameter importances."""
        if self.study is None:
            raise ValueError("Run optimization first")
        import optuna as optuna_mod
        fig = optuna_mod.visualization.plot_param_importances(self.study)
        if save_path:
            fig.write_image(save_path)
        return fig


class PrunedHyperparameterOptimizer(HyperparameterOptimizer):
    """Hyperparameter optimizer with pruning (early stopping)."""

    def __init__(self, *args, pruner: str = "median", **kwargs):
        super().__init__(*args, **kwargs)
        import optuna as optuna_mod
        if pruner == "median":
            self.pruner = optuna_mod.pruners.MedianPruner()
        elif pruner == "halving":
            self.pruner = optuna_mod.pruners.HalvingPruner()
        elif pruner == "hyperband":
            self.pruner = optuna_mod.pruners.HyperbandPruner()
        else:
            self.pruner = optuna_mod.pruners.MedianPruner()


def quick_optimize(
    train_fn: Callable,
    eval_fn: Callable,
    n_trials: int = 10,
    study_name: str = "quick_opt",
) -> Dict[str, Any]:
    """
    Quick hyperparameter optimization with sensible defaults.
    """
    config_template = {
        'learning_rate': 2e-5,
        'batch_size': 16,
        'rank': 128,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
    }
    optimizer = HyperparameterOptimizer(
        train_fn, eval_fn, config_template, study_name
    )
    return optimizer.run(n_trials=n_trials)