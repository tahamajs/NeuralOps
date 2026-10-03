import json
from collections.abc import Callable
from typing import Any

import pandas as pd

try:
    import optuna
    from optuna.trial import TrialState

    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


class HyperparameterOptimizer:
    def __init__(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        config_template: dict[str, Any],
        study_name: str = "svd_attention_opt",
        storage: str | None = None,
        direction: str = "maximize",
    ) -> None:
        if not _HAS_OPTUNA:
            raise ImportError("optuna is required. pip install optuna")
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.config_template = config_template
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.study = None
        self.best_params = None

    def objective(self, trial) -> float:
        """Objective."""
        config = self.config_template.copy()
        config["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
        config["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32])
        config["rank"] = trial.suggest_int("rank", 32, 256, step=32)
        config["weight_decay"] = trial.suggest_float("weight_decay", 0.0, 0.1)
        config["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.0, 0.3)
        if "compression_method" in config:
            config["compression_method"] = trial.suggest_categorical(
                "compression_method",
                ["standard_svd", "randomized_svd", "quantized_svd"],
            )
        if "energy_threshold" in config:
            config["energy_threshold"] = trial.suggest_float("energy_threshold", 0.85, 0.99)
        self.train_fn(config)
        metrics = self.eval_fn(config)
        return metrics.get("accuracy", metrics.get("matthews_correlation", 0.0))

    def run(
        self,
        n_trials: int = 20,
        timeout: int | None = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
    ) -> dict[str, Any]:
        """Execute the run operation and return results."""
        self.study = optuna.create_study(
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
        with open("best_params.json", "w") as f:
            json.dump(self.best_params, f, indent=2)
        df = self.study.trials_dataframe()
        df.to_csv("optuna_trials.csv", index=False)
        print(f"\n✅ Optimization complete! Best value: {self.study.best_value:.4f}")
        print(f"Best params: {self.best_params}")
        return self.best_params

    def get_trial_dataframe(self) -> pd.DataFrame:
        """Return the trial dataframe."""
        if self.study is None:
            raise ValueError("Run optimization first")
        return self.study.trials_dataframe()

    def get_importance(self) -> dict[str, float]:
        """Return the importance."""
        if self.study is None:
            raise ValueError("Run optimization first")
        return optuna.importance.get_param_importances(self.study)

    def plot_optimization_history(self, save_path=None) -> Any:
        """Plot optimization history."""
        if self.study is None:
            raise ValueError("Run optimization first")
        fig = optuna.visualization.plot_optimization_history(self.study)
        if save_path:
            fig.write_image(save_path)
        return fig

    def plot_parallel_coordinate(self, save_path=None) -> Any:
        """Plot parallel coordinate."""
        if self.study is None:
            raise ValueError("Run optimization first")
        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        if save_path:
            fig.write_image(save_path)
        return fig

    def plot_param_importances(self, save_path=None) -> Any:
        """Plot param importances."""
        if self.study is None:
            raise ValueError("Run optimization first")
        fig = optuna.visualization.plot_param_importances(self.study)
        if save_path:
            fig.write_image(save_path)
        return fig


class PrunedHyperparameterOptimizer(HyperparameterOptimizer):
    def __init__(self, *args, pruner="median", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if pruner == "median":
            self.pruner = optuna.pruners.MedianPruner()
        elif pruner == "halving":
            self.pruner = optuna.pruners.HalvingPruner()
        elif pruner == "hyperband":
            self.pruner = optuna.pruners.HyperbandPruner()
        else:
            self.pruner = optuna.pruners.MedianPruner()


def quick_optimize(train_fn, eval_fn, n_trials=10, study_name="quick_opt") -> dict[str, Any]:
    """Quick optimize."""
    config_template = {
        "learning_rate": 2e-5,
        "batch_size": 16,
        "rank": 128,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
    }
    optimizer = HyperparameterOptimizer(train_fn, eval_fn, config_template, study_name)
    return optimizer.run(n_trials=n_trials)
