# framework/hyperopt/__init__.py
from .optimizer import HyperOptimizer, HPOAlgorithm, HPOBackend

# framework/hyperopt/optimizer.py
import optuna
import ray.tune as tune
from hyperopt import fmin, tpe, hp, Trials

class HyperOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.backend = config.get("backend", "optuna")
        self.algorithm = config.get("algorithm", "tpe")
        self.num_trials = config.get("num_trials", 50)
        self.search_space = config.get("search_space", {})

    def optimize(self, objective_fn):
        if self.backend == "optuna":
            return self._run_optuna(objective_fn)
        elif self.backend == "ray_tune":
            return self._run_ray_tune(objective_fn)
        elif self.backend == "hyperopt":
            return self._run_hyperopt(objective_fn)
        else:
            raise ValueError(f"Unknown HPO backend: {self.backend}")

    def _run_optuna(self, objective_fn):
        study = optuna.create_study(direction="maximize")
        def wrapped(trial):
            params = {k: trial._suggest(k, v) for k, v in self.search_space.items()}
            return objective_fn(params)
        study.optimize(wrapped, n_trials=self.num_trials)
        return study.best_params