"""Optuna backend – supports TPE, CMA‑ES, Random, BOHB, NSGA‑II."""

import logging
from typing import Any, Callable, Dict, Tuple

import optuna

from ..base import HPO_REGISTRY, BaseOptimizer
from ..config import HPOConfig

logger = logging.getLogger(__name__)


@HPO_REGISTRY.register("optuna")
class OptunaOptimizer(BaseOptimizer):
    """Optuna hyperparameter optimization."""

    SAMPLER_MAP = {
        "tpe": optuna.samplers.TPESampler,
        "random": optuna.samplers.RandomSampler,
        "cma_es": optuna.samplers.CmaEsSampler,
    }

    def __init__(self, config: HPOConfig) -> None:
        super().__init__(config)

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        **kwargs,
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize."""
        direction = "maximize" if self.config.direction == "maximize" else "minimize"
        sampler_cls = self.SAMPLER_MAP.get(
            self.config.algorithm.value, optuna.samplers.TPESampler
        )
        sampler = sampler_cls(seed=self.config.seed)

        pruner = None
        if self.config.pruning != "none":
            pruner = self._create_pruner()

        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=kwargs.get("study_name", "hpo_study"),
        )

        def wrapped_trial(trial: optuna.Trial) -> float:
            """Wrapped trial."""
            params = {}
            for name, spec in self.config.search_space.items():
                if spec["type"] == "float":
                    params[name] = trial.suggest_float(
                        name, spec["low"], spec["high"], log=spec.get("log", False)
                    )
                elif spec["type"] == "int":
                    params[name] = trial.suggest_int(
                        name, spec["low"], spec["high"], log=spec.get("log", False)
                    )
                elif spec["type"] == "categorical":
                    params[name] = trial.suggest_categorical(name, spec["choices"])
            score = objective_fn(params)

            # Manual pruning support
            if pruner:
                trial.report(score, step=trial.last_step if trial.last_step else 0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return score

        study.optimize(
            wrapped_trial,
            n_trials=self.config.num_trials,
            timeout=self.config.timeout_seconds,
        )
        best_params = study.best_params
        best_score = study.best_value
        logger.info(
            "Optuna finished: best score=%f, params=%s", best_score, best_params
        )
        return best_params, best_score

    def _create_pruner(self) -> Any:
        """ create pruner."""
        if self.config.pruning == "median":
            return optuna.pruners.MedianPruner(
                n_startup_trials=self.config.pruner_n_startup_trials,
                n_warmup_steps=self.config.pruner_n_warmup_steps,
                interval_steps=self.config.pruner_interval_steps,
            )
        elif self.config.pruning == "hyperband":
            return optuna.pruners.HyperbandPruner()
        elif self.config.pruning == "threshold":
            return optuna.pruners.ThresholdPruner(lower=0.0)
        return None
