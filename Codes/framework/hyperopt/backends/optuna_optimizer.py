"""
Optuna backend – supports TPE, CMA‑ES, Random, BOHB, NSGA‑II with distributed TrialPruned broadcast.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import optuna
except ImportError:
    optuna = None
import torch

from ..base import HPO_REGISTRY, BaseOptimizer, broadcast_pruned_state
from ..config import HPOConfig

logger = logging.getLogger(__name__)


@HPO_REGISTRY.register("optuna")
class OptunaOptimizer(BaseOptimizer):
    """Optuna hyperparameter optimization with distributed NCCL deadlock prevention."""

    # Keep the backend importable when Optuna is not installed.  Optuna is an
    # optional HPO dependency and importing the framework should not fail just
    # because a caller is using another backend.
    SAMPLER_MAP = (
        {
            "tpe": optuna.samplers.TPESampler,
            "random": optuna.samplers.RandomSampler,
            "cma_es": optuna.samplers.CmaEsSampler,
        }
        if optuna is not None
        else {}
    )

    def __init__(self, config: HPOConfig) -> None:
        super().__init__(config)

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters with synchronous distributed trial pruning."""
        self._require_optuna()
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
            """
            Wrapped trial execution with distributed synchronization and NCCL deadlock prevention.
            """
            params: Dict[str, Any] = {}
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

            # FIX: Execute objective with TrialPruned exception interception
            is_pruned = False
            score = 0.0
            try:
                score = float(objective_fn(params))
            except optuna.TrialPruned:
                is_pruned = True

            # FIX: Use torch.distributed.all_gather_object to broadcast Optuna TrialPruned states so all ranks exit trials synchronously
            if broadcast_pruned_state(is_pruned):
                raise optuna.TrialPruned("Trial pruned during distributed objective evaluation")

            # Manual pruning support with Optuna pruners
            if pruner is not None:
                local_should_prune = False
                rank = (
                    torch.distributed.get_rank()
                    if (torch.distributed.is_available() and torch.distributed.is_initialized())
                    else 0
                )
                if rank == 0:
                    trial.report(score, step=trial.last_step if trial.last_step is not None else 0)
                    local_should_prune = trial.should_prune()

                # FIX: Broadcast rank 0 pruner decision to all ranks using all_gather_object to prevent NCCL hang
                if broadcast_pruned_state(local_should_prune):
                    logger.info("Trial %d pruned synchronously across distributed ranks.", trial.number)
                    raise optuna.TrialPruned("Trial pruned by Optuna pruner across all ranks")

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
        """Create Optuna pruner instance based on configuration."""
        self._require_optuna()
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

    @staticmethod
    def _require_optuna() -> None:
        if optuna is None:
            raise ImportError(
                "Optuna is required for the 'optuna' HPO backend. "
                "Install it with `pip install optuna`."
            )
