"""Hyperopt backend – TPE, Random, Adaptive TPE."""

import logging
from typing import Any, Callable, Dict, Tuple

from hyperopt import Trials, fmin, hp, rand, tpe

from ..base import HPO_REGISTRY, BaseOptimizer
from ..config import HPOConfig

logger = logging.getLogger(__name__)


@HPO_REGISTRY.register("hyperopt")
class HyperoptOptimizer(BaseOptimizer):
    """Hyperopt hyperparameter optimization."""

    ALGO_MAP = {
        "tpe": tpe.suggest,
        "random": rand.suggest,
    }

    def __init__(self, config: HPOConfig) -> None:
        super().__init__(config)

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        **kwargs,
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize."""
        space = {}
        for name, spec in self.config.search_space.items():
            if spec["type"] == "float":
                space[name] = hp.uniform(name, spec["low"], spec["high"])
            elif spec["type"] == "int":
                space[name] = hp.quniform(name, spec["low"], spec["high"], 1)
            elif spec["type"] == "categorical":
                space[name] = hp.choice(name, spec["choices"])

        algo = self.ALGO_MAP.get(self.config.algorithm.value, tpe.suggest)

        # Direction
        def wrapped(params) -> Any:
            """Wrapped."""
            score = objective_fn(params)
            return -score if self.config.direction == "maximize" else score

        trials = Trials()
        best = fmin(
            fn=wrapped,
            space=space,
            algo=algo,
            max_evals=self.config.num_trials,
            trials=trials,
        )
        best_score = min([t["result"]["loss"] for t in trials.trials])
        if self.config.direction == "maximize":
            best_score = -best_score
        logger.info("Hyperopt finished: best score=%f, params=%s", best_score, best)
        return best, best_score
