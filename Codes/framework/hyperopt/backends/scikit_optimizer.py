"""Scikit‑optimize (skopt) backend – Gaussian Processes, Random Forests."""

import logging
from typing import Any, Callable, Dict, Tuple

from skopt import dummy_minimize, forest_minimize, gp_minimize
from skopt.space import Categorical, Integer, Real

from ..base import HPO_REGISTRY, BaseOptimizer
from ..config import HPOConfig

logger = logging.getLogger(__name__)


@HPO_REGISTRY.register("scikit_optimize")
class ScikitOptimizer(BaseOptimizer):
    """Scikit‑optimize hyperparameter optimization."""

    METHOD_MAP = {
        "tpe": "ET",  # Extra Trees proxy
        "bayesian": "GP",
        "random": "dummy",
    }

    def __init__(self, config: HPOConfig):
        super().__init__(config)

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        **kwargs,
    ) -> Tuple[Dict[str, Any], float]:
        dimensions = []
        names = []
        for name, spec in self.config.search_space.items():
            names.append(name)
            if spec["type"] == "float":
                dimensions.append(
                    Real(
                        spec["low"],
                        spec["high"],
                        prior="log-uniform" if spec.get("log") else "uniform",
                        name=name,
                    )
                )
            elif spec["type"] == "int":
                dimensions.append(Integer(spec["low"], spec["high"], name=name))
            elif spec["type"] == "categorical":
                dimensions.append(Categorical(spec["choices"], name=name))

        def wrapped(params_list):
            params = dict(zip(names, params_list))
            score = objective_fn(params)
            return -score if self.config.direction == "maximize" else score

        method = self.METHOD_MAP.get(self.config.algorithm.value, "GP")
        if method == "GP":
            opt_fn = gp_minimize
        elif method == "ET":
            opt_fn = forest_minimize
        else:
            opt_fn = dummy_minimize

        res = opt_fn(
            wrapped,
            dimensions,
            n_calls=self.config.num_trials,
            random_state=self.config.seed,
        )
        best_params = dict(zip(names, res.x))
        best_score = -res.fun if self.config.direction == "maximize" else res.fun
        logger.info("SKopt finished: best score=%f", best_score)
        return best_params, best_score
