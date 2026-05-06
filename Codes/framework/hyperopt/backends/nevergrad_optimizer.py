"""Nevergrad backend – NGOpt, PSO, DE, etc."""

import logging
from typing import Callable, Dict, Any, Tuple
import nevergrad as ng
from ..base import BaseOptimizer, HPO_REGISTRY
from ..config import HPOConfig

logger = logging.getLogger(__name__)


@HPO_REGISTRY.register("nevergrad")
class NevergradOptimizer(BaseOptimizer):
    """Nevergrad hyperparameter optimization."""

    ALGO_MAP = {
        "ngopt": "NGOpt",
        "pso": "PSO",
        "de": "DE",
        "random": "RandomSearch",
    }

    def __init__(self, config: HPOConfig):
        super().__init__(config)

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        **kwargs,
    ) -> Tuple[Dict[str, Any], float]:
        instrumentation = ng.p.Instrumentation(**{
            name: ng.p.Scalar(init=spec.get("low") or 0.5)
            if spec["type"] in ("float", "int")
            else ng.p.Choice(spec["choices"])
            for name, spec in self.config.search_space.items()
        })

        algo_name = self.ALGO_MAP.get(self.config.algorithm.value, "NGOpt")
        optimizer = ng.optimizers.registry[algo_name](
            parametrization=instrumentation,
            budget=self.config.num_trials,
            num_workers=1,
        )

        def wrapped(**params):
            score = objective_fn(params)
            return -score if self.config.direction == "maximize" else score

        recommendation = optimizer.minimize(wrapped)
        best_params = recommendation.value
        best_score = -recommendation.loss if self.config.direction == "maximize" else recommendation.loss
        logger.info("Nevergrad finished: best score=%f, params=%s", best_score, best_params)
        return best_params, best_score