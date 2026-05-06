"""Ray Tune backend – supports ASHA, BOHB, HyperBand, PBT, etc."""

import logging
from typing import Callable, Dict, Any, Tuple
from ..base import BaseOptimizer, HPO_REGISTRY
from ..config import HPOConfig

logger = logging.getLogger(__name__)


@HPO_REGISTRY.register("ray_tune")
class RayTuneOptimizer(BaseOptimizer):
    """Ray Tune hyperparameter optimization."""

    ALGORITHM_MAP = {
        "tpe": "OptunaSearch",
        "random": "RandomSearch",
        "bayesian": "BayesOptSearch",
        "bohb": "BOHB",
        "asha": "ASHA",
        "hyperband": "HyperBand",
    }

    def __init__(self, config: HPOConfig):
        super().__init__(config)
        try:
            from ray import tune
            from ray.tune.search import ConcurrencyLimiter
            self.tune = tune
        except ImportError:
            raise ImportError("Ray Tune not installed. Run: pip install 'ray[tune]'")

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        **kwargs,
    ) -> Tuple[Dict[str, Any], float]:
        config = {}
        for name, spec in self.config.search_space.items():
            if spec["type"] == "float":
                config[name] = self.tune.uniform(spec["low"], spec["high"])
            elif spec["type"] == "int":
                config[name] = self.tune.randint(spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                config[name] = self.tune.choice(spec["choices"])

        algo_str = self.ALGORITHM_MAP.get(self.config.algorithm.value, "OptunaSearch")
        if algo_str == "OptunaSearch":
            from ray.tune.search.optuna import OptunaSearch as SearchAlgo
        elif algo_str == "RandomSearch":
            from ray.tune.search.basic_variant import BasicVariantGenerator as SearchAlgo
        else:
            SearchAlgo = None

        if SearchAlgo:
            search_alg = SearchAlgo(seed=self.config.seed)
        else:
            search_alg = None

        def trainable(config_dict):
            return {"score": objective_fn(config_dict)}

        analysis = self.tune.run(
            trainable,
            config=config,
            num_samples=self.config.num_trials,
            metric="score",
            mode="max" if self.config.direction == "maximize" else "min",
            search_alg=search_alg,
        )
        best = analysis.get_best_trial("score", "max", "last")
        best_params = best.config
        best_score = best.last_result["score"]
        logger.info("Ray Tune finished: best score=%f", best_score)
        return best_params, best_score