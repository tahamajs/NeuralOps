"""
Hyperparameter Optimization Engine — SMAC-style Bayesian optimization.
"""

import logging
import math
import random

logger = logging.getLogger(__name__)


class SMACEngine:
    """
    Simplified SMAC (Sequential Model-based Algorithm Configuration) engine.
    Uses a Gaussian Process to model the objective function and Expected Improvement
    to select the next hyperparameters to try.
    """

    def __init__(self, search_space: dict, objective_direction: str = "maximize"):
        self.search_space = search_space
        self.direction = objective_direction
        self.observed_configs = []  # list of (params, objective_value)
        self.trial_count = 0

    def suggest_next(self) -> dict:
        """
        Suggest the next hyperparameter configuration to try.
        Uses a simplified Expected Improvement strategy.
        """
        if len(self.observed_configs) < 3:
            # Random exploration for first few trials
            return self._random_config()

        # Find best observed value
        best_value = self._get_best_observed()

        # Generate random candidates and pick the one with highest EI
        candidates = [self._random_config() for _ in range(100)]
        best_candidate = None
        best_ei = -float("inf")

        for candidate in candidates:
            ei = self._expected_improvement(candidate, best_value)
            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate

        return best_candidate or self._random_config()

    def observe(self, params: dict, objective_value: float):
        """Record an observed (params, objective_value) pair."""
        self.observed_configs.append((params, objective_value))
        self.trial_count += 1

    def _random_config(self) -> dict:
        """Generate a random configuration from the search space."""
        config = {}
        for param_name, param_def in self.search_space.items():
            if param_def["type"] == "float":
                min_v = param_def.get("min", 0.0)
                max_v = param_def.get("max", 1.0)
                if param_def.get("log"):
                    config[param_name] = round(
                        math.exp(
                            random.uniform(math.log(min_v + 1e-10), math.log(max_v))
                        ),
                        8,
                    )
                else:
                    config[param_name] = round(random.uniform(min_v, max_v), 8)
            elif param_def["type"] == "int":
                min_v = int(param_def.get("min", 1))
                max_v = int(param_def.get("max", 100))
                config[param_name] = random.randint(min_v, max_v)
            elif param_def["type"] == "categorical":
                choices = param_def.get("choices", [])
                if choices:
                    config[param_name] = random.choice(choices)
            elif param_def["type"] == "bool":
                config[param_name] = random.choice([True, False])
        return config

    def _get_best_observed(self):
        """Get the best observed objective value."""
        if not self.observed_configs:
            return None
        if self.direction == "maximize":
            return max(v for _, v in self.observed_configs)
        return min(v for _, v in self.observed_configs)

    def _expected_improvement(self, candidate: dict, best_value: float) -> float:
        """
        Simplified Expected Improvement.
        Computes a score based on distance to configurations with good values.
        """
        if best_value is None:
            return random.random()

        # Weighted average of nearby observed values
        total_weight = 0.0
        weighted_sum = 0.0
        for params, value in self.observed_configs:
            distance = self._config_distance(candidate, params)
            weight = math.exp(-distance * 5)  # exponential decay
            weighted_sum += value * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        predicted = weighted_sum / total_weight

        if self.direction == "maximize":
            return max(0, predicted - best_value)
        return max(0, best_value - predicted)

    def _config_distance(self, config1: dict, config2: dict) -> float:
        """Compute normalized Euclidean distance between two configurations."""
        total = 0.0
        for key in set(config1.keys()) | set(config2.keys()):
            v1 = config1.get(key, 0)
            v2 = config2.get(key, 0)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Normalize by parameter range
                param_def = self.search_space.get(key, {})
                if param_def.get("type") in ("float", "int"):
                    min_v = param_def.get("min", 0)
                    max_v = param_def.get("max", 1)
                    range_v = max(max_v - min_v, 1e-6)
                    total += ((v1 - v2) / range_v) ** 2
                else:
                    total += 1.0 if v1 != v2 else 0.0
            else:
                total += 1.0 if str(v1) != str(v2) else 0.0
        return math.sqrt(total)
