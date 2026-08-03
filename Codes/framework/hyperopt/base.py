from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple

from framework.core.registry import Registry

HPO_REGISTRY = Registry("hpo_optimizers")


class BaseOptimizer(ABC):
    """Interface for HPO backends."""

    def __init__(self, config: "HPOConfig") -> None:
        self.config = config

    @abstractmethod
    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        **kwargs,
    ) -> Tuple[Dict[str, Any], float]:
        """Run optimization and return (best_params, best_score)."""
        pass


class BasePruner(ABC):
    """Interface for trial pruners."""

    def __init__(self, config: "HPOConfig") -> None:
        self.config = config

    @abstractmethod
    def should_prune(
        self, trial_id: int, intermediate_result: float, step: int
    ) -> bool:
        """Should prune."""
        pass
