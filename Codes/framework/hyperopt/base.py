from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

import torch

from framework.core.registry import Registry

if TYPE_CHECKING:
    from .config import HPOConfig

logger = logging.getLogger(__name__)

HPO_REGISTRY = Registry("hpo_optimizers")


def broadcast_pruned_state(should_prune: bool) -> bool:
    """
    Broadcast Optuna / HPO TrialPruned decision across all distributed ranks via torch.distributed.all_gather_object.

    # FIX: Prevents NCCL collective communication deadlocks when a trial is pruned on rank 0 or any worker rank.
    # All ranks synchronously receive the decision and raise TrialPruned together.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return bool(should_prune)

    world_size = torch.distributed.get_world_size()
    pruned_flags = [False for _ in range(world_size)]
    # FIX: Synchronous broadcast of trial pruning decision across all distributed workers
    torch.distributed.all_gather_object(pruned_flags, bool(should_prune))
    return any(pruned_flags)


class BaseOptimizer(ABC):
    """Interface for HPO backends."""

    def __init__(self, config: "HPOConfig") -> None:
        self.config = config

    @abstractmethod
    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        **kwargs: Any,
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

