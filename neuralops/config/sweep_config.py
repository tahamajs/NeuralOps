from dataclasses import dataclass, field
from typing import Any


@dataclass
class SweepConfig:
    name: str = "sweep"
    algorithm: str = "tpe"  # tpe, random, grid, cmaes
    direction: str = "maximize"
    metric: str = "eval_accuracy"
    max_trials: int = 20
    parallel_workers: int = 1
    pruner: str = "hyperband"
    search_space: dict[str, Any] = field(default_factory=dict)
    base_config: dict[str, Any] = field(default_factory=dict)
