#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hyperparameter Sweep Configuration

Search space configurations and scheduling parameters for multi-seed and multi-method evaluation sweeps.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SweepConfig:
    name: str = "sweep"
    algorithm: str = "tpe"
    direction: str = "maximize"
    metric: str = "eval_accuracy"
    max_trials: int = 20
    parallel_workers: int = 1
    pruner: str = "hyperband"
    search_space: dict[str, Any] = field(default_factory=dict)
    base_config: dict[str, Any] = field(default_factory=dict)
