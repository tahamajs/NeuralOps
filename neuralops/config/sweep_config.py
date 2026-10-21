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


@dataclass
class AttentionSweepConfig:
    """Configuration for attention sweeps and compatible flat CLI presets."""

    model_name: str = "distilbert-base-uncased"
    task: str = "sst2"
    num_labels: int = 2
    max_len: int = 128
    attn_type: str = "full"
    epochs: int = 2
    batch_size: int = 16
    lr: float = 2e-5
    grad_accum: int = 1
    compressor: str | None = None
    compression_rank: int = 64
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    glue_base_path: str | None = None
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    attn_types: list[str] = field(default_factory=lambda: ["full"])
    output_dir: str = "./results/sweeps"
    log_dir: str = "./logs"
