# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Experiment Configuration

Master configuration schema orchestrating complete benchmark, training, and evaluation pipelines.
"""

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    name: str
    path: str
    task_type: str
    num_choices: int = 0
    num_labels: int = 2


@dataclass
class ExperimentConfig:
    base_models: dict[str, str] = field(default_factory=lambda: {"bert": "bert-base-uncased"})
    svd_ranks: list[int] = field(default_factory=lambda: [8, 16, 32])
    datasets: list[DatasetConfig] = field(default_factory=list)
    min_samples: int = 10
    max_samples: int = 200
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 2e-5
    max_length: int = 128
    val_split: float = 0.2
    seed: int = 42
    gradient_accumulation_steps: int = 1

    optimizer_type: str = "adamw"
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 0
    weight_decay: float = 0.01
    mixed_precision: str = "no"
    gradient_checkpointing: bool = (
        False
    )
    max_grad_norm: float = 1.0

    save_checkpoint_interval: int = 10
    output_dir: str = "./results"
    plots_dir: str = "./plots/comprehensive"
    log_dir: str = "./logs"
    selected_models: list[str] = field(default_factory=list)
    selected_datasets: list[str] = field(default_factory=list)
    selected_ranks: list[int] = field(default_factory=list)
