# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Results Saver Configuration

Configuration dataclasses and options for result serialization and artifact persistence.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SaveConfig:
    """User‑facing configuration for result persistence."""

    save_model: bool = True
    save_checkpoints: bool = True
    save_optimizer: bool = False
    save_logs: bool = True
    save_plots: bool = True
    save_artifacts: bool = True
    create_model_version: bool = True

    auto_save_if_better_than: float | None = None

    plot_options: dict[str, Any] = field(
        default_factory=lambda: {
            "loss_curve": True,
            "accuracy_curve": True,
            "confusion_matrix": False,
            "attention_heatmap": False,
            "gradient_flow": False,
            "lr_schedule": True,
            "gpu_memory": False,
            "training_time": True,
            "parameter_count": False,
            "rank_allocation": False,
            "pareto_front": False,
            "radar_chart": False,
            "loss_smooth": 0.6,
            "figsize": [10, 6],
            "dpi": 150,
            "class_names": None,
        }
    )

    storage_backends: list[str] = field(default_factory=lambda: ["local"])

    git_options: dict[str, Any] = field(default_factory=dict)
    dvc_options: dict[str, Any] = field(default_factory=dict)

    experiment_trackers: list[str] = field(default_factory=list)

    model_export_formats: list[str] = field(default_factory=list)

    model_registries: list[str] = field(default_factory=list)

    metadata_store: str | None = None

    docker_options: dict[str, Any] = field(default_factory=dict)

    release_options: dict[str, Any] = field(default_factory=dict)

    notification: dict[str, Any] = field(default_factory=dict)

    cloud_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Any) -> "SaveConfig":
        """Build SaveConfig from dict or return if already SaveConfig."""
        if isinstance(d, cls):
            return d
        if not isinstance(d, dict):
            return cls()
        valid = {k for k in d if k in cls.__dataclass_fields__}
        return cls(**{k: d[k] for k in valid})
