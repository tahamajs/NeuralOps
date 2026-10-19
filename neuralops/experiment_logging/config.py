#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Experiment Logging Configuration

Defines structured configuration dataclasses for logging and experiment tracking.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoggingConfig:
    """Complete logging & experiment tracking configuration."""

    log_level: str = "INFO"
    log_format: str = "structured"
    log_to_console: bool = True
    log_to_file: bool = True
    log_dir: str = "./logs"
    experiment_name: str = "experiment"
    run_id: str | None = None
    use_timestamp: bool = True
    rank: int | None = None
    rank_specific_logs: bool = True
    use_queue_handler: bool = True
    metrics_enabled: bool = True
    metrics_flush_every: int = 10
    extra_metadata: dict[str, Any] = field(default_factory=dict)
