# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Experiment Logging Subpackage.
"""

from neuralops.trainer.results_manager import ResultsManager

from .callbacks import LoggingCallback
from .config import LoggingConfig
from .integration import auto_setup
from .logger import get_logger, get_rank, setup_logging, shutdown_logging
from .tracker import MetricsTracker

__all__ = [
    "LoggingConfig",
    "setup_logging",
    "shutdown_logging",
    "get_logger",
    "get_rank",
    "MetricsTracker",
    "LoggingCallback",
    "auto_setup",
    "ResultsManager",
]

from neuralops.experiment_logging import callbacks
from neuralops.experiment_logging import config
from neuralops.experiment_logging import integration
from neuralops.experiment_logging import logger
from neuralops.experiment_logging import tracker
