"""Experiment logging – structured logging, metrics tracking, and Trainer callbacks."""

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

# Auto-wired package members
from neuralops.experiment_logging import callbacks
from neuralops.experiment_logging import config
from neuralops.experiment_logging import integration
from neuralops.experiment_logging import logger
from neuralops.experiment_logging import tracker
