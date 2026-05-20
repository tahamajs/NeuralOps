"""Experiment logging – structured logging, metrics tracking, and Trainer callbacks."""

from .config import LoggingConfig
from .logger import setup_logging, get_logger
from .tracker import MetricsTracker
from .callbacks import LoggingCallback
from .integration import auto_setup

__all__ = [
    "LoggingConfig",
    "setup_logging",
    "get_logger",
    "MetricsTracker",
    "LoggingCallback",
    "auto_setup",
]