"""
Training modules re-exported for convenience.
"""

from framework.trainer import HuggingFaceTrainer
from framework.trainer import methods
from framework.trainer.methods import get_method
from framework.experiment_logging import ResultsManager
from framework.evaluation.metrics import get_task_metrics

__all__ = [
    "HuggingFaceTrainer",
    "ResultsManager",
    "get_task_metrics",
    "methods",
    "get_method",
]