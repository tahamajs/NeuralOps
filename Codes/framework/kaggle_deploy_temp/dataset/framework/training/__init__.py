"""
Training modules re-exported for convenience.
"""

from framework.trainer import HuggingFaceTrainer
from framework.trainer import methods
from framework.trainer.methods import get_method
from framework.trainer.results_manager import ResultsManager
from framework.trainer.metrics import get_task_metrics

__all__ = [
    "HuggingFaceTrainer",
    "ResultsManager",
    "get_task_metrics",
    "methods",
    "get_method",
]