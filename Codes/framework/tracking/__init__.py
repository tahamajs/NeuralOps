"""
Experiment Tracking – pluggable backends (local, TensorBoard, WandB, MLflow)
with a unified API and HuggingFace Trainer callback.
"""

from .backends import (
    LOCAL_BACKEND,
    MLFLOW_BACKEND,
    TENSORBOARD_BACKEND,
    WANDB_BACKEND,
    TrackingBackend,
)
from .callbacks import TrackerCallback
from .tracker import ExperimentTracker

__all__ = [
    "ExperimentTracker",
    "TrackingBackend",
    "TrackerCallback",
    "LOCAL_BACKEND",
    "TENSORBOARD_BACKEND",
    "WANDB_BACKEND",
    "MLFLOW_BACKEND",
]
