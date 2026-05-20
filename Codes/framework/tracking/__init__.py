"""
Experiment Tracking – pluggable backends (local, TensorBoard, WandB, MLflow)
with a unified API and HuggingFace Trainer callback.
"""

from .tracker import ExperimentTracker
from .backends import TrackingBackend, LOCAL_BACKEND, TENSORBOARD_BACKEND, WANDB_BACKEND, MLFLOW_BACKEND
from .callbacks import TrackerCallback

__all__ = [
    "ExperimentTracker",
    "TrackingBackend",
    "TrackerCallback",
    "LOCAL_BACKEND",
    "TENSORBOARD_BACKEND",
    "WANDB_BACKEND",
    "MLFLOW_BACKEND",
]