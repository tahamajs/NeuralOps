# framework/tracking/__init__.py
from .experiment_tracker import ExperimentTracker, TrackingBackend

# framework/tracking/experiment_tracker.py
from enum import Enum
import wandb, mlflow, torch.utils.tensorboard as tb

class TrackingBackend(Enum):
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    MLFLOW = "mlflow"
    LOCAL = "local"

class ExperimentTracker:
    def __init__(self, config):
        self.backends = []
        if "tensorboard" in config:
            self.backends.append(TensorBoardBackend(config["tensorboard"]))
        if "wandb" in config:
            self.backends.append(WandbBackend(config["wandb"]))
        if "mlflow" in config:
            self.backends.append(MLflowBackend(config["mlflow"]))
        if "local" in config:
            self.backends.append(LocalBackend(config["local"]))

    def log(self, metrics, step, prefix=""):
        for b in self.backends:
            b.log(metrics, step, prefix)

    def finalize(self, result):
        for b in self.backends:
            b.finalize()