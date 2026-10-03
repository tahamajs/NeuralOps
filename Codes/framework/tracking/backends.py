"""
Backend implementations for experiment tracking.
Each backend handles the specifics of initialisation, metric/param/artifact logging, and finalisation.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Enum & backend keys
# ----------------------------------------------------------------------
class TrackingBackend(Enum):
    LOCAL = "local"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    MLFLOW = "mlflow"


# Convenience constants for configuration
LOCAL_BACKEND = TrackingBackend.LOCAL.value
TENSORBOARD_BACKEND = TrackingBackend.TENSORBOARD.value
WANDB_BACKEND = TrackingBackend.WANDB.value
MLFLOW_BACKEND = TrackingBackend.MLFLOW.value


# ----------------------------------------------------------------------
# Abstract base
# ----------------------------------------------------------------------
class BaseTracker(ABC):
    """Every backend must implement these five methods."""

    @abstractmethod
    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log metric."""
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log params."""
        pass

    @abstractmethod
    def log_artifact(self, local_path: str) -> None:
        """Log artifact."""
        pass

    @abstractmethod
    def finalize(self, result: dict[str, Any] | None = None) -> None:
        """Finalize."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close."""
        pass


# ----------------------------------------------------------------------
# Local JSON backend
# ----------------------------------------------------------------------
class LocalTracker(BaseTracker):
    """
    Writes metrics and parameters to a JSON lines file.
    Artifacts are copied into the log directory.
    """

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = os.path.join(
            self.log_dir, f"{experiment_name}_{timestamp}_metrics.jsonl"
        )
        self.params_file = os.path.join(
            self.log_dir, f"{experiment_name}_{timestamp}_params.json"
        )
        self.metrics = []

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log metric."""
        entry = {
            "name": name,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }
        self.metrics.append(entry)
        # Append to file immediately for resilience
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params."""
        with open(self.params_file, "w") as f:
            json.dump(params, f, indent=2)

    def log_artifact(self, local_path: str) -> None:
        """Log artifact."""
        import shutil

        dest = os.path.join(self.log_dir, os.path.basename(local_path))
        shutil.copy2(local_path, dest)

    def finalize(self, result: dict[str, Any] | None = None) -> None:
        """Finalize."""
        if result:
            summary_file = os.path.join(
                self.log_dir, f"{self.experiment_name}_summary.json"
            )
            with open(summary_file, "w") as f:
                json.dump(result, f, indent=2)

    def close(self) -> None:
        """Close."""
        pass


# ----------------------------------------------------------------------
# TensorBoard backend
# ----------------------------------------------------------------------
class TensorBoardTracker(BaseTracker):
    """Writes scalar metrics to TensorBoard event files."""

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        run_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(log_dir=run_dir)
        self._params = {}

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log metric."""
        if step is not None:
            self.writer.add_scalar(name, value, global_step=step)
        else:
            self.writer.add_scalar(name, value, global_step=0)

    def log_params(self, params: dict[str, Any]) -> None:
        # TensorBoard doesn't natively support hyperparameters well;
        # we store them as an HParams experiment
        """Log params."""
        self._params.update(params)

    def log_artifact(self, local_path: str) -> None:
        # TensorBoard doesn't support artifacts; fallback to local copy
        """Log artifact."""
        import shutil

        dest = os.path.join(self.writer.log_dir, os.path.basename(local_path))
        shutil.copy2(local_path, dest)

    def finalize(self, result: dict[str, Any] | None = None) -> None:
        """Finalize."""
        from torch.utils.tensorboard.summary import hparams

        if self._params or result:
            combined = {**self._params, **(result or {})}
            exp, ssi, sei = hparams(combined, {})
            self.writer.file_writer.add_summary(exp)
            self.writer.file_writer.add_summary(ssi)
            self.writer.file_writer.add_summary(sei)
            self.writer.flush()

    def close(self) -> None:
        """Close."""
        self.writer.close()


# ----------------------------------------------------------------------
# Weights & Biases backend
# ----------------------------------------------------------------------
class WandBTracker(BaseTracker):
    """Logs to Weights & Biases."""

    def __init__(
        self,
        project: str,
        experiment_name: str,
        config_dict: dict[str, Any] | None = None,
    ) -> None:
        import wandb

        self.run = wandb.init(project=project, name=experiment_name, config=config_dict)

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log metric."""
        self.run.log({name: value, "step": step} if step is not None else {name: value})

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params."""
        self.run.config.update(params)

    def log_artifact(self, local_path: str) -> None:
        """Log artifact."""
        artifact = self.run.Artifact(name=os.path.basename(local_path), type="model")
        artifact.add_file(local_path)
        self.run.log_artifact(artifact)

    def finalize(self, result: dict[str, Any] | None = None) -> None:
        """Finalize."""
        if result:
            self.run.summary.update(result)

    def close(self) -> None:
        """Close."""
        self.run.finish()


# ----------------------------------------------------------------------
# MLflow backend
# ----------------------------------------------------------------------
class MLflowTracker(BaseTracker):
    """Logs to MLflow."""

    def __init__(self, experiment_name: str, tracking_uri: str | None = None) -> None:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log metric."""
        import mlflow

        mlflow.log_metric(name, value, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params."""
        import mlflow

        mlflow.log_params(params)

    def log_artifact(self, local_path: str) -> None:
        """Log artifact."""
        import mlflow

        mlflow.log_artifact(local_path)

    def finalize(self, result: dict[str, Any] | None = None) -> None:
        """Finalize."""
        if result:
            import mlflow

            mlflow.log_metrics(result)

    def close(self) -> None:
        """Close."""
        import mlflow

        mlflow.end_run()
