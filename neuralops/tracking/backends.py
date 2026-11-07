#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Experiment Tracking Backends

Provides adapters for logging metrics to Weights & Biases, MLflow, TensorBoard, and local files.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TrackingBackend(Enum):
    LOCAL = "local"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    MLFLOW = "mlflow"


LOCAL_BACKEND = TrackingBackend.LOCAL.value
TENSORBOARD_BACKEND = TrackingBackend.TENSORBOARD.value
WANDB_BACKEND = TrackingBackend.WANDB.value
MLFLOW_BACKEND = TrackingBackend.MLFLOW.value


class BaseTracker(ABC):
    """Every backend must implement these methods."""

    @abstractmethod
    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log metric."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once."""
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log params."""

    @abstractmethod
    def log_artifact(self, local_path: str) -> None:
        """Log artifact."""

    @abstractmethod
    def finalize(self, result: dict[str, Any] | None = None) -> None:
        """Finalize."""

    @abstractmethod
    def close(self) -> None:
        """Close."""


class LocalTracker(BaseTracker):
    """
    Writes metrics to metrics.jsonl and parameters to params.json inside log_dir.
    Artifacts are stored in log_dir (or log_dir/artifacts if present).
    """

    def __init__(self, log_dir: str, experiment_name: str = "experiment") -> None:
        self.log_dir = str(log_dir)
        self.experiment_name = experiment_name
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.log_dir, "metrics.jsonl")
        self.params_file = os.path.join(self.log_dir, "params.json")
        self.metrics: list[dict[str, Any]] = []
        self.params: dict[str, Any] = {}

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log metric."""
        entry = {
            "name": name,
            "value": float(value),
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }
        self.metrics.append(entry)
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once."""
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params."""
        self.params.update(params)
        with open(self.params_file, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=2)
            f.flush()

    def log_artifact(self, local_path: str) -> None:
        """Log artifact."""
        import shutil

        artifacts_dir = os.path.join(self.log_dir, "artifacts")
        dest_dir = artifacts_dir if os.path.isdir(artifacts_dir) else self.log_dir
        dest = os.path.join(dest_dir, os.path.basename(local_path))
        shutil.copy2(local_path, dest)

    def finalize(self, result: dict[str, Any] | None = None) -> None:
        """Finalize."""
        if result:
            summary_file = os.path.join(self.log_dir, "summary.json")
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                f.flush()

    def close(self) -> None:
        """Close."""


class TensorBoardTracker(BaseTracker):
    """Writes scalar metrics to TensorBoard event files inside log_dir/tensorboard/."""

    def __init__(self, log_dir: str, experiment_name: str = "experiment") -> None:
        log_dir_str = str(log_dir)
        if os.path.basename(os.path.normpath(log_dir_str)) == "tensorboard":
            tb_dir = log_dir_str
        else:
            tb_dir = os.path.join(log_dir_str, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        self.log_dir = tb_dir
        self.experiment_name = experiment_name
        self._params: dict[str, Any] = {}

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=tb_dir)
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning("TensorBoard unavailable (%s); events will not be written to disk.", e)
            self.writer = None

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log metric."""
        if self.writer is not None:
            global_step = step if step is not None else 0
            self.writer.add_scalar(name, float(value), global_step=global_step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once."""
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params."""
        self._params.update(params)

    def log_artifact(self, local_path: str) -> None:
        """Log artifact."""
        import shutil

        dest = os.path.join(self.log_dir, os.path.basename(local_path))
        shutil.copy2(local_path, dest)

    def finalize(self, result: dict[str, Any] | None = None) -> None:
        """Finalize."""
        if self.writer is not None and (self._params or result):
            try:
                from torch.utils.tensorboard.summary import hparams

                combined = {**self._params, **(result or {})}
                exp, ssi, sei = hparams(combined, {})
                self.writer.file_writer.add_summary(exp)
                self.writer.file_writer.add_summary(ssi)
                self.writer.file_writer.add_summary(sei)
                self.writer.flush()
            except Exception as e:
                logger.warning("Failed to record hparams in TensorBoard: %s", e)

    def close(self) -> None:
        """Close."""
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass


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

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once."""
        data = {**metrics}
        if step is not None:
            data["step"] = step
        self.run.log(data)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params."""
        self.run.config.update(params)

    def log_artifact(self, local_path: str) -> None:
        """Log artifact."""
        import wandb

        artifact = wandb.Artifact(name=os.path.basename(local_path), type="model")
        artifact.add_file(local_path)
        self.run.log_artifact(artifact)

    def finalize(self, result: dict[str, Any] | None = None) -> None:
        """Finalize."""
        if result:
            self.run.summary.update(result)

    def close(self) -> None:
        """Close."""
        self.run.finish()


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

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once."""
        import mlflow

        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)

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
