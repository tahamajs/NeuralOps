"""
Unified MLOps experiment tracking and telemetry interface for NeuralOps.

Supports:
  - Weights & Biases (`wandb`)
  - MLflow (`mlflow`)
  - TensorBoard (`torch.utils.tensorboard`)
  - Local disk loggers (`jsonl`, `csv`, `parquet`)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("NeuralOps.MLOps")


class BaseTracker(ABC):
    """Abstract base class for experiment trackers."""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass


class LocalTracker(BaseTracker):
    """Local JSONL / CSV experiment logger."""

    def __init__(self, output_dir: Union[str, Path], run_name: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or f"run_{int(time.time())}"
        self.jsonl_path = self.output_dir / f"{self.run_name}_metrics.jsonl"
        self.params_path = self.output_dir / f"{self.run_name}_params.json"

    def log_params(self, params: Dict[str, Any]) -> None:
        with open(self.params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, default=str)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        record = dict(metrics)
        record["_timestamp"] = time.time()
        if step is not None:
            record["_step"] = step
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def finish(self) -> None:
        pass


class WandbTracker(BaseTracker):
    """Weights & Biases integration."""

    def __init__(self, project_name: str = "neuralops", run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.available = False
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=project_name, name=run_name, config=config or {}, reinit=True)
            self.available = True
        except Exception as e:
            logger.warning("Failed to initialize Weights & Biases: %s", e)

    def log_params(self, params: Dict[str, Any]) -> None:
        if self.available:
            self.wandb.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.available:
            self.wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.available and self.run:
            self.run.finish()


class MLflowTracker(BaseTracker):
    """MLflow experiment tracking integration."""

    def __init__(self, experiment_name: str = "neuralops", run_name: Optional[str] = None):
        self.available = False
        try:
            import mlflow
            self.mlflow = mlflow
            mlflow.set_experiment(experiment_name)
            self.active_run = mlflow.start_run(run_name=run_name)
            self.available = True
        except Exception as e:
            logger.warning("Failed to initialize MLflow: %s", e)

    def log_params(self, params: Dict[str, Any]) -> None:
        if self.available:
            flat_params = {k: str(v) for k, v in params.items()}
            self.mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.available:
            self.mlflow.log_metrics(metrics, step=step)

    def finish(self) -> None:
        if self.available:
            self.mlflow.end_run()


class TensorBoardTracker(BaseTracker):
    """PyTorch TensorBoard summary writer."""

    def __init__(self, log_dir: Union[str, Path]):
        self.available = False
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(log_dir))
            self.available = True
        except Exception as e:
            logger.warning("Failed to initialize TensorBoard SummaryWriter: %s", e)

    def log_params(self, params: Dict[str, Any]) -> None:
        if self.available:
            text_summary = "\n".join([f"**{k}**: {v}" for k, v in params.items()])
            self.writer.add_text("hyperparameters", text_summary, global_step=0)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.available:
            s = step if step is not None else 0
            for tag, val in metrics.items():
                if isinstance(val, (int, float)):
                    self.writer.add_scalar(tag, float(val), global_step=s)

    def finish(self) -> None:
        if self.available:
            self.writer.flush()
            self.writer.close()


class CompositeTracker(BaseTracker):
    """Multiplexes logging across multiple trackers."""

    def __init__(self, trackers: List[BaseTracker]):
        self.trackers = trackers

    def log_params(self, params: Dict[str, Any]) -> None:
        for t in self.trackers:
            t.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for t in self.trackers:
            t.log_metrics(metrics, step=step)

    def finish(self) -> None:
        for t in self.trackers:
            t.finish()


def create_tracker(
    tracker_type: str = "none",
    project_name: str = "neuralops",
    run_name: Optional[str] = None,
    output_dir: Union[str, Path] = "results",
    config: Optional[Dict[str, Any]] = None,
) -> BaseTracker:
    """Factory helper to build requested tracker(s)."""
    t_type = (tracker_type or "none").lower()
    trackers: List[BaseTracker] = [LocalTracker(output_dir=output_dir, run_name=run_name)]

    if t_type == "wandb" or t_type == "all":
        trackers.append(WandbTracker(project_name=project_name, run_name=run_name, config=config))
    if t_type == "mlflow" or t_type == "all":
        trackers.append(MLflowTracker(experiment_name=project_name, run_name=run_name))
    if t_type == "tensorboard" or t_type == "all":
        tb_dir = Path(output_dir) / "tensorboard" / (run_name or "run")
        trackers.append(TensorBoardTracker(log_dir=tb_dir))

    if len(trackers) == 1:
        return trackers[0]
    return CompositeTracker(trackers)


__all__ = [
    "BaseTracker",
    "LocalTracker",
    "WandbTracker",
    "MLflowTracker",
    "TensorBoardTracker",
    "CompositeTracker",
    "create_tracker",
]
