#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Experiment Tracker

Manages experiment lifecycles, run parameters, and metric artifact persistence.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any, Sequence

from .backends import (
    LOCAL_BACKEND,
    MLFLOW_BACKEND,
    TENSORBOARD_BACKEND,
    WANDB_BACKEND,
    BaseTracker,
    LocalTracker,
    MLflowTracker,
    TensorBoardTracker,
    WandBTracker,
)


class ExperimentTracker:
    """
    A facade that delegates to one or more tracking backends.

    Usage:
        tracker = ExperimentTracker(log_dir=run_dir)
        tracker.log_metric("loss", 0.5, step=100)
        tracker.finalize({"final_accuracy": 0.93})
    """

    BACKEND_MAP = {
        LOCAL_BACKEND: LocalTracker,
        TENSORBOARD_BACKEND: TensorBoardTracker,
        WANDB_BACKEND: WandBTracker,
        MLFLOW_BACKEND: MLflowTracker,
    }

    def __init__(
        self,
        backend: Any = LOCAL_BACKEND,
        experiment_name: Any = "experiment",
        log_dir: Any = "./experiments/logs",
        project: str | None = None,
        config_dict: dict[str, Any] | None = None,
        tracking_uri: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            backend: "local", "tensorboard", "wandb", "mlflow", or a list/comma-separated string.
                     If a Path or object with `run_dir` is passed, it is treated as `log_dir`.
            experiment_name: Name of the experiment / run, or a config dict if passed positionally.
            log_dir: Directory for local / tensorboard logs (e.g. run_dir from RunDirectoryContext).
            project: Project name (for WandB / MLflow).
            config_dict: Hyperparameter or tracking config dictionary.
            tracking_uri: MLflow tracking URI.
        """
        # Resolve log_dir and backend flexibly
        resolved_log_dir = log_dir
        resolved_backend = backend

        if hasattr(backend, "run_dir"):
            resolved_log_dir = backend.run_dir
            resolved_backend = kwargs.get("backend", LOCAL_BACKEND)
        elif isinstance(backend, Path):
            resolved_log_dir = str(backend)
            resolved_backend = kwargs.get("backend", LOCAL_BACKEND)
        elif isinstance(backend, str) and (
            os.path.sep in backend or "/" in backend or backend not in self.BACKEND_MAP
        ):
            if os.path.exists(backend) or "/" in backend or "\\" in backend:
                resolved_log_dir = backend
                resolved_backend = kwargs.get("backend", LOCAL_BACKEND)

        if hasattr(resolved_log_dir, "run_dir"):
            resolved_log_dir = resolved_log_dir.run_dir
        if "run_dir" in kwargs:
            resolved_log_dir = kwargs["run_dir"]

        if isinstance(experiment_name, dict):
            config_dict = experiment_name
            experiment_name = "experiment"

        self.log_dir = str(resolved_log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.experiment_name = str(experiment_name)
        self.project = project
        self.config_dict = config_dict or {}
        self.tracking_uri = tracking_uri

        # Resolve backend list
        backend_names: list[str] = []
        if isinstance(resolved_backend, (list, tuple, set)):
            backend_names = [str(b).lower() for b in resolved_backend]
        elif isinstance(resolved_backend, str):
            backend_names = [b.strip().lower() for b in resolved_backend.split(",") if b.strip()]
        elif hasattr(resolved_backend, "value"):
            backend_names = [str(resolved_backend.value).lower()]
        else:
            backend_names = [str(resolved_backend).lower()]

        # Respect config_dict flags if present
        if self.config_dict.get("enable_local_jsonl") is True and LOCAL_BACKEND not in backend_names:
            backend_names.append(LOCAL_BACKEND)
        if self.config_dict.get("enable_tensorboard") is True and TENSORBOARD_BACKEND not in backend_names:
            backend_names.append(TENSORBOARD_BACKEND)

        if not backend_names:
            backend_names = [LOCAL_BACKEND]

        self.backend_name = backend_names[0] if len(backend_names) == 1 else ",".join(backend_names)

        # Initialize backends
        self._trackers: list[BaseTracker] = []
        for name in backend_names:
            if name not in self.BACKEND_MAP:
                raise ValueError(
                    f"Unknown backend '{name}'. Available: {list(self.BACKEND_MAP.keys())}"
                )
            if name == LOCAL_BACKEND:
                self._trackers.append(
                    LocalTracker(log_dir=self.log_dir, experiment_name=self.experiment_name)
                )
            elif name == TENSORBOARD_BACKEND:
                self._trackers.append(
                    TensorBoardTracker(log_dir=self.log_dir, experiment_name=self.experiment_name)
                )
            elif name == WANDB_BACKEND:
                self._trackers.append(
                    WandBTracker(
                        project=self.project or "default",
                        experiment_name=self.experiment_name,
                        config_dict=self.config_dict,
                    )
                )
            elif name == MLFLOW_BACKEND:
                self._trackers.append(
                    MLflowTracker(
                        experiment_name=self.experiment_name,
                        tracking_uri=self.tracking_uri,
                    )
                )

        self._tracker: BaseTracker = self._trackers[0] if self._trackers else LocalTracker(self.log_dir)
        self._current_step = 0
        self._lock = threading.Lock()
        self._closed = False

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def set_step(self, step: int) -> None:
        """Set the global step index explicitly."""
        with self._lock:
            self._current_step = step

    def step(self, increment: int = 1) -> int:
        """Advance and return the active global step counter."""
        with self._lock:
            self._current_step += increment
            return self._current_step

    def get_step(self) -> int:
        """Retrieve current global step."""
        with self._lock:
            return self._current_step

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a single metric (scalar)."""
        active_step = step if step is not None else self._current_step
        with self._lock:
            for t in self._trackers:
                t.log_metric(name, value, active_step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once."""
        active_step = step if step is not None else self._current_step
        with self._lock:
            for t in self._trackers:
                t.log_metrics(metrics, step=active_step)

    def log_metrics_batch(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a batch of metrics."""
        self.log_metrics(metrics, step=step)

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter."""
        self.log_params({key: value})

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple hyperparameters."""
        for t in self._trackers:
            t.log_params(params)

    def log_artifact(self, local_path: str) -> None:
        """Save or upload a file (model checkpoint, plot, etc.)."""
        for t in self._trackers:
            t.log_artifact(local_path)

    def log_hardware_telemetry(self, step: int | None = None) -> dict[str, float]:
        """Sample and log GPU and CPU telemetry to active backends."""
        from neuralops.benchmarks.telemetry.hardware import get_hardware_telemetry

        telemetry = get_hardware_telemetry()
        key_metrics = {
            "gpu_temp_c": telemetry["gpu_temp_c"],
            "gpu_vram_used_mb": telemetry["gpu_vram_used_mb"],
            "gpu_power_w": telemetry["gpu_power_w"],
        }
        if "cpu_util_pct" in telemetry:
            key_metrics["cpu_util_pct"] = telemetry["cpu_util_pct"]
        self.log_metrics(key_metrics, step=step)
        return telemetry

    def finalize(self, result: dict[str, Any] | None = None) -> None:
        """
        Mark the run as completed and store final results.
        After calling this, the tracker backends are finalized and closed.
        """
        for t in self._trackers:
            t.finalize(result)
            t.close()

    def close(self) -> None:
        """Close the backend(s)."""
        for t in self._trackers:
            t.close()

    def get_callback(self):
        """Return a transformers-compatible TrainerCallback for experiment tracking."""
        from transformers import TrainerCallback

        class TrackingCallback(TrainerCallback):
            def __init__(self, tracker_instance: ExperimentTracker):
                """Execute __init__ operation."""
                self.tracker = tracker_instance

            def on_log(self, args, state, control, logs=None, **kwargs):
                """Execute on_log operation."""
                if logs:
                    step = state.global_step if state else None
                    self.tracker.log_metrics(logs, step=step)

        return TrackingCallback(self)
