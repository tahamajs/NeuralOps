"""
Unified ExperimentTracker – select a backend via string or TrackingBackend enum.
Provides a single, consistent API: log_metric, log_params, log_artifact, finalize.
"""

from typing import Any

from .backends import (
    LOCAL_BACKEND,
    MLFLOW_BACKEND,
    TENSORBOARD_BACKEND,
    WANDB_BACKEND,
    LocalTracker,
    MLflowTracker,
    TensorBoardTracker,
    WandBTracker,
)


class ExperimentTracker:
    """
    A facade that delegates to the chosen tracking backend.

    Usage:
        tracker = ExperimentTracker(backend="wandb", project="my_project", experiment_name="run1")
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
        backend: str = LOCAL_BACKEND,
        experiment_name: str = "experiment",
        log_dir: str = "./experiments/logs",
        project: str | None = None,
        config_dict: dict[str, Any] | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        """
        Args:
            backend: "local", "tensorboard", "wandb", or "mlflow".
            experiment_name: Name of the experiment / run.
            log_dir: Directory for local / tensorboard logs.
            project: Project name (for WandB / MLflow).
            config_dict: Hyperparameter dictionary (for WandB).
            tracking_uri: MLflow tracking URI.
        """
        if hasattr(backend, "value"):
            backend = str(backend.value)
        elif not isinstance(backend, str):
            backend = str(backend)
        backend = backend.lower()
        if backend not in self.BACKEND_MAP:
            raise ValueError(
                f"Unknown backend '{backend}'. Available: {list(self.BACKEND_MAP.keys())}"
            )

        self.backend_name = backend
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.project = project
        self.config_dict = config_dict or {}
        self.tracking_uri = tracking_uri

        # Instantiate the backend
        if backend == LOCAL_BACKEND:
            self._tracker = LocalTracker(
                log_dir=log_dir, experiment_name=experiment_name
            )
        elif backend == TENSORBOARD_BACKEND:
            self._tracker = TensorBoardTracker(
                log_dir=log_dir, experiment_name=experiment_name
            )
        elif backend == WANDB_BACKEND:
            self._tracker = WandBTracker(
                project=project or "default",
                experiment_name=experiment_name,
                config_dict=self.config_dict,
            )
        elif backend == MLFLOW_BACKEND:
            self._tracker = MLflowTracker(
                experiment_name=experiment_name, tracking_uri=tracking_uri
            )
        else:
            raise NotImplementedError(f"Backend '{backend}' not implemented")

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a single metric (scalar)."""
        self._tracker.log_metric(name, value, step)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Log multiple metrics at once."""
        for k, v in metrics.items():
            self._tracker.log_metric(k, v, step)

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter."""
        self._tracker.log_params({key: value})

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple hyperparameters."""
        self._tracker.log_params(params)

    def log_artifact(self, local_path: str) -> None:
        """Save or upload a file (model checkpoint, plot, etc.)."""
        self._tracker.log_artifact(local_path)

    def finalize(self, result: dict[str, Any] | None = None) -> None:
        """
        Mark the run as completed and store final results.
        After calling this, the tracker should not be used again.
        """
        self._tracker.finalize(result)
        self._tracker.close()

    def close(self) -> None:
        """Close the backend (useful for cleaning up without final results)."""
        self._tracker.close()

    def get_callback(self):
        """Return a transformers-compatible TrainerCallback for experiment tracking."""
        from transformers import TrainerCallback

        class TrackingCallback(TrainerCallback):
            def __init__(self, tracker_instance):
                self.tracker = tracker_instance

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    step = state.global_step if state else None
                    self.tracker.log_metrics(logs, step=step)

        return TrackingCallback(self)
