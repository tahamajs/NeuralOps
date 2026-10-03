Module neuralops.tracking.tracker
=================================
Unified ExperimentTracker – select a backend via string or TrackingBackend enum.
Provides a single, consistent API: log_metric, log_params, log_artifact, finalize.

Classes
-------

`ExperimentTracker(backend: str = 'local', experiment_name: str = 'experiment', log_dir: str = './experiments/logs', project: str | None = None, config_dict: dict[str, typing.Any] | None = None, tracking_uri: str | None = None)`
:   A facade that delegates to the chosen tracking backend.
    
    Usage:
        tracker = ExperimentTracker(backend="wandb", project="my_project", experiment_name="run1")
        tracker.log_metric("loss", 0.5, step=100)
        tracker.finalize({"final_accuracy": 0.93})
    
    Args:
        backend: "local", "tensorboard", "wandb", or "mlflow".
        experiment_name: Name of the experiment / run.
        log_dir: Directory for local / tensorboard logs.
        project: Project name (for WandB / MLflow).
        config_dict: Hyperparameter dictionary (for WandB).
        tracking_uri: MLflow tracking URI.

    ### Class variables

    `BACKEND_MAP`
    :   The type of the None singleton.

    ### Methods

    `close(self) ‑> None`
    :   Close the backend (useful for cleaning up without final results).

    `finalize(self, result: dict[str, typing.Any] | None = None) ‑> None`
    :   Mark the run as completed and store final results.
        After calling this, the tracker should not be used again.

    `get_callback(self)`
    :   Return a transformers-compatible TrainerCallback for experiment tracking.

    `log_artifact(self, local_path: str) ‑> None`
    :   Save or upload a file (model checkpoint, plot, etc.).

    `log_metric(self, name: str, value: float, step: int | None = None) ‑> None`
    :   Log a single metric (scalar).

    `log_metrics(self, metrics: dict[str, float], step: int | None = None) ‑> None`
    :   Log multiple metrics at once.

    `log_param(self, key: str, value: Any) ‑> None`
    :   Log a single hyperparameter.

    `log_params(self, params: dict[str, typing.Any]) ‑> None`
    :   Log multiple hyperparameters.