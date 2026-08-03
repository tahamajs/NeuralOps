# `framework/tracking` – Experiment Tracking

The **tracking** module provides a **pluggable experiment tracking** system for logging metrics, hyperparameters, and artifacts during training and evaluation.  
It is designed to be **backend‑agnostic** – you write the same code regardless of whether you’re logging to a local JSON file, TensorBoard, Weights & Biases, or MLflow.

---

## Installation

All tracking dependencies are optional except for the local backend. Install the ones you need:

```bash
pip install tensorboard        # for TensorBoard
pip install wandb              # for Weights & Biases
pip install mlflow             # for MLflow
```

The core tracker requires only Python ≥ 3.10 and `torch` (for the TensorBoard backend). The rest of the framework uses the tracker via the `ResultsManager`, so no additional installs are needed if you already have the framework.

---

## Quick Start

```python
from framework.tracking import ExperimentTracker, TrackingBackend

# 1. Create a tracker (default: local JSON)
tracker = ExperimentTracker(
    backend="local",                     # or "tensorboard", "wandb", "mlflow"
    log_dir="./my_experiment",
    experiment_name="test_run",
)

# 2. Log a single metric
tracker.log_metric("loss", 0.5, step=100)

# 3. Log multiple metrics at once
tracker.log_metrics({"accuracy": 0.92, "f1": 0.88}, step=100)

# 4. Log hyperparameters
tracker.log_params({"learning_rate": 2e-5, "batch_size": 16})

# 5. Log a file (e.g., model checkpoint, plot)
tracker.log_artifact("./model_checkpoint.pt")

# 6. Mark the run as completed and store final results
tracker.finalize({"final_accuracy": 0.93})
```

---

## Supported Backends

| Backend | Key | Dependencies | Notes |
|---------|-----|--------------|-------|
| **Local JSON** | `"local"` | None | Writes metrics to a JSON lines file, copies artifacts to the log directory. |
| **TensorBoard** | `"tensorboard"` | `tensorboard` | Writes scalars to event files. Supports hyperparameter logging. |
| **Weights & Biases** | `"wandb"` | `wandb` | Full W&B integration with config, artifacts, and summaries. |
| **MLflow** | `"mlflow"` | `mlflow` | Logs metrics, params, artifacts to an MLflow tracking server. |

Select the backend via the `backend` parameter (string). The tracker will handle backend‑specific initialisation and logging automatically.

---

## API Reference

### `ExperimentTracker`

The main class you interact with. It delegates to the chosen backend.

| Method | Description |
|--------|-------------|
| `log_metric(name, value, step=None)` | Log a single scalar metric. |
| `log_metrics(metrics, step=None)` | Log multiple scalars at once. |
| `log_param(key, value)` | Log a single hyperparameter. |
| `log_params(params)` | Log a dictionary of hyperparameters. |
| `log_artifact(local_path)` | Save / upload a file. |
| `finalize(result=None)` | Finalize the run and store final results. |
| `close()` | Close the backend without final results. |

All methods are safe to call multiple times; the backend handles the rest.

### `TrackingBackend` (Enum)

- `LOCAL` – local JSON
- `TENSORBOARD` – TensorBoard event files
- `WANDB` – Weights & Biases
- `MLFLOW` – MLflow

### `TrackerCallback` (HuggingFace Trainer callback)

Automatically logs `loss`, `learning_rate`, `epoch`, and evaluation metrics to the tracker during training.

```python
from framework.tracking import TrackerCallback
trainer.add_callback(TrackerCallback(tracker))
```

---

## Configuration

All configuration is passed as keyword arguments to `ExperimentTracker`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `"local"` | Backend to use (`"local"`, `"tensorboard"`, `"wandb"`, `"mlflow"`) |
| `experiment_name` | `"experiment"` | Name of the experiment / run. |
| `log_dir` | `"./experiments/logs"` | Directory for local / tensorboard logs. |
| `project` | `None` | Project name (WandB) or experiment name (MLflow). |
| `config_dict` | `{}` | Hyperparameter dictionary (for WandB config). |
| `tracking_uri` | `None` | MLflow tracking URI (e.g., SQLite path or HTTP endpoint). |

---

## Integration with the Training Module

### 1. Using `TrackerCallback`

```python
from framework.tracking import ExperimentTracker, TrackerCallback

tracker = ExperimentTracker(backend="wandb", project="my_project", experiment_name="run1")
trainer.add_callback(TrackerCallback(tracker))
```

The callback automatically forwards all scalar metrics logged by the HuggingFace Trainer to the tracker.

### 2. Using with `ResultsManager`

The `ResultsManager` (in `framework/training`) can be extended to use a tracker. Add this method to `ResultsManager`:

```python
def enable_tracking(self, backend="local", **kwargs):
    from framework.tracking import ExperimentTracker
    self.tracker = ExperimentTracker(
        backend=backend,
        log_dir=str(self.log_dir),
        experiment_name=self.experiment_name,
        **kwargs
    )
    original_log = self.log_metrics
    def new_log(metrics, step=None):
        original_log(metrics, step)
        if self.tracker:
            self.tracker.log_metrics(metrics, step)
    self.log_metrics = new_log
```

Then use it:

```python
results = ResultsManager("my_experiment")
results.enable_tracking(backend="tensorboard")
# all subsequent metrics logged via results.log_metrics() will also go to TensorBoard
```

---

## Extending with a New Backend

1. Create a class that inherits from `BaseTracker` (in `framework/tracking/backends.py`).
2. Implement the five required methods: `log_metric`, `log_params`, `log_artifact`, `finalize`, `close`.
3. Add the new backend key to the `BACKEND_MAP` dictionary in `tracker.py`.
4. (Optional) Add an entry in the `TrackingBackend` enum.

The existing backends serve as templates and are heavily commented.

---

## File Structure

```
framework/tracking/
├── __init__.py          # Public API exports
├── backends.py          # Base class + Local / TensorBoard / WandB / MLflow backends
├── tracker.py           # Main ExperimentTracker (facade)
├── callbacks.py         # HuggingFace Trainer callback
└── README.md            # This file
```

---

## Examples

### Local JSON logging

```python
tracker = ExperimentTracker(backend="local", log_dir="./runs", experiment_name="debug")
for step in range(100):
    tracker.log_metric("loss", 1.0 / (step + 1), step=step)
tracker.finalize({"final_loss": 0.01})
```

After the run, the directory `./runs/` contains:

- `debug_20250301_120000_metrics.jsonl` – all metrics
- `debug_20250301_120000_params.json` – hyperparameters
- `debug_summary.json` – final results

### TensorBoard

```python
tracker = ExperimentTracker(backend="tensorboard", log_dir="./runs", experiment_name="train")
tracker.log_metrics({"accuracy": 0.92, "loss": 0.3}, step=100)
tracker.close()
```

Launch TensorBoard: `tensorboard --logdir ./runs`

### Weights & Biases

```python
tracker = ExperimentTracker(backend="wandb", project="my_project", experiment_name="run1",
                            config_dict={"lr": 2e-5, "batch_size": 16})
tracker.log_metric("accuracy", 0.92, step=100)
tracker.finalize()
```

### MLflow

```python
tracker = ExperimentTracker(backend="mlflow", tracking_uri="sqlite:///mlflow.db",
                            experiment_name="my_experiment")
tracker.log_metric("loss", 0.5, step=10)
tracker.log_artifact("./model.pth")
tracker.finalize()
```

---

## Dependencies

- `torch` – used only by the TensorBoard backend
- `tensorboard` – optional, for TensorBoard backend
- `wandb` – optional, for W&B backend
- `mlflow` – optional, for MLflow backend
- `psutil` – optional, for artifact copying in the local backend (uses `shutil` from stdlib)

The local backend works with Python standard library only.

---

## Contributing

If you add a new tracking backend, follow the pattern in `backends.py` and submit a pull request. Ensure that the backend gracefully handles missing dependencies (raise an `ImportError` at construction time, not at import time).
