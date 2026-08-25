# Tracking Subsystem (`Codes/framework/tracking`)

The `tracking` subsystem provides an extensible telemetry and experiment tracking interface with support for multiple MLOps backends and automated training callbacks.

---

## 🏗 Architecture Overview

- **Unified Experiment Tracker (`tracker.py`)**: High-level `ExperimentTracker` managing run lifecycles, parameter logging, metric series recording, and artifact uploads.
- **Pluggable Backends (`backends.py`)**: Backend implementations for popular tracking engines:
  - **Weights & Biases (`WandbTracker`)**: Live metric charting, system resource telemetry, and artifact tracking.
  - **MLflow (`MLflowTracker`)**: Parameter logging, metric history, and model registry registration.
  - **Console / File (`ConsoleTracker`)**: Lightweight terminal progress logging and JSONL metric recording.
- **Training Callbacks (`callbacks.py`)**: Automated callbacks hooking into training loops to log step loss, learning rate, GPU memory allocation, and token throughput (TFLOPs).

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `tracker.py` | Central `ExperimentTracker` and `TRACKER_REGISTRY`. |
| `backends.py` | Concrete tracker backends (W&B, MLflow, TensorBoard, Console). |
| `callbacks.py` | PyTorch and HuggingFace training callbacks for automated logging. |

---

## 🚀 Quick Start & Usage

### 1. Initializing Experiment Tracker
```python
from framework.tracking.tracker import ExperimentTracker

tracker = ExperimentTracker(backend="console", project_name="svd_attention_research")
tracker.init_run(run_name="distilbert_svd_rank32", config={"rank": 32, "lr": 2e-5})

# Log metrics during training
for epoch in range(3):
    tracker.log_metrics({"train_loss": 0.35 - epoch * 0.05, "eval_acc": 0.82 + epoch * 0.03}, step=epoch)

tracker.finish_run()
```

### 2. Using Tracking Callbacks with HuggingFace Trainer
```python
from framework.tracking.callbacks import MetricTrackingCallback

callback = MetricTrackingCallback(tracker=tracker)
# Pass callback to trainer
```
