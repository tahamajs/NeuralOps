# `framework/results_saver` – Results Persistence & Distribution

The **results_saver** module provides a single, configuration‑driven orchestrator for **saving, exporting, plotting, and distributing** all artifacts generated during a training run.  
It integrates seamlessly with the rest of the framework (analysis, tracking, cloud backends) and is designed to be called at the end of any training job.

---

## Features

- **Save model** – full HuggingFace model, optimizer state, checkpoints.
- **Generate comprehensive plots** – loss curves, accuracy, learning rate schedule, GPU memory, gradient flow, confusion matrix, radar chart, etc.
- **Export to ONNX / TorchScript** (and other formats via pluggable exporters).
- **Push to cloud storage** – S3, GCP, Azure, HuggingFace Hub, lakeFS, Pachyderm.
- **Track metrics** in experiment trackers – Weights & Biases, MLflow, Neptune, Comet, Aim, ZenML.
- **Git commit / DVC tracking** – version control for your results.
- **Create GitHub Releases** – with attached artifacts.
- **Send notifications** – Slack, email, custom webhooks.
- **Model registries** – push to HuggingFace Hub, OCI registries, etc.
- **Metadata stores** – OpenMetadata integration.
- **All features are optional** and configured via a single `SaveConfig` dataclass.

---

## Installation

The base saver has no additional dependencies beyond the framework.  
Backends require their respective libraries (see [Backend Dependencies](#backend-dependencies)).

```bash
pip install -e .
```

---

## Quick Start

```python
from framework.results_saver import ResultsSaver, SaveConfig

# After training, you have:
# model, config, history (list of dicts), final_metrics (dict)

saver = ResultsSaver(
    run_id="my_experiment_20250301",
    base_dir="./runs",
    config=SaveConfig(
        save_model=True,
        save_plots=True,
        storage_backends=["s3"],
        model_export_formats=["onnx"],
        experiment_trackers=["wandb"],
        cloud_options=dict(aws_s3_bucket="my-bucket"),
    ),
)

saver.save_all(model, config, history, final_metrics, tokenizer, input_example)
saver.push_all()          # upload to S3, push to W&B, etc.
```

After this call, the directory `./runs/my_experiment_20250301/` will contain:

```
my_experiment_20250301/
├── artifacts/
│   ├── config.json
│   ├── final_metrics.json
│   ├── history.json
│   ├── model/            (if save_model=True)
│   └── ...
├── logs/
│   └── metrics.jsonl
├── plots/                (if save_plots=True)
│   ├── loss_curve.png
│   └── ...
├── checkpoints/
│   └── final/
├── results.tar.gz        (if archive)
└── model_card.md
```

---

## Configuration – `SaveConfig`

All aspects of the saver are controlled by a `SaveConfig` dataclass. You can pass a dictionary or an instance.

### Core Switches

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save_model` | `bool` | `True` | Save the final model |
| `save_checkpoints` | `bool` | `True` | Save model checkpoints |
| `save_optimizer` | `bool` | `False` | Save optimizer state |
| `save_logs` | `bool` | `True` | Save training logs |
| `save_plots` | `bool` | `True` | Generate and save plots |
| `save_artifacts` | `bool` | `True` | Save all artifacts |
| `create_model_version` | `bool` | `True` | Create a version in the DB |
| `auto_save_if_better_than` | `float` or `None` | `None` | Only save heavy artifacts if metric ≥ threshold |

### Plot Options (`plot_options` dict)

| Key | Default | Description |
|-----|---------|-------------|
| `loss_curve` | `True` | Plot training loss |
| `accuracy_curve` | `True` | Plot validation accuracy |
| `lr_schedule` | `True` | Plot learning rate |
| `training_time` | `True` | Plot training time vs steps |
| `gpu_memory` | `False` | Plot GPU memory usage (if logged) |
| `gradient_flow` | `False` | Plot gradient norms (if logged) |
| `confusion_matrix` | `False` | Plot confusion matrix (requires eval predictions) |
| `radar_chart` | `False` | Plot radar chart of final metrics |
| `rank_allocation` | `False` | Plot SVD rank allocation (if saved) |
| `attention_heatmap` | `False` | Plot attention heatmap (requires model) |
| `loss_smooth` | `0.6` | Smoothing factor for loss curve |
| `figsize` | `[10, 6]` | Default figure size |
| `dpi` | `150` | DPI for saved plots |
| `class_names` | `None` | List of class names for confusion matrix |

### Storage Backends

| Backend | Key | Dependencies |
|---------|-----|--------------|
| Local filesystem | `"local"` | None |
| Amazon S3 | `"s3"` | `boto3` |
| Google Cloud Storage | `"gcp"` | `google-cloud-storage` |
| Azure Blob Storage | `"azure"` | `azure-storage-blob` |
| HuggingFace Hub | `"hf_hub"` | `huggingface_hub` |
| Git | `"git"` | `gitpython` |
| DVC | `"dvc"` | `dvc` CLI |
| Docker / OCI | `"oci"` | `oras` CLI |
| lakeFS | `"lakefs"` | `lakefs` |
| Pachyderm | `"pachyderm"` | `python-pachyderm` |
| Custom webhook | `"custom_webhook"` | `requests` |

Set `storage_backends` to a list of strings, e.g., `["local", "s3", "hf_hub"]`.

### Experiment Trackers

| Tracker | Key | Dependencies |
|---------|-----|--------------|
| Weights & Biases | `"wandb"` | `wandb` |
| MLflow | `"mlflow"` | `mlflow` |
| Neptune | `"neptune"` | `neptune` |
| Comet | `"comet"` | `comet_ml` |
| AimStack | `"aim"` | `aim` |
| ZenML | `"zenml"` | `zenml` |

Set `experiment_trackers` to a list of strings, e.g., `["wandb", "mlflow"]`.

### Model Export Formats

| Format | Key | Dependencies |
|--------|-----|--------------|
| ONNX | `"onnx"` | `onnx` (PyTorch built‑in) |
| TorchScript | `"torchscript"` | PyTorch built‑in |
| TensorFlow Lite | `"tflite"` | TensorFlow (not implemented) |
| TensorRT | `"tensorrt"` | `trtexec` CLI (not implemented) |

Set `model_export_formats` to a list of strings, e.g., `["onnx", "torchscript"]`.

### Model Registries

| Registry | Key | Dependencies |
|----------|-----|--------------|
| HuggingFace Hub | `"hf_hub"` | `huggingface_hub` |
| OCI (Docker) | `"oci"` | `oras` CLI |

Set `model_registries` to a list of strings.

### Metadata Store

| Store | Key | Dependencies |
|-------|-----|--------------|
| OpenMetadata | `"openmetadata"` | `requests` |

Set `metadata_store` to `"openmetadata"` if desired.

### Git / DVC Options

- `git_options`: dict with keys like `"commit_results"`, `"push_to_remote"`, `"tag_pattern"`, `"use_git_lfs"`.
- `dvc_options`: dict with `"track_model"`, `"auto_push"`.

### Notifications

- `notification`: dict with keys `"on_success"` (list of channels), `"on_failure"`, `"slack_webhook"`.

### Cloud Options

- `cloud_options`: dict containing credentials and endpoints for S3 (`aws_s3_bucket`), GCP (`gcp_bucket`), Azure (`azure_conn_str`, `azure_container`), etc.

---

## Using with the Training Pipeline

The saver is designed to be called at the end of a training job.  
If you use the framework’s `ResultsManager`, you can add a `save_with_saver` method (or directly call the saver).

### Example with `ResultsManager`

```python
from framework.training import ResultsManager
from framework.results_saver import ResultsSaver, SaveConfig

results = ResultsManager("my_experiment", base_dir="./runs")
# ... train your model ...

save_cfg = SaveConfig(
    save_model=True,
    save_plots=True,
    storage_backends=["s3"],
    experiment_trackers=["wandb"],
    cloud_options=dict(aws_s3_bucket="my-bucket"),
)

saver = ResultsSaver(
    run_id=results.run_dir.name,
    base_dir=results.base_dir,
    config=save_cfg,
)
saver.save_all(model, config, results.history, final_metrics, tokenizer)
saver.push_all()
```

Alternatively, you can inherit the `ResultsManager` and override the finalization step.

---

## Integration with Other Framework Components

- **Plots**: delegates to `framework.analysis.plotting`.
- **Unified Tracking**: if the `framework.tracking` module is installed, it can be used as the primary tracker instead of the legacy per‑backend trackers.
- **Model Serialisation**: uses the standard HuggingFace `save_pretrained` (and `UnifiedTransformerModel` if wrapped).
- **Architecture Detection**: uses the core architecture module to save model cards with proper naming.

---

## Extending the Saver

### Adding a New Storage Backend

1. Create a file `backends/my_backend.py` with a function `push(run_dir, cfg)`.
2. The `cfg` object is a `SaveConfig` instance; access options via `cfg.cloud_options`, etc.
3. No need to register – simply add the name to `storage_backends` in the config.

### Adding a New Experiment Tracker

1. Create `backends/my_tracker_tracker.py` with a function `track(run_id, run_dir, metrics, params, cfg)`.
2. Add the name to `experiment_trackers`.

### Adding a New Export Format

1. In `backends/exports.py`, use the `@_register` decorator on a function that takes `(model, input_example, output_dir)`.
2. The format name is automatically added to `model_export_formats`.

---

## Dependencies

The saver core depends only on:

- `torch` (for ONNX/TorchScript export and checkpoint saving)
- `transformers` (for `save_pretrained`)
- `matplotlib`, `seaborn`, `numpy` (for plotting)
- `PyYAML` (optional, for config)

Backend dependencies are listed in the respective sections above. All are optional and will only be imported when the backend is actually used. If a library is missing, a warning is logged and the operation is skipped.

---

## Complete Example

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.results_saver import ResultsSaver, SaveConfig

# Simulate a training run
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
history = [{"loss": 0.5, "eval_accuracy": 0.8, "step": 0}]
final_metrics = {"eval_accuracy": 0.8, "eval_loss": 0.4}
config = {"model_name": "distilbert", "task_name": "sst2", "attention_type": "full", "rank": 64}
input_example = torch.randint(0, 1000, (1, 32))

# Create saver
saver = ResultsSaver(
    run_id="demo_run",
    base_dir="./runs",
    config=SaveConfig(
        save_model=True,
        save_plots=True,
        model_export_formats=["onnx"],
        storage_backends=["local"],
        experiment_trackers=["wandb"],  # if wandb installed
    ),
)

# Save everything
saver.save_all(model, config, history, final_metrics, tokenizer, input_example)
# Push to all configured backends (local is no-op, wandb uploads metrics)
saver.push_all()
```

---

The **results_saver** module ensures that your experiments are **reproducible, shareable, and production‑ready**—whether you’re running on a laptop or in a cloud pipeline.