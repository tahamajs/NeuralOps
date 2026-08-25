# Results Saver Subsystem (`Codes/framework/results_saver`)

The `results_saver` subsystem provides a multi-backend persistence engine for training metrics, evaluation results, model checkpoints, plots, and experiment artifacts.

---

## 🏗 Architecture Overview

- **Unified Persistence Orchestrator (`saver.py`)**: Central `ResultsSaver` class exposing standardized methods (`save_metrics`, `save_model_weights`, `save_config`, `save_plots`).
- **Pluggable Backends (`backends/`)**: Decorator-registered backends via `BACKEND_REGISTRY`:
  - **Local Storage (`local.py`, `exports.py`)**: JSON, JSONL, Parquet, CSV, and SQLite tabular storage.
  - **MLOps & Tracking Platforms**: MLflow (`mlflow.py`), Weights & Biases (`wandb.py`), Comet (`comet.py`), Aim (`aim.py`), Neptune (`neptune.py`), ZenML (`zenml.py`).
  - **Cloud & Object Storage**: AWS S3 (`s3.py`), Google Cloud Storage (`gcp.py`), Azure Blob (`azure.py`), LakeFS (`lakefs.py`), Pachyderm (`pachyderm.py`).
  - **Model Hubs & Registries**: HuggingFace Hub (`huggingface.py`), GitHub Releases (`github_release.py`), OCI Container Registries (`oci_registry.py`).
  - **Notifications & Webhooks**: Slack/Discord/HTTP webhooks (`custom_webhook.py`, `notification.py`).

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `saver.py` | Core `ResultsSaver` coordinating data export across configured backends. |
| `config.py` | `ResultsSaverConfig` dataclass defining destination directories and active backends. |
| `backends/local.py` | Local disk serializer exporting JSON, Parquet, and directory trees. |
| `backends/wandb.py` | Weights & Biases experiment logging and artifact sync. |
| `backends/mlflow.py` | MLflow tracking server and model artifact logging. |
| `backends/s3.py` | Amazon S3 cloud checkpoint upload handler. |

---

## 🚀 Quick Start & Usage

### 1. Saving Experiment Metrics & Artifacts
```python
from framework.results_saver.saver import ResultsSaver
from framework.results_saver.config import ResultsSaverConfig

config = ResultsSaverConfig(
    output_dir="./experiment_results/run_01",
    save_json=True,
    save_parquet=True,
    enable_wandb=False,
)
saver = ResultsSaver(config)

# Save configuration and metrics
saver.save_config({"model": "distilbert", "rank": 32, "lr": 2e-5})
saver.save_metrics({"eval_loss": 0.342, "eval_accuracy": 0.885, "epoch": 3})
```
