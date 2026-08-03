# Efficient Transformer Toolkit

A **model‑agnostic, production‑ready** framework for compressing, adapting, and fine‑tuning **any HuggingFace Transformer**.  
Swap in **80+ state‑of‑the‑art attention mechanisms**, compress linear layers, and train with a full‑featured pipeline—all with automatic architecture detection and pretrained weight transfer.

---

## 🚀 Quick Start

```python
# 1. Load a pretrained model with a custom attention variant
from framework.models import load_model
model, tokenizer = load_model(
    "distilbert-base-uncased",
    attention_type="galerkin",
    attention_rank=32
)

# 2. Prepare data (GLUE / custom / synthetic)
from framework.data import load_dataset, DataConfig, TaskType, create_dataloader
dataset = load_dataset(ExperimentConfig(dataset_name="sst2"))
config = DataConfig(task_type=TaskType.SINGLE_CLASSIFICATION,
                    text_column_1="sentence", label_column="label")
train_loader = create_dataloader(dataset["train"], tokenizer, config)

# 3. Train with a full experiment tracker and automatic results saving
from framework.training import HuggingFaceTrainer, ResultsManager
from framework.tracking import ExperimentTracker
from framework.results_saver import ResultsSaver, SaveConfig

results = ResultsManager("my_experiment")
tracker = ExperimentTracker(backend="tensorboard", log_dir="./runs")
trainer = HuggingFaceTrainer(
    model=model,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset,
    config={"learning_rate": 2e-5, "epochs": 3, "batch_size": 16},
    results_manager=results,
)
trainer.train()

# 4. Save and share your compressed model
saver = ResultsSaver(run_id=results.run_dir.name, base_dir=results.base_dir,
                     config=SaveConfig(storage_backends=["hf_hub"]))
saver.save_all(model, config, results.history, metrics, tokenizer)
saver.push_all()
```

---

## ✨ Key Features

- **Universal Model Loading** – DistilBERT, BERT, RoBERTa, GPT‑2, Llama, Mistral, Gemma, … all supported out‑of‑the‑box.
- **80+ Attention Mechanisms** – from minimal SVD to full AURORA, NOVA, PRISM, ORION, NEXUS, and many more.
- **Zero‑Shot Weight Transfer** – automatically copies pretrained projection weights into the new attention module so you don’t have to retrain from scratch.
- **Compression Pipeline** – apply multiple compressors (A3, AASVD, KMeans, FRanDI, …) sequentially with a single config.
- **Data Pipeline** – load GLUE, custom CSV/JSON/Parquet, synthetic text, and multi‑choice datasets with automatic tokenization and collation.
- **Complete Training Stack** – HuggingFace Trainer wrapper, custom training loop, hyperparameter optimisation (Optuna), learning rate schedulers, loss functions, and optimizers.
- **Experiment Tracking** – log to TensorBoard, Weights & Biases, MLflow, Neptune, Comet, AimStack, ZenML with a single interface.
- **Results Persistence** – save models, generate comprehensive plots (loss, accuracy, LR, confusion matrix, …), export to ONNX/TorchScript, push to cloud storage (S3, GCP, Azure, HuggingFace Hub), create GitHub Releases, and more.
- **Security** – sign and verify model artifacts with Ed25519 to guarantee integrity and provenance.
- **Analysis & Comparison** – statistical tests (ANOVA, Friedman, Quade, post‑hoc, Bayesian), effect sizes, Critical Difference diagrams, and full HTML/LaTeX reports.

---

## 📦 Package Structure

```
framework/
├── core/                # Architecture detection, adapters, attention injection, compression engine
├── data/                # DataConfig, loaders, tokenization, augmentation
├── models/              # Universal model loader + custom compressed architectures
├── training/            # Trainers, results manager, callbacks, methods (LoRA, Mixout, R‑Drop…)
├── callbacks/           # Analysis & custom training callbacks
├── tracking/            # Experiment tracking backends (local, TensorBoard, WandB, MLflow)
├── results_saver/       # Artifact persistence, export, cloud push
├── security/            # Model signing & verification
├── analysis/            # Plots, metrics, layer‑wise analysis, model comparison
└── comparison/          # Statistical tests, leaderboard, comparison engine
```

---

## 🔧 Core Concepts

### 1. Architecture Detection & Adapters
`framework.core.architecture` automatically identifies the model family (DistilBERT, Llama, …).  
`framework.core.adapters` knows how to extract Q/K/V/O weights, handle GQA, and RoPE for each family.

### 2. Attention Injection
`framework.core.attention_utils.inject_attention(model, "attention_type")` replaces all self‑attention layers with one of the 80+ registered variants. Weights are copied automatically.

### 3. Compression Pipeline
`framework.core.compress_model(model, config)` runs a sequence of compressors (SVD, k‑means, …) from `COMPRESSOR_REGISTRY`.

### 4. Data Pipeline
`framework.data` provides `DataConfig`, `TaskType`, `load_dataset`, and `create_dataloader` for GLUE, custom files, synthetic text, and multi‑choice tasks.

### 5. Training & Callbacks
`framework.training.HuggingFaceTrainer` wraps a HuggingFace Trainer with `ResultsManager` for logging, plot saving, and optional method application (LoRA, DoRA, Mixout, …).  
A library of callbacks (Memory, Time, GradientNorm, SVDAttention) is included.

### 6. Experiment Tracking
`framework.tracking.ExperimentTracker` offers a unified API for logging metrics and parameters to local JSON, TensorBoard, WandB, MLflow, Neptune, Comet, Aim, or ZenML.

### 7. Results Saving
`framework.results_saver.ResultsSaver` with a `SaveConfig` can save models, generate all analysis plots, export to ONNX/TorchScript, and push to cloud storage (S3, GCP, Azure, HuggingFace Hub, …).

### 8. Security
`framework.security` provides Ed25519‑based model signing and verification so you can ensure model integrity and authenticity.

---

## 📚 Attention Modules

All attention variants are registered in `ATTENTION_REGISTRY` and can be injected by name. Here is a selection:

| Key | Description |
|-----|-------------|
| `full` | Original BERT / DistilBERT attention |
| `minimal_svd` | SVD‑compressed projections, zero‑shot |
| `galerkin` | Galerkin linear attention |
| `linformer` | Linformer (linear complexity) |
| `performer` | Performer (FAVOR+) |
| `orion` | SVD + head‑split kernel/full + Softpick |
| `nexus` | XAttention + SeerAttention + DuSA + … |
| `oha` | Optimized Hybrid Attention |
| `lsh` | Reformer LSH |
| `gqa` | Grouped‑Query Attention |
| `wgqa` | Weighted GQA |
| `qca` | Quality‑Capacity‑Aware GQA |
| … | (80+ in total) |

Use `ATTENTION_REGISTRY.keys()` to see the full list.

---

## 🛠️ Customization

### Adding a New Attention Module
1. Inherit from `BaseAttention`.
2. Implement `forward`.
3. Register with `@ATTENTION_REGISTRY.register("my_attn")`.

### Adding a New Model Family
1. Extend `framework.core.adapters` with a new adapter class.
2. Add the architecture detection logic in `framework.core.architecture`.

### Adding a New Compressor
1. Inherit from `BaseCompressor`.
2. Implement `compress()` → `(model, stats)`.
3. Register with `@COMPRESSOR_REGISTRY.register("my_comp")`.

### Adding a New Data Source
1. Write a loader function and hook it into `framework.data.dataset_loader.load_dataset`.

---

## ⚙️ Configuration Reference

- **DataConfig** – `task_type`, `max_length`, `batch_size`, …
- **ExperimentConfig** – `dataset_name`, `split_ratio`, `synthetic_samples`, …
- **SaveConfig** – `save_model`, `plot_options`, `storage_backends`, `experiment_trackers`, …
- **Compression Pipeline Config** – `{"compressors": [{"method": "aasvd", ...}]}`

All configs can be passed as dictionaries or dataclass instances.

---

## 🔒 Dependencies

Core:
- `torch ≥ 2.0`
- `transformers`
- `datasets`
- `numpy`
- `scipy`

Optional (for specific features):
- `tensorboard`, `wandb`, `mlflow`, `neptune`, `comet_ml`, `aim`, `zenml` – tracking
- `boto3`, `google-cloud-storage`, `azure-storage-blob` – cloud storage
- `huggingface_hub` – Hub upload
- `optuna` – hyperparameter search
- `gitpython` – Git integration
- `matplotlib`, `seaborn` – plotting
- `unsloth` – fast Llama/Mistral training
- `cryptography` – model signing

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Contributing

Contributions are welcome! See the `CONTRIBUTING.md` for guidelines on adding new attention variants, compressors, or model families. Please ensure that any new module passes the existing test suite (`pytest tests/`).

---

*Built with ❤️ for the efficient Transformer community.*