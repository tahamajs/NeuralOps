# `framework/training` — Unified Training System

The `training` module provides a **complete, modular training pipeline** for any HuggingFace model. It supports:
- multiple trainer backends (HuggingFace Trainer, custom loop, unsloth, RL)
- results management with automatic logging, plotting, and model saving
- a library of training methods (LoRA, DoRA, Mixout, LLRD, R‑Drop, R3F, Fisher, GeLoRA)
- hyperparameter optimisation with Optuna
- custom callbacks for memory, time, gradient norms, SVD attention
- standardised metrics for GLUE tasks

All components are designed to work seamlessly with the core framework, data pipeline, and custom attention modules.

---

## Directory Structure

```
training/
├── __init__.py              # Public API
├── base_trainer.py          # ABC for all trainers
├── registry.py              # TrainerRegistry (name → class)
├── trainer_factory.py       # Dynamic trainer selection
├── results_manager.py       # Per‑run directories, metric logging, plotting
├── callbacks.py             # Memory, Time, GradientNorm, SVDAttention, Preemption
├── metrics.py               # GLUE task‑specific compute_metrics, aggregation
├── trainer.py               # SVDAwareTrainer (HuggingFace with compression hooks)
├── hf_trainer.py            # HuggingFaceTrainer (wraps SVDAwareTrainer + ResultsManager)
├── custom_loop_trainer.py   # CustomLoopTrainer (manual PyTorch loop)
├── utils.py                 # train_model, evaluate_model, benchmark_model, plotting
├── hyperopt.py              # HyperparameterOptimizer using Optuna
├── losses.py                # Loss registry (cross_entropy, focal, dice, etc.)
├── optimizers.py            # Optimizer registry (adamw, sgd, lamb, etc.)
├── schedulers.py            # Scheduler registry (linear, cosine, warmup, etc.)
├── unsloth_trainer.py       # Placeholder for Unsloth
├── rl_trainer.py            # Placeholder for TRL (RLHF)
│
└── methods/                 # Training method plugins
    ├── __init__.py
    ├── registry.py          # METHOD_REGISTRY
    ├── lora.py              # LoRA, DoRA, DisLoRA
    ├── fisher.py            # Fisher importance scoring
    ├── gelora.py            # Intrinsic dimension based rank selection
    ├── llrd.py              # Layer‑wise learning rate decay
    ├── mixout.py            # Mixout regularisation
    └── rdrop_r3f.py         # R‑Drop and R3F robust training
```

---

## Quick Start

### 1. Use the HuggingFace Trainer with ResultsManager

```python
from framework.training import HuggingFaceTrainer, ResultsManager, get_task_metrics
from framework.data import load_dataset, DataConfig, TaskType, create_dataloader

# Data
config = DataConfig(task_type=TaskType.SINGLE_CLASSIFICATION,
                    text_column_1="sentence", label_column="label")
dataset = load_dataset(ExperimentConfig(dataset_name="sst2"))
tokenizer = ...
train_loader = create_dataloader(dataset["train"], tokenizer, config)
val_loader = create_dataloader(dataset["validation"], tokenizer, config)

# Results
results = ResultsManager("my_experiment", base_dir="./results")

# Trainer
trainer = HuggingFaceTrainer(
    model=model,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset,
    config={"learning_rate": 2e-5, "epochs": 3, "batch_size": 16},
    results_manager=results,
    compute_metrics=get_task_metrics("sst2"),
)
trainer.train()
```

### 2. Apply a training method (LoRA, etc.)

```python
from framework.trainer.methods import apply_lora
from types import SimpleNamespace

cfg = SimpleNamespace(lora_r=8, lora_alpha=16, lora_dropout=0.1)
apply_lora(model, cfg)
```

### 3. Run hyperparameter optimisation

```python
from framework.training import quick_optimize

def train_fn(config):
    # build and train model with config
    pass
def eval_fn(config):
    # evaluate model
    return {"accuracy": 0.92}

best = quick_optimize(train_fn, eval_fn, n_trials=20)
```

---

## Key Components

### Base Classes and Registries

- **`BaseTrainer`** – abstract interface with `train()` and `evaluate()`.
- **`TrainerRegistry`** – maps string keys (`"hf"`, `"custom_loop"`) to trainer classes.
- **`METHOD_REGISTRY`** – maps method names (`"lora"`, `"rdrop"`) to callables that modify the model or training process.

### ResultsManager

Handles all output for a single run:
- Creates timestamped run directory (`results/my_experiment_2025-...`)
- Saves config as YAML, metrics as JSONL, training plots as PNG
- Wraps a HuggingFace `Trainer` with memory, time, and gradient callbacks
- Automatically saves final model

### Trainers

| Trainer | Description |
|---------|-------------|
| `HuggingFaceTrainer` | Wraps `SVDAwareTrainer` with `ResultsManager`, auto‑saves model and plots |
| `CustomLoopTrainer` | Manual training loop using the provided `train_model` utility |
| `SVDAwareTrainer` | HuggingFace `Trainer` with pre/post callbacks, attention map hooks, memory cleanup |

### Callbacks (built into ResultsManager wrapping)

- `MemoryCallback` – logs GPU/MPS memory usage
- `TimeCallback` – logs elapsed time
- `GradientNormCallback` – logs total gradient norm
- `SVDAttentionCallback` – logs compression statistics (SVD layers, compression ratio)
- `PreemptionCallback` – stops training when a preemption file is present

### Metrics

`get_task_metrics(task_name)` returns a `compute_metrics` function for the following GLUE tasks:
- `cola` (Matthews)
- `stsb` (Pearson/Spearman)
- `mrpc` / `qqp` (Accuracy + F1)
- others (Accuracy, F1 macro, F1 weighted)

Utility functions: `compute_accuracy`, `compute_f1`, `compute_matthews_corrcoef`, `compute_pearson_spearman`, `aggregate_results`.

### Training Methods (`methods/`)

| Method | Description | Registered Key |
|--------|-------------|---------------|
| **LoRA** | Low‑Rank Adaptation (standard) | `lora` |
| **DoRA** | Weight‑decomposed Low‑Rank Adaptation | `dora` |
| **DisLoRA** | Dissimilarity‑based LoRA with orthogonality regularisation | `dislora` |
| **Fisher** | Fisher information based importance scoring (for adaptive rank) | `fisher_lora` |
| **GeLoRA** | Intrinsic dimension estimation for per‑layer rank selection | `fisher_gelora` |
| **LLRD** | Layer‑wise learning rate decay | `llrd` |
| **Mixout** | Parameter mixing for regularisation | `mixout` |
| **R‑Drop** | Regularised Dropout (KL divergence between two forward passes) | `rdrop` |
| **R3F** | Robust Representation through Random Noise | `r3f` |

All methods are applied by calling `apply_<method>(model, config)`.

### Hyperparameter Optimisation

`HyperparameterOptimizer` uses Optuna to search over learning rate, batch size, rank, weight decay, warmup, and SVD‑specific parameters. Supports pruning (`median`, `hyperband`) and visualisation (optimisation history, parallel coordinates, parameter importances).

### Losses, Optimizers, Schedulers

Each has a registry that can be extended:

- **Losses**: `cross_entropy`, `focal`, `label_smoothing`, `dice`, `poly`, `contrastive`
- **Optimizers**: `adamw`, `sgd`, `adam`, `adabelief`, `ranger`, `lamb`, `lars`
- **Schedulers**: `linear`, `cosine`, `reduce_on_plateau`, `one_cycle`, `cosine_warmup`, `polynomial`

---

## Integration with the Framework

The training module expects the model to be a standard HuggingFace `PreTrainedModel` (or a wrapper like `UnifiedTransformerModel`).  
It works with any tokenizer and any dataset loaded via `framework.data`.

After training, the `ResultsManager` saves the model to `results/<run>/final_model/` and generates plots and logs.  
These can be consumed by the analysis and comparison modules (`framework.analysis`, `framework.comparison`).

---

## Example: Full Training Pipeline

```python
from framework.core import UnifiedTransformerModel
from framework.data import load_dataset, DataConfig, TaskType, create_dataloader
from framework.training import HuggingFaceTrainer, ResultsManager, get_task_metrics
from framework.trainer.methods import apply_lora

# 1. Model
model = UnifiedTransformerModel.from_pretrained("distilbert-base-uncased")
apply_lora(model, SimpleNamespace(lora_r=8, lora_alpha=16, lora_dropout=0.1))

# 2. Data
dataset = load_dataset(ExperimentConfig(dataset_name="sst2"))
config = DataConfig(task_type=TaskType.SINGLE_CLASSIFICATION, text_column_1="sentence", label_column="label")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
train_loader = create_dataloader(dataset["train"], tokenizer, config)
val_loader = create_dataloader(dataset["validation"], tokenizer, config)

# 3. Results
results = ResultsManager("lora_sst2")

# 4. Train
trainer = HuggingFaceTrainer(
    model=model,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset,
    config={"learning_rate": 2e-5, "epochs": 3, "batch_size": 16},
    results_manager=results,
    compute_metrics=get_task_metrics("sst2"),
)
trainer.train()
```

After training, the output directory contains:
```
results/lora_sst2_2025-.../
├── checkpoints/
├── logs/run.log
├── final_model/
├── metrics.jsonl
├── training_history.png
└── config.yaml
```
