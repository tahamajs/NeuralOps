
# `framework/config` — Unified Configuration System

The **config** module provides a **typed, hierarchical configuration** for the entire Efficient Transformer Toolkit.  
Every setting is defined in a dataclass with sensible defaults, and can be loaded from YAML, CLI arguments, or constructed programmatically.

---

## Quick Start

```python
from framework.config import Config, load_config

# Build in code
cfg = Config()
cfg.model.name = "distilbert-base-uncased"
cfg.compression.attn_type = "galerkin"
cfg.compression.rank = 64
cfg.training.epochs = 5

# Load from YAML
cfg = load_config("experiments/sst2_galerkin.yaml")

# Load from YAML + CLI overrides
cfg = load_config("base.yaml", cli_args=args)
```

---

## Top‑Level Structure (`Config`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `ModelConfig` | `ModelConfig()` | Model name, labels, attention type, variants |
| `data` | `DataConfig` | `DataConfig()` | GLUE path, max length, preprocessing |
| `training` | `TrainingConfig` | `TrainingConfig()` | Batch size, epochs, LR, scheduler, evaluation |
| `compression` | `CompressionConfig` | `CompressionConfig()` | Attention injection params, rank, sparsity, hybrid |
| `method` | `MethodConfig` | `MethodConfig()` | Training method (LoRA, Mixout, R‑Drop…) |
| `hardware` | `HardwareConfig` | `HardwareConfig()` | Device, mixed precision, workers |
| `sweep` | `SweepConfig` | `SweepConfig()` | Hyperparameter sweep (Optuna) |
| `experiment` | `ExperimentConfig` | `ExperimentConfig()` | Multi‑run grid (datasets, ranks, models) |

---

## Sub‑Configurations

### `ModelConfig`

| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | `"distilbert-base-uncased"` |
| `num_labels` | `int` | `2` |
| `attention_type` | `str` | `"full"` |
| `use_fast_tokenizer` | `bool` | `True` |
| `trust_remote_code` | `bool` | `False` |
| `cache_dir` | `str` | `"./cache"` |
| `variants` | `List[str]` | `[]` |

### `DataConfig`

| Field | Type | Default |
|-------|------|---------|
| `local_glue_path` | `str` | Auto‑resolved |
| `max_seq_length` | `int` | `128` |
| `max_train_samples` | `Optional[int]` | `None` |
| `max_val_samples` | `Optional[int]` | `None` |
| `num_proc` | `int` | `4` |
| `streaming` | `bool` | `False` |
| `split_mapping` | `dict` | GLUE standard splits |

### `TrainingConfig`

| Field | Type | Default |
|-------|------|---------|
| `batch_size` | `int` | `16` |
| `eval_batch_size` | `int` | `32` |
| `epochs` | `int` | `3` |
| `learning_rate` | `float` | `2e-5` |
| `weight_decay` | `float` | `0.01` |
| `warmup_ratio` | `float` | `0.1` |
| `max_grad_norm` | `float` | `1.0` |
| `lr_scheduler_type` | `str` | `"linear"` |
| `optimizer` | `str` | `"adamw"` |
| `early_stopping_patience` | `Optional[int]` | `None` |
| `save_strategy` | `str` | `"epoch"` |
| `eval_strategy` | `str` | `"epoch"` |
| `metric_for_best_model` | `str` | `"eval_accuracy"` |
| `fp16` / `bf16` | `bool` | `False` |

### `CompressionConfig`

All parameters used by `inject_attention()` and compressors.  
Selected highlights:

| Field | Default | Used by |
|-------|---------|--------|
| `attn_type` | `"full"` | Attention injection |
| `rank` | `32` | Generic rank (SVD, LoRA) |
| `svd_energy` | `0.95` | Energy‑thresholded SVD |
| `block_size` | `64` | Block‑sparse attention |
| `local_window` | `64` | Sliding window |
| `top_k_inter` | `4` | Inter‑block top‑k |
| `num_landmarks` | `16` | Landmark attention |
| `rff_features` | `128` | Random Fourier features |
| `num_kernel_heads` | `8` | ORION / hybrid |
| `num_full_heads` | `4` | ORION / hybrid |
| `num_experts` | `8` | MoE |
| `ratio` | `9` | Hybrid Mamba/attention |

Full list of 50+ fields in the source code.

### `MethodConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `str` | `"baseline"` | Training method key |
| `lora` | `LoRAConfig` | `LoRAConfig()` | r=8, alpha=16, dropout=0.1 |
| `rdrop` | `RDropConfig` | `RDropConfig()` | alpha=5.0 |
| `r3f` | `R3FConfig` | `R3FConfig()` | epsilon=1e-5 |
| `fisher` | `FisherConfig` | `FisherConfig()` | num_batches=20, sample_size=200 |
| `gelora` | `GeLoRAConfig` | `GeLoRAConfig()` | margin=1.0, min_rank=2, max_rank=32 |
| `llrd` | `LLRDConfig` | `LLRDConfig()` | decay=0.95 |
| `mixout` | `MixoutConfig` | `MixoutConfig()` | p=0.9 |
| `dislora` | `DisLoRAConfig` | `DisLoRAConfig()` | beta=0.5, capacity_factor=1.0 |

### `HardwareConfig`

| Field | Type | Default |
|-------|------|---------|
| `device` | `str` | `"auto"` |
| `force_cpu_for_svd` | `bool` | `True` |
| `enable_fp16` / `enable_bf16` | `bool` | `False` |
| `num_workers` | `int` | `4` |
| `torch_compile` | `bool` | `False` |

### `SweepConfig`

| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | `"sweep"` |
| `algorithm` | `str` | `"tpe"` |
| `direction` | `str` | `"maximize"` |
| `metric` | `str` | `"eval_accuracy"` |
| `max_trials` | `int` | `20` |
| `pruner` | `str` | `"hyperband"` |
| `search_space` | `dict` | `{}` |

### `ExperimentConfig`

| Field | Type | Default |
|-------|------|---------|
| `base_models` | `dict` | `{"bert": "bert-base-uncased"}` |
| `svd_ranks` | `List[int]` | `[8, 16, 32]` |
| `datasets` | `List[DatasetConfig]` | `[]` |
| `epochs` | `int` | `1` |
| `batch_size` | `int` | `4` |
| `output_dir` | `str` | `"./results"` |

---

## YAML Example

```yaml
model:
  name: "distilbert-base-uncased"
  attention_type: "galerkin"
  num_labels: 2

data:
  max_seq_length: 128

training:
  epochs: 5
  batch_size: 16
  learning_rate: 2e-5

compression:
  rank: 64
  svd_energy: 0.95
  block_size: 128

method:
  method: "lora"
  lora:
    r: 16
    alpha: 32

hardware:
  device: "cuda"
  enable_fp16: true
```

## Usage in the Pipeline

```python
cfg = load_config("config.yaml")

# Model
model, tokenizer = load_model(
    cfg.model.name,
    attention_type=cfg.compression.attn_type,
    attention_rank=cfg.compression.rank
)

# Data
dataset = load_dataset(ExperimentConfig(dataset_name="sst2"))

# Training
trainer = HuggingFaceTrainer(
    model=model,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset,
    config=cfg.training.__dict__,
    results_manager=results,
)
```
