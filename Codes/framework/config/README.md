# Configuration Subsystem (`Codes/framework/config`)

The `config` subsystem provides type-safe, dataclass-driven hierarchical configuration management for models, attention mechanisms, compression algorithms, datasets, trainers, and hyperparameter sweeps.

---

## 🏗 Architecture Overview

- **Hierarchical Dataclasses**: Strictly typed configuration structures ensuring compile-time safety and automatic JSON/YAML schema validation:
  - **`ExperimentConfig` (`experiment_config.py`)**: Top-level orchestrator combining model, training, data, compression, and hardware configs.
  - **`ModelConfig` (`model_config.py`)**: Transformer architecture parameters (hidden dimension, heads, layers, vocab size, attention variant).
  - **`TrainingConfig` (`training_config.py`)**: Hyperparameters (learning rate, optimizer, batch size, warmup, weight decay, FP16/BF16).
  - **`CompressionConfig` (`compression_config.py`)**: SVD rank, spectral energy thresholds, pruning ratios, and quantization bits.
  - **`HardwareConfig` (`hardware_config.py`)**: Device selection (CUDA, MPS, CPU), precision mode, and worker threads.
- **Config Loader & Serializer (`loader.py`, `config.py`)**: Loads configurations from YAML/JSON files, environment variables, and CLI flags with override merging.
- **Hyperparameter Sweeps (`sweep_config.py`)**: Grid search and Bayesian optimization sweep space definitions.

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `experiment_config.py` | `ExperimentConfig` root container aggregating all sub-configurations. |
| `model_config.py` | Model architecture and attention variant configuration dataclass. |
| `training_config.py` | Training hyperparameters and optimization settings. |
| `compression_config.py` | Low-rank SVD and compression parameter configuration. |
| `loader.py` | YAML/JSON file parser and configuration builder with override support. |
| `sweep_config.py` | Hyperparameter sweep grid and random sampling configurations. |

---

## 🚀 Quick Start & Usage

### 1. Programmatic Configuration
```python
from framework.config.experiment_config import ExperimentConfig
from framework.config.model_config import ModelConfig
from framework.config.training_config import TrainingConfig

exp_cfg = ExperimentConfig(
    model=ModelConfig(name="distilbert-base-uncased", attention_type="svd_linear_attention", rank=32),
    training=TrainingConfig(learning_rate=2e-5, epochs=3, batch_size=32, fp16=True),
)
print(f"Active Attention: {exp_cfg.model.attention_type}, Rank: {exp_cfg.model.rank}")
```

### 2. Loading from YAML File
```python
from framework.config.loader import load_config

cfg = load_config("configs/svd_linear_experiment.yaml")
```
