# 🧰 Utilities Package

A comprehensive collection of **helper functions, managers, and profilers** for the Efficient Transformer Toolkit.  
These utilities support model compression experiments, checkpoint management, profiling, configuration handling, and more.

## 📁 Package Structure

```
framework/utils/
├── __init__.py
├── architecture.py        # Model architecture detection & layer patterns
├── checkpoint.py          # CheckpointManager for resumable experiments
├── config.py              # YAML/JSON config loading & merging
├── device.py              # Automatic device selection
├── helpers.py             # General helpers (seed, formatting, file I/O)
├── layer_norm_fix.py      # Fix LayerNorm naming in DistilBERT checkpoints
├── logger.py              # Simple logging setup
├── metrics.py             # GLUE metrics computation
├── profiler.py            # Memory, time, and latency profiler
├── seed.py                # Comprehensive random seed setting
└── README.md
```

---

## 📦 Modules

### 1. Architecture Detection (`architecture.py`)

Detects model architecture from its name and returns layer name patterns for various Transformer models.

**Supported architectures:** `bert`, `distilbert`, `roberta`, `xlnet`, `albert`, `electra`, `deberta`, `gpt2`

**Key functions:**
- `detect_architecture(model_name)` → architecture string
- `get_layer_patterns(arch)` → list of layer name patterns
- `get_compression_targets(model_name, include_ffn)` → layers to compress
- `get_model_info(model_name)` → dict with architecture, patterns, estimated config
- `get_layer_count(model)` → number of transformer layers
- `get_hidden_size(model)` → hidden dimension size

```python
from framework.utils.architecture import detect_architecture, get_layer_patterns

arch = detect_architecture("distilbert-base-uncased")
patterns = get_layer_patterns(arch)
# ['attention.q_lin', 'attention.k_lin', 'attention.v_lin', ...]
```

### 2. Checkpoint Management (`checkpoint.py`)

`CheckpointManager` handles saving and loading experiment state for **resumable long‑running experiments**.

```python
from framework.utils.checkpoint import CheckpointManager

manager = CheckpointManager("./checkpoints")
manager.save_checkpoint(experiment_id=1, results=results_list)
experiment_id, results, metadata = manager.load_checkpoint()
```

**Methods:**
- `save_checkpoint(experiment_id, results, name, metadata)`
- `load_checkpoint(name)` → `(experiment_id, results, metadata)`
- `delete_checkpoint(name)`, `list_checkpoints()`, `get_latest_checkpoint()`

### 3. Configuration Management (`config.py`)

Load, save, and merge **YAML / JSON** configuration files.

```python
from framework.utils.config import load_config, merge_configs, save_config

base = load_config("base.yaml")
override = load_config("experiment.yaml")
config = merge_configs(base, override)
save_config(config, "merged.yaml")
```

**Functions:** `load_config`, `save_config`, `update_config`, `merge_configs`

### 4. Device Selection (`device.py`)

Automatically picks the best available device.

```python
from framework.utils.device import get_device

device = get_device()        # auto: cuda > mps > cpu
device = get_device("cuda")  # explicit
```

### 5. General Helpers (`helpers.py`)

A collection of frequently used utilities.

| Function | Description |
|----------|-------------|
| `set_seed(seed, deterministic)` | Set random seeds for reproducibility |
| `format_time(seconds)` | `"1h 23m 45s"` |
| `format_number(num)` | `"1.2M"`, `"45K"` |
| `count_parameters(model, trainable_only)` | Parameter count |
| `ensure_dir(path)` | Create directory if needed |
| `save_json(data, path)` / `load_json(path)` | JSON file I/O |
| `flatten_dict(d)` / `unflatten_dict(d)` | Nested ↔ flat dict |

### 6. LayerNorm Fix (`layer_norm_fix.py`)

Fixes **DistilBERT checkpoint compatibility** – renames `beta`/`gamma` → `bias`/`weight`.

```python
from framework.utils.layer_norm_fix import fix_layernorm_naming

model = load_model("distilbert-base-uncased")
fix_layernorm_naming(model)
```

### 7. Logger Setup (`logger.py`)

Quick console and file logging.

```python
from framework.utils.logger import setup_logger

logger = setup_logger("experiment", log_file="logs/run.log")
logger.info("Training started")
```

**Functions:** `setup_logger`, `get_logger`, `set_log_level`

### 8. Metrics (`metrics.py`)

GLUE benchmark metric computation (supports classification and regression).

```python
from framework.utils.metrics import compute_metrics_for_task

metrics = compute_metrics_for_task(predictions, labels, task_name="mrpc")
# {'accuracy': 0.85, 'f1_macro': ..., 'f1_weighted': ...}
```

**Functions:** `compute_accuracy`, `compute_f1`, `compute_matthews_corrcoef`, `compute_pearson`, `compute_spearman`, `compute_precision_recall`, `compute_metrics_for_task`

### 9. Profiler (`profiler.py`)

Profiles **memory, time, and inference latency** per layer.

```python
from framework.utils.profiler import ModelProfiler, measure_inference_latency

profiler = ModelProfiler(model)
profiler.start_profiling()
output = model(input)
profiler.stop_profiling()
profiler.print_report()

latency_stats = measure_inference_latency(model, dataloader, device)
```

**Classes & Functions:**
- `ModelProfiler` – hooks‑based per‑layer profiling
- `measure_inference_latency` – end‑to‑end latency stats
- `measure_memory_usage` – peak GPU memory
- `profile_context` – context manager for profiling

### 10. Seed (`seed.py`)

Sets **all random seeds** (Python, NumPy, PyTorch, CUDA).

```python
from framework.utils.seed import set_all_seeds

set_all_seeds(42)
```

---

## 🔧 Integration

All utilities are designed to work standalone or together. Import them directly:

```python
from framework.utils import (
    detect_architecture, CheckpointManager, load_config, get_device,
    set_seed, format_time, fix_layernorm_naming, setup_logger,
    compute_metrics_for_task, ModelProfiler, set_all_seeds
)
```

---

*Reliable utilities for reproducible and efficient research.*
