# 🧰 Utilities Package

A comprehensive set of helper functions, managers, and profilers for the **Efficient Transformer Toolkit**.  
These utilities support reproducibility, device handling, logging, metrics, configuration, checkpointing, profiling, model architecture detection, and more.

## 📁 Package Structure

```
framework/utils/
├── __init__.py                 # Consolidated exports (including replace_attention)
├── architecture.py            # Model architecture detection & layer patterns
├── checkpoint.py              # CheckpointManager for resumable experiments
├── config.py                  # YAML/JSON config loading & merging
├── device.py                  # Automatic device selection (CUDA, MPS, CPU)
├── helpers.py                 # General helpers: seeds, formatting, file I/O, dict flattening
├── layer_norm_fix.py          # Fix DistilBERT LayerNorm naming
├── logger.py                  # Simple logging setup (console + file)
├── metrics.py                 # GLUE metrics computation
├── profiler.py                # ModelProfiler (per‑layer timing/memory), latency/memory measurement
├── save_results.py            # Save experiment results (JSON, CSV, plots, model, checkpoints)
├── seed.py                    # Set all random seeds for reproducibility
└── README.md                  # This file
```

## 🚀 Quick Start

```python
from framework.utils import (
    set_all_seeds,
    get_device,
    load_config,
    CheckpointManager,
    ModelProfiler,
    replace_attention,
    save_experiment_results,
)

# 1. Set seeds for reproducibility
set_all_seeds(42)

# 2. Get best available device
device = get_device()

# 3. Load configuration
config = load_config("experiment.yaml")

# 4. Replace attention layers in a model
model = replace_attention(model, "goldfinch_ahn", rank=64)

# 5. Profile model
profiler = ModelProfiler(model)
with profiler.start_profiling():
    output = model(**batch)
profiler.print_report()

# 6. Manage checkpoints
checkpoint_manager = CheckpointManager("./checkpoints")
checkpoint_manager.save_checkpoint(experiment_id=1, results=[...])

# 7. Save full experiment results
save_experiment_results(run, trainer, tokenizer, history, final_metrics)
```

## 📦 Detailed Module Documentation

### 1. `seed.py` – Comprehensive Random Seed Setting

```python
def set_all_seeds(seed: int) -> None
```

Sets Python (`random`), NumPy, PyTorch, and CUDA seeds. Also sets `torch.backends.cudnn.deterministic = True` and `benchmark = False` for full reproducibility.

**Example:**
```python
from framework.utils import set_all_seeds
set_all_seeds(42)
```

---

### 2. `device.py` – Automatic Device Selection

```python
def get_device(device_str: str = "auto") -> torch.device
```

Returns the best available device: `cuda` → `mps` → `cpu`, or a specific device by string.

**Example:**
```python
from framework.utils import get_device
device = get_device()           # auto
device = get_device("cuda")     # explicit
```

---

### 3. `helpers.py` – General Utilities

| Function | Description | Example |
|----------|-------------|---------|
| `set_seed(seed, deterministic)` | Seeds + optional deterministic CuDNN | `set_seed(42, deterministic=True)` |
| `format_time(seconds)` | Human‑readable time | `format_time(3665)` → `"1h 1m 5s"` |
| `format_number(num)` | Suffix (K, M, B) | `format_number(12345)` → `"12.3K"` |
| `count_parameters(model, trainable_only)` | Parameter count | `count_parameters(model)` |
| `ensure_dir(path)` | Create directory if missing | `ensure_dir("logs/")` |
| `save_json(data, path)` / `load_json(path)` | JSON I/O | `save_json({"a":1}, "file.json")` |
| `flatten_dict(d)` / `unflatten_dict(d)` | Nested ↔ flat dict | `flatten_dict({"a":{"b":1}})` → `{"a.b":1}` |

---

### 4. `logger.py` – Easy Logging Setup

```python
def setup_logger(name, level=logging.INFO, log_file=None, format_string=None) -> logging.Logger
```

Sets up a logger with console output and optional file output.

**Example:**
```python
from framework.utils import setup_logger
logger = setup_logger("exp", log_file="logs/run.log")
logger.info("Training started")
```

---

### 5. `metrics.py` – GLUE Task Metrics

```python
def compute_metrics_for_task(predictions, labels, task_name, is_regression=False) -> Dict[str, float]
```

Returns metrics appropriate for the given GLUE task.

- Classification: accuracy, F1 macro/weighted. For `cola`: Matthews correlation. For `mrpc`/`qqp`: binary F1.
- Regression (`stsb`): Pearson, Spearman, and their average.

**Individual metric functions** also provided: `compute_accuracy`, `compute_f1`, `compute_matthews_corrcoef`, `compute_pearson`, `compute_spearman`, `compute_precision_recall`.

**Example:**
```python
from framework.utils import compute_metrics_for_task
metrics = compute_metrics_for_task(preds, labels, task_name="mrpc")
# {'accuracy': 0.85, 'f1_macro': 0.84, 'f1_weighted': 0.85, 'f1': 0.86}
```

---

### 6. `config.py` – Configuration Management

```python
load_config(path)     # JSON/YAML → dict
save_config(config, path)
update_config(config, updates)   # recursive update
merge_configs(base, override)    # override takes precedence
```

**Example:**
```python
from framework.utils import load_config, merge_configs
base = load_config("base.yaml")
override = load_config("exp.yaml")
config = merge_configs(base, override)
save_config(config, "merged.yaml")
```

---

### 7. `checkpoint.py` – `CheckpointManager`

Manages checkpoints for long‑running experiments (e.g., sweeping).

**Methods:**
- `save_checkpoint(experiment_id, results, checkpoint_name, metadata) → Path`
- `load_checkpoint(checkpoint_name) → (experiment_id, results, metadata)`
- `delete_checkpoint()`
- `list_checkpoints() → List[Path]`
- `get_latest_checkpoint() → Optional[Path]`

**Example:**
```python
from framework.utils import CheckpointManager
manager = CheckpointManager("./checkpoints")
manager.save_checkpoint(exp_id=1, results=partial_results)
exp_id, results, meta = manager.load_checkpoint()
```

---

### 8. `profiler.py` – Model Profiling

#### `ModelProfiler` – Per‑layer timing and memory

```python
profiler = ModelProfiler(model)
profiler.start_profiling()
output = model(**batch)
profiler.stop_profiling()
profiler.print_report()
profiler.save_report("profile.json")
```

The report includes average time, standard deviation, min/max time, and average memory (if CUDA) per identified layer (`attention`, `ffn`, `mlp`, `intermediate`).

#### Inference latency measurement

```python
from framework.utils import measure_inference_latency
latency_stats = measure_inference_latency(model, dataloader, device, num_samples=100, warmup=10)
# returns mean, std, p50, p95, p99, samples/sec
```

#### Peak memory measurement

```python
from framework.utils import measure_memory_usage
mem_stats = measure_memory_usage(model, dataloader, device, num_samples=10)
# returns peak_memory_mb, avg_peak_memory_mb
```

#### Context manager

```python
with profile_context(model) as profiler:
    output = model(**batch)
# profiler automatically started and stopped
```

---

### 9. `architecture.py` – Model Detection & Layer Patterns

Supports: `bert`, `distilbert`, `roberta`, `xlnet`, `albert`, `electra`, `deberta`, `gpt2`.

**Functions:**
- `detect_architecture(model_name) → str`
- `get_layer_patterns(arch) → List[str]` – layer name patterns for attention/FFN replacement
- `get_compression_targets(model_name, include_ffn) → List[str]` – target layer names for compression
- `get_model_info(model_name) → Dict` – architecture, patterns, estimated config
- `get_layer_count(model) → int`
- `get_hidden_size(model) → int`

**Example:**
```python
from framework.utils import detect_architecture, get_compression_targets
arch = detect_architecture("distilbert-base-uncased")  # 'distilbert'
targets = get_compression_targets("bert-base-uncased", include_ffn=False)
# ['attention.self.query', 'attention.self.key', ...]
```

---

### 10. `layer_norm_fix.py` – DistilBERT LayerNorm Fix

```python
def fix_layernorm_naming(model: nn.Module) -> None
```

Renames LayerNorm parameters from `beta`/`gamma` (used in some saved checkpoints) to `bias`/`weight` (expected by Hugging Face).

**Example:**
```python
from framework.utils import fix_layernorm_naming
fix_layernorm_naming(model)
```

---

### 11. `save_results.py` – Save Full Experiment Results

```python
def save_experiment_results(run, trainer, tokenizer, history, final_metrics, output_dir="results") -> Path
```

Creates a timestamped folder containing:
- `config.json`
- `training_history.json`
- `final_metrics.json`
- `training_history.csv`, `eval_history.csv`
- `logs.txt`
- `loss_curve.png`
- `model/` (saved with `trainer.save_model` and tokenizer)
- `checkpoints/` (copied from trainer output dir)
- `README.txt`

**Example:**
```python
from framework.utils import save_experiment_results
save_experiment_results(run, trainer, tokenizer, history, final_metrics)
```

---

### 12. `replace_attention` – Dynamic Attention Replacement

Uses `ATTENTION_REGISTRY` from `framework.attentions.core.registry`.

```python
def replace_attention(model, attention_key, **kwargs) -> nn.Module
```

Replaces every module with `'attention'` or `'attn'` in its name with a registered attention block. Passes the model config and extra keyword arguments.

**Example:**
```python
from framework.utils import replace_attention
model = replace_attention(model, "goldfinch_ahn", rank=64, window_size=256)
```

---

## 🔧 Integration with the Framework

All utilities are available from `framework.utils`. The `__init__.py` exports all public functions/classes.

```python
# Typical usage in a training script
from framework.utils import set_all_seeds, get_device, load_config, get_logger, compute_metrics_for_task
```

The `replace_attention` function relies on `ATTENTION_REGISTRY`, which must be populated by importing the desired attention modules (e.g., `framework.attentions.hybrid_ssm`).

## 📜 Dependencies

- `torch`, `numpy`, `scipy`, `matplotlib`
- `scikit-learn` (for metrics)
- `pyyaml` (for YAML config)

Optional: `psutil` (not used in utils, only in other modules).

## 📄 License

Part of the **Efficient Transformer Toolkit** – MIT License.
