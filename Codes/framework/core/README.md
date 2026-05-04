# 🧩 Core Package

The `framework/core` package provides the foundational building blocks for the Efficient Transformer Toolkit – including registries, base classes, model wrappers, architecture detection, experiment tracking, and compression utilities.

## 📁 Structure

```
framework/core/
├── __init__.py               # Public API
├── utils.py                  # Device, seed, parameter counting, memory utilities
├── registry.py               # Centralised registries (ATTENTION, COMPRESSOR, TRAINER, VISUALIZER)
├── base.py                   # Concrete base classes (BaseAttention, BaseCompressor, BaseVisualizer)
├── interfaces.py             # Abstract interfaces for pluggable components
├── model_wrapper.py          # UnifiedTransformerModel – universal HF model wrapper
├── architecture.py           # Architecture detection and attention replacement
├── experiment_tracker.py     # ExperimentTracker for logging metrics and checkpoints
├── config.py                 # Simple YAML/JSON config loader
├── attention_utils.py        # Standalone attention replacement helpers
└── README.md
```

## 🚀 Quick Start

```python
from framework.core import (
    get_device, set_seed,
    ATTENTION_REGISTRY,
    BaseAttention,
    UnifiedTransformerModel,
    ExperimentTracker,
    replace_attention_in_model
)

# 1. Set device and seed
device = get_device()
set_seed(42)

# 2. Wrap a Hugging Face model
model = UnifiedTransformerModel.from_pretrained("bert-base-uncased")

# 3. Replace attention modules
model.replace_attention("goldfinch_ahn", rank=64)

# 4. Track experiment
tracker = ExperimentTracker("results", "exp1")
tracker.log_metric("accuracy", 0.92, step=100)
tracker.save_config({"model": "bert", "rank": 64})

# 5. Count parameters
print(model.get_parameter_count())
```

## 📦 Modules

### `utils.py` – Essential helpers

| Function | Description |
|----------|-------------|
| `get_device(device_str="auto")` | Returns best available device (cuda/mps/cpu). |
| `set_seed(seed, deterministic=False)` | Sets all random seeds for reproducibility. |
| `count_parameters(model, trainable_only)` | Parameter count. |
| `format_number(num)` | Human‑readable formatting (1.2M, 45K). |
| `clear_memory()` | Empties GPU/MPS cache. |
| `human_readable_time(seconds)` | `"1h 23m 45s"`. |

### `registry.py` – Central registries

Registries allow dynamic component lookup:

- `ATTENTION_REGISTRY` – for attention mechanisms.
- `COMPRESSOR_REGISTRY` – for compression methods.
- `TRAINER_REGISTRY` – for custom trainers.
- `VISUALIZER_REGISTRY` – for visualizers.

```python
@ATTENTION_REGISTRY.register("my_attn")
class MyAttention(BaseAttention):
    pass

attn_class = ATTENTION_REGISTRY.get("my_attn")
```

### `base.py` – Concrete base classes

- **`BaseAttention`** – all attention modules inherit from this (handles config, num_heads, head_dim).
- **`BaseCompressor`** – for compression algorithms. Provides parameter counting and summary.
- **`BaseVisualizer`** – base for visualizers (output directory management).

### `interfaces.py` – Abstract interfaces

Defines the **contracts** for pluggable components:

- `AttentionInterface`
- `CompressorInterface`
- `TrainerInterface`
- `VisualizerInterface`
- `DataLoaderInterface`

These are used for type checking and enforcing method signatures.

### `model_wrapper.py` – `UnifiedTransformerModel`

Universal wrapper for any Hugging Face transformer model. Features:

- Auto‑detect architecture (BERT, RoBERTa, DistilBERT, GPT‑2, etc.)
- `replace_attention(attention_type, rank, **kwargs)`
- `freeze_layers(num_layers)` / `unfreeze_all()`
- `get_attention_maps()`, `get_hidden_states()`
- `get_parameter_count()`
- `save_pretrained()` and `from_pretrained()` (task‑aware)

### `architecture.py` – Detection & replacement

- `detect_architecture(model)` → `'bert'`, `'roberta'`, `'gpt2'`, etc.
- `get_attention_modules(model)` → list of (name, module) for attention layers.
- `replace_attention_module(model, new_class, rank, **kwargs)` – replaces all attentions.

### `experiment_tracker.py` – `ExperimentTracker`

Logs metrics, saves config, checkpoints, and generates basic HTML reports.

```python
tracker = ExperimentTracker("results", "run1")
tracker.log_metrics({"loss": 0.23, "acc": 0.91}, step=10)
tracker.save_checkpoint(model, optimizer, epoch=3)
tracker.generate_report()
```

### `config.py` – Config loader

```python
config = load_config("experiment.yaml")
```

### `attention_utils.py` – Standalone attention helpers

- `replace_attention_in_model(model, attention_type, rank, copy_weights)`
- `get_attention_module_names(model)`
- `count_attention_modules(model)`

## 🔧 Integration with the Rest of the Toolkit

The `core` package is used by all other modules:
- `framework.attentions` registers attention classes via `ATTENTION_REGISTRY`.
- `framework.compression` registers compressors via `COMPRESSOR_REGISTRY`.
- Training scripts use `UnifiedTransformerModel` and `ExperimentTracker`.
- Architecture‑specific logic (layer freezing, attention replacement) uses `architecture.py`.

## 📜 Dependencies

- `torch`, `transformers`, `numpy`, `pandas`, `pyyaml`

## 📄 License

Part of the **Efficient Transformer Toolkit** – MIT License.
```

---

## 📝 Commit Message

```
