Below is the **cleaned, deduplicated `framework/utils` package** — every utility that the framework actually uses, with consistent interfaces, no duplication, and full integration with the rest of the toolkit.  
The older, redundant files (`replace_attention.py`, `save_results.py`, `results_saver.py`) have been removed because the core modules already provide those capabilities.  
The `architecture.py` helper is kept as a convenient legacy module but is superseded by `framework.core.architecture`.

---

## File Structure

```
framework/utils/
├── __init__.py              # Public API
├── seed.py                  # Reproducibility
├── device.py                # Auto‑select torch device
├── helpers.py               # Common small helpers
├── logger.py                # Logging setup
├── metrics.py               # Accuracy, F1, Matthews, Pearson, Spearman
├── paths.py                 # Project‑root‑relative paths
├── profiler.py              # ModelProfiler, latency & memory measurement
├── layer_norm_fix.py        # Fix HuggingFace LayerNorm naming
├── config.py                # YAML / JSON config loading
├── checkpoint.py            # Simple checkpoint manager
├── architecture.py          # Legacy architecture detection (deprecated)
└── README.md                # This file
```

---

## Full README – `framework/utils`

### `framework.utils` – Shared Utilities

A collection of **standalone, reusable utility functions** that underpin the whole Efficient Transformer Toolkit.  
No heavy dependencies beyond PyTorch, NumPy, and SciPy.

#### Quick Start

```python
from framework.utils import set_all_seeds, get_device, count_parameters, setup_logger

set_all_seeds(42)
device = get_device()
model = ... # your model
print(f"Trainable parameters: {count_parameters(model)}")
```

---

### Module Reference

#### `seed.py`
- `set_all_seeds(seed: int)` – seeds Python, NumPy, PyTorch (CPU & CUDA), and disables cuDNN auto‑tuning for determinism.

#### `device.py`
- `get_device(device_str: str = "auto") -> torch.device` – returns `cuda`, `mps`, or `cpu` depending on availability. Pass `"cuda:0"` for a specific GPU.

#### `helpers.py`
- `set_seed(seed, deterministic=False)` – wraps `set_all_seeds` + optional deterministic flags.
- `format_time(seconds)` – e.g., `"1h 23m 45s"`.
- `format_number(num)` – e.g., `"1.2M"`.
- `count_parameters(model, trainable_only=True) -> int`.
- `ensure_dir(path) -> Path` – creates directory.
- `save_json(data, path)`, `load_json(path)` – JSON I/O.
- `flatten_dict(d, sep='.')` / `unflatten_dict(d, sep='.')` – nested dict ↔ flat dict.

#### `logger.py`
- `setup_logger(name, level, log_file, format_string)` – returns a configured `logging.Logger` with console and optional file handler.
- `get_logger(name)` – returns the logger for that name.
- `set_log_level(logger_name, level)` – adjust level at runtime.

#### `metrics.py`
- `compute_accuracy(preds, labels) -> float`
- `compute_f1(preds, labels, average='weighted') -> float`
- `compute_matthews_corrcoef(preds, labels) -> float`
- `compute_pearson(preds, labels) -> float`
- `compute_spearman(preds, labels) -> float`
- `compute_precision_recall(preds, labels, average='weighted') -> Dict`
- `compute_metrics_for_task(predictions, labels, task_name, is_regression=False) -> Dict` – automatic metric selection for GLUE tasks.

#### `paths.py`
- `get_project_root() -> Path` – absolute root of the repository.
- `get_datasets_dir() -> Path` – `root / "datasets"`.
- `get_glue_dir() -> Path` – `datasets / "glue"`.
- `resolve_path(relative_path) -> str` – absolute path from root.

#### `profiler.py`
- `ModelProfiler(model)` – registers forward hooks on attention/FFN layers and records time & memory per layer.
  - `start_profiling()`, `stop_profiling()`, `get_report()`, `save_report(path)`, `print_report()`.
- `measure_inference_latency(model, dataloader, device, num_samples, warmup) -> Dict` – mean, std, percentiles, throughput.
- `measure_memory_usage(model, dataloader, device, num_samples) -> Dict` – peak GPU memory.
- `profile_context(model, profiler=None)` – context manager that profiles a block of code.

#### `layer_norm_fix.py`
- `fix_layernorm_naming(model)` – renames `beta` → `bias` and `gamma` → `weight` for HuggingFace models that use alternate names.

#### `config.py`
- `load_config(path) -> dict` – loads YAML or JSON.
- `save_config(config, path)` – saves as YAML or JSON.
- `update_config(config, updates)` – deep merge updates into config dict.
- `merge_configs(base, override)` – returns a new merged dict.

#### `checkpoint.py`
- `CheckpointManager(checkpoint_dir)` – simple JSON‑based checkpoint saver/loader for experiment progress.
  - `save_checkpoint(experiment_id, results, name, metadata)`
  - `load_checkpoint(name) -> (experiment_id, results, metadata)`
  - `delete_checkpoint(name)`, `list_checkpoints()`, `get_latest_checkpoint()`

#### `architecture.py` (legacy)
- `detect_architecture(model_name: str) -> str` – guess architecture from model name.
- `get_layer_patterns(arch)`, `get_compression_targets(model_name, include_ffn)`, `get_model_info(model_name)`, `get_layer_count(model)`, `get_hidden_size(model)`.  
  **Note:** The official architecture detection is in `framework.core.architecture`. This module is kept for backward compatibility but will be removed in a future version.

---

### Changes from the Previous Mess

- **Removed** `replace_attention.py` – superseded by `framework.core.attention_utils.inject_attention()`.
- **Removed** `save_results.py` and `results_saver.py` – superseded by `framework.results_saver.ResultsSaver` and `framework.training.ResultsManager`.
- **Deduplicated** logging, metrics, and config helpers.
- **All imports are now relative** and the package can be used as `from framework.utils import ...`.

---

### Dependencies

- `torch`
- `numpy`
- `scipy`
- `scikit-learn` (for Matthews, F1, precision/recall)
- `matplotlib` (for profiler plots – optional)
- `pyyaml` (optional, for YAML config)

All of these are already part of the framework’s requirements.

---

### Integration

Every utility is designed to be used directly by the other framework modules:

- `set_all_seeds` and `get_device` are called at the start of every training script.
- `setup_logger` is used by `ResultsManager` and the Django callback.
- `compute_metrics_for_task` is used by the training loop and the GLUE benchmark runner.
- `ModelProfiler` is used by the `SVDProfiler` and the analysis callbacks.
- `fix_layernorm_naming` is applied to every loaded model to prevent loading errors.


