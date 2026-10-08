# Migration Guide: NeuralOps v1.0.0

This guide covers architectural migrations, module reorganizations, deprecated aliases, and new unified CLI workflows introduced in NeuralOps v1.0.0.

---

## 1. Modular Reorganization Overview

In v1.0.0, large monolithic ("god") files have been decomposed into domain-focused subpackages while retaining **100% backward compatibility** via shim re-exports.

| Old File Path | New Package Location | Re-export Status |
|:---|:---|:---|
| `neuralops/data/benchmark_datasets.py` | `neuralops/data/benchmark_datasets/` | Re-exported via shim |
| `neuralops/runners/run_monitor_server.py` | `neuralops/runners/monitor/` | Re-exported via shim |
| `neuralops/trainer/optimizers.py` | `neuralops/trainer/optimizers/` | Re-exported via shim |
| `neuralops/utils/compat.py` | `neuralops/utils/compat/` | Re-exported via shim |
| `neuralops/results_saver/saver.py` | `neuralops/results_saver/` | Re-exported via shim |
| `neuralops/analysis/generate_all_modules_analysis.py` | `neuralops/analysis/module_analysis/` | Re-exported via shim |

---

## 2. Benchmark CLI Migration

The legacy evaluation and sweep entry points (`run_sweep.py`, standalone runner scripts) are now consolidated under the canonical benchmark CLI:

### Running Sweeps
```bash
# Old:
python -m neuralops.runners.run_sweep --module all

# New v1.0.0 CLI:
python -m neuralops.benchmarks run --all
```

### Dry-run & Execution Plan
```bash
python -m neuralops.benchmarks plan --all-attention
```

### Listing Capabilities
```bash
python -m neuralops.benchmarks list --family attention --requires-gpu
```

---

## 3. Registry & Lazy Loading

Registry lookups now lazy-load implementations on first access, dramatically reducing cold import latency to under 3 seconds.

```python
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

# Canonical access:
attn_cls = ATTENTION_REGISTRY.get("flash_attention_2")
comp_cls = COMPRESSOR_REGISTRY.get("svd_compressor")
```

### Deprecated Aliases
All 641 legacy aliases continue to resolve, but accessing them emits a `DeprecationWarning`. Users are encouraged to migrate to canonical snake_case names:
- `fast_softmax` → `fast_softmax_attention`
- `cur_linear` → `cur_compressor`
