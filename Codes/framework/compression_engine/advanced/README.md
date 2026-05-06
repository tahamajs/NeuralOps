# 🎯 Zero‑Shot & Budget‑Aware Compressors

This package provides **four intelligent compressors** that automatically decide *how much* to compress each layer. Instead of applying a uniform rank everywhere, they use energy thresholds, entropy, per‑layer parameter budgets, or one of 14 built‑in strategies.

All compressors inherit from `BaseCompressor`, return `(model, stats)`, and are registered in `COMPRESSOR_REGISTRY` for instant access via the CLI or Python scripts.

## 📦 Available Compressors

| Registry Key | Class | Core Technique |
|--------------|-------|---------------|
| `adaptive_svd` | `AdaptiveTransformerCompressor` | Energy‑threshold SVD on MultiheadAttention layers |
| `advanced_zero_shot` | `AdvancedZeroShotCompressor` | 14 pre‑defined rank assignment strategies |
| `entropy_budget` | `EntropyBudgetCompressor` | Global rank allocation via whitened activation entropy |
| `sdrm` | `SDRMCompressor` | Proportional rank budget for edge inference |

---

## 🚀 Quick Start

### Compress via the Registry

```python
from framework.compression import COMPRESSOR_REGISTRY

# Energy‑threshold SVD for attention layers
compressor = COMPRESSOR_REGISTRY.get("adaptive_svd")(
    model, energy_threshold=0.95, min_rank=16
)
model, stats = compressor.compress()

# Advanced zero‑shot – choose from 14 strategies
compressor = COMPRESSOR_REGISTRY.get("advanced_zero_shot")(
    model, method="hybrid_advanced"
)
model, stats = compressor.compress()

# Global entropy budget (needs calibration loader)
compressor = COMPRESSOR_REGISTRY.get("entropy_budget")(
    model, calibration_loader=calib_loader, total_budget_ratio=0.7
)
model, stats = compressor.compress()

# Proportional rank budget for edge devices
compressor = COMPRESSOR_REGISTRY.get("sdrm")(
    model, total_param_ratio=0.6
)
model, stats = compressor.compress()
```

### Compress via CLI

```bash
python cli.py compress --method adaptive_svd --energy-threshold 0.95
python cli.py compress --method advanced_zero_shot --method-strategy hybrid_advanced
python cli.py compress --method sdrm --total-param-ratio 0.6
```

---

## 🔬 Detailed Descriptions

### 1. AdaptiveTransformerCompressor (`adaptive_svd`)

- **What it does:** Iterates over all `nn.MultiheadAttention` layers in the model, computes the SVD of the in‑projection weight and the output weight, and chooses the rank by retaining a given fraction of the singular‑value energy.
- **Parameters:**
  - `energy_threshold` – fraction of singular‑value energy kept (default 0.95).
  - `min_rank` – minimum rank allowed (default 16).
- **When to use:** When you only want to compress the attention layers and let the rest of the model untouched, using a simple, proven energy criterion.

### 2. AdvancedZeroShotCompressor (`advanced_zero_shot`)

- **What it does:** Offers **14 ready‑to‑use rank‑assignment strategies**. You select a method name, and the compressor automatically decides the rank for each MultiheadAttention layer based on its singular values, layer position, or other heuristics.
- **Available methods:**

| Method | Description |
|--------|-------------|
| `standard_svd` | Energy‑threshold SVD per layer |
| `head_pruning` | Uniform low rank |
| `layer_skipping` | Lower rank for deeper layers |
| `quantized_svd` | Lower energy threshold (75%) |
| `blockwise_lowrank` | Decreasing rank per block of layers |
| `randomized_svd` | Fixed rank (16) for all layers |
| `tensor_train_decomp` | Exponentially decreasing rank with depth |
| `cp_decomposition` | Fixed small rank (6) – extreme compression |
| `dynamic_token_pruning` | Lower rank for later layers (beyond 60% depth) |
| `spectral_clustering_prune` | Low rank for low‑energy layers, high rank for high‑energy |
| `hash_quantization` | Fixed very low rank (8) |
| `moe_sparsification` | Higher rank for first half, 30% of that for second half |
| `hybrid_advanced` | Combination: exponential decrease in early layers, standard in middle, extreme pruning in deep layers |

- **Parameters:**
  - `method` – one of the above strings.
  - `base_threshold` – energy threshold used by many methods (default 0.90).
  - `min_rank` – minimum rank allowed (default 4).
- **When to use:** When you want to quickly experiment with many different rank schedules without writing custom code.

### 3. EntropyBudgetCompressor (`entropy_budget`)

- **What it does:**
  1. Collects input activation covariances using a calibration loader.
  2. For each Linear layer, computes the whitened weight matrix `M = W @ L` (where `L` is the Cholesky factor of the activation covariance).
  3. Performs SVD on `M` and calculates the entropy of the singular value distribution.
  4. Gathers all singular components from all layers, sorts them by *importance* (variance / entropy), and keeps the top `total_budget_ratio` fraction globally.
  5. Each layer then gets the rank equal to the number of its components that survived.
- **Parameters:**
  - `calibration_loader` – a DataLoader providing a few batches of calibration data.
  - `total_budget_ratio` – fraction of all singular components to keep globally (default 0.7).
- **When to use:** When you have access to calibration data and want a mathematically grounded global budget allocation that accounts for both the magnitude and the concentration of singular values.

### 4. SDRMCompressor (`sdrm`)

- **What it does:** Designed for **edge inference**. It computes the total parameter budget (`total_param_ratio × original_params`). Then, for each Linear layer, it allocates a rank such that the parameters of the low‑rank representation (`rank × (out_features + in_features)`) stay within that layer’s proportional budget.
- **Parameters:**
  - `total_param_ratio` – overall fraction of original parameters to keep (default 0.7).
- **When to use:** When you have a hard parameter budget (e.g., for on‑device deployment) and need a fast, deterministic method that guarantees the total size.

---

## ⚙️ Common Parameters

| Parameter | Used by | Meaning |
|-----------|---------|---------|
| `energy_threshold` | `adaptive_svd`, `advanced_zero_shot` (via base_threshold) | Fraction of singular‑value energy retained (0‑1) |
| `min_rank` | `adaptive_svd`, `advanced_zero_shot` | Minimum rank to avoid degenerate layers |
| `method` | `advanced_zero_shot` | Name of the pre‑defined strategy |
| `total_budget_ratio` | `entropy_budget` | Fraction of global singular components kept |
| `total_param_ratio` | `sdrm` | Fraction of total parameters to retain after compression |
| `calibration_loader` | `entropy_budget` | DataLoader for activation statistics |

---

## 📊 Comparison

| Compressor | Requires calibration? | Rank selection | Layer scope |
|------------|------------------------|---------------|-------------|
| `adaptive_svd` | No | Energy threshold per layer | Attention only |
| `advanced_zero_shot` | No | 14 heuristic strategies | Attention only |
| `entropy_budget` | Yes (DataLoader) | Entropy + global budget | All Linear layers |
| `sdrm` | No | Proportional parameter budget | All Linear layers |

---

## 🔌 Integration

All compressors are importable from `framework.compression` and can be used with any HuggingFace or custom PyTorch model.

```python
from framework.compression import AdaptiveTransformerCompressor

compressor = AdaptiveTransformerCompressor(model, energy_threshold=0.95, min_rank=8)
compressed_model, stats = compressor.compress()
print(stats['reduction_percentage'])
```

---

## 📁 File Structure

```
framework/compression/
├── adaptive_compressor.py
├── advanced_methods.py
├── entropy_budget_allocation.py
├── sdrm_compressor.py
└── __init__.py
```

---

*Intelligent compression – the right rank for every layer, automatically.*
