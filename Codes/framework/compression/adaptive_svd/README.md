# AdaSVD, AASVD, and Adaptive Rank Compressors

This package provides three tools for applying SVD‑based compression in an intelligent, adaptive way:

- **AdaSVD** – assigns a per‑layer rank based on the importance of that layer (nuclear norm), then iteratively refines the factor matrices.
- **AASVD** – “anchors” the SVD by first projecting the weight matrix onto the principal directions of the input activations, preserving the most functional directions.
- **Adaptive Rank Utilities** – helper functions that map layer name patterns to predefined ranks, and apply SVD compression accordingly.

All compressors inherit from `BaseCompressor`, return `(model, stats)`, and are registered in `COMPRESSOR_REGISTRY`.

---

## 🚀 Quick Start

### Compress via the Registry

```python
from framework.compression import COMPRESSOR_REGISTRY

# Importance‑driven rank assignment + iterative refinement
compressor = COMPRESSOR_REGISTRY.get("adasvd")(model, target_param_ratio=0.3)
model, stats = compressor.compress()

# Anchor‑based SVD (needs a tokenizer for calibration)
compressor = COMPRESSOR_REGISTRY.get("aasvd")(model, tokenizer=tokenizer, rank=64)
model, stats = compressor.compress()

# Pattern‑based rank assignment
from framework.compression import get_adaptive_ranks, apply_adaptive_svd
rank_map = get_adaptive_ranks(patterns=['q_lin', 'k_lin', 'out_lin'])
n = apply_adaptive_svd(model, rank_map)
```

### Compress via CLI

```bash
python cli.py compress --method adasvd --target-param-ratio 0.3
python cli.py compress --method aasvd --rank 64   # needs tokenizer; will use default calib text
```

---

## 🧬 Detailed Descriptions

### 1. AdaSVD – Adaptive Iterative SVD (`adasvd`)

- **How it works:**
  1. Compute the *importance* of each linear layer as the sum of its singular values (nuclear norm).
  2. Distribute a global parameter budget (given by `target_param_ratio`) proportionally to these importances – more important layers get higher ranks.
  3. Truncate the SVD to the assigned rank.
  4. Optionally refine the factors `U` and `V` by alternating least‑squares steps to further reduce reconstruction error.

- **Parameters:**
  - `target_param_ratio` – fraction of total parameters to **remove** (0‑1, default 0.3). A value of 0.3 means roughly 30% of original parameters will be pruned.
  - `num_iters` – number of alternating refinement steps (default 5).

- **When to use:** When different layers have different sensitivity to compression and you want to preserve the most critical ones while aggressively reducing others.

### 2. AASVD – Anchored Adaptive SVD (`aasvd`)

- **How it works:**
  1. Feed a few calibration texts through the model and capture the input activations of each linear layer.
  2. Compute the principal directions (top left singular vectors) of those input activations – this is the *anchor*.
  3. Project the weight matrix onto the anchor subspace: `W_anchored = W @ anchor @ anchor^T`.
  4. Perform truncated SVD on the anchored weight matrix, so that the low‑rank approximation preserves the directions that are most important for the actual data distribution.

- **Parameters:**
  - `tokenizer` – HuggingFace tokenizer (required).
  - `rank` – target rank for the SVD (default 64).
  - `calibration_texts` – list of strings for activation capture (default: one example sentence).

- **When to use:** When you have access to a few representative inputs and want a compression that is tightly aligned with the model’s usage.

### 3. Adaptive Rank Utilities (`adaptive_ranks.py`)

- **`get_adaptive_ranks(patterns)`** – creates a dictionary that maps each pattern (e.g., `'q_lin'`, `'out_lin'`) to a rank. Patterns containing `'out'` or `'output'` get higher rank (64), attention projections (`'q_lin'`, `'k_lin'`, `'v_lin'`) get lower rank (32), and others get a default rank (48).
- **`apply_adaptive_svd(model, rank_map)`** – iterates over all `nn.Linear` layers, matches their name against the patterns in `rank_map`, and replaces them with `SVDLinearLayer` (from `svd_linear.py`) at the corresponding rank. Returns the number of replaced layers.

- **When to use:** For quick, rule‑based compression without training or calibration, especially when you know which layers are more sensitive (e.g., output projections often need higher precision).

---

## ⚙️ Common Parameters

| Parameter | Typical default | Meaning |
|-----------|-----------------|---------|
| `target_param_ratio` | 0.3 | Overall fraction of parameters to remove (AdaSVD) |
| `rank` | 64 | Target SVD rank (AASVD / adaptive ranks) |
| `num_iters` | 5 | Refinement steps (AdaSVD) |
| `tokenizer` | required | Tokenizer for calibration (AASVD) |
| `calibration_texts` | `["The quick brown fox..."]` | Texts for activation capture (AASVD) |

---

## 📂 File Structure

```
framework/compression/
├── adasvd_compressor.py       # AdaSVDCompressor
├── aa_svd_compressor.py       # AASVDCompressor
├── adaptive_ranks.py          # get_adaptive_ranks, apply_adaptive_svd
└── __init__.py                # exports all classes
```

---

## 🔌 Integration

All compressors are registered and can be used identically to the other compression methods. They return a tuple `(compressed_model, stats_dict)` where `stats_dict` contains standard keys like `original_params`, `compressed_params`, and `compression_ratio`.

---

*Adaptive compression – more rank where it matters.*
