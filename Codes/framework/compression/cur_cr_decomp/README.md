
# CR, CUR & MoDeGPT Compressors

This package provides compressors based on **column‑row low‑rank approximations**:
- **CR** (Interpolative Decomposition): keeps actual columns of the weight matrix.
- **CUR**: selects both columns and rows, approximating `W ≈ C @ U @ R`.
- **CUR‑MoE**: applies CUR to Mixture‑of‑Experts layers.
- **MoDeGPT**: jointly decomposes pairs of attention projections (e.g., Q & K) in transformer blocks.

All compressors are registered in `COMPRESSOR_REGISTRY` and return `(model, stats)`.

## Usage

```python
from framework.compression import COMPRESSOR_REGISTRY

# Interpolative decomposition
compressor = COMPRESSOR_REGISTRY.get("cr")(model, rank=32)
model, stats = compressor.compress()

# CUR decomposition
compressor = COMPRESSOR_REGISTRY.get("cur")(model, rank=32, method='leverage')
model, stats = compressor.compress()

# CUR for MoE
compressor = COMPRESSOR_REGISTRY.get("cur_moe")(model, rank=32)
model, stats = compressor.compress()

# MoDeGPT (joint Q/K decomposition)
compressor = COMPRESSOR_REGISTRY.get("modegpt")(model, rank=32)
model, stats = compressor.compress()
```

## Parameters

| Method | Parameter | Description |
|--------|----------|-------------|
| CR | `rank` | Number of columns to keep |
| CUR | `rank`, `method` | Number of rows/columns; 'leverage' or 'uniform' |
| CUR‑MoE | `rank`, `method` | Same as CUR, applied to expert weights |
| MoDeGPT | `rank` | Rank for joint SVD of Q & K pairs |

## How They Work

- **CR** uses scipy's `interpolative.interp_decomp` to select a subset of columns that approximate the full matrix.
- **CUR** samples columns and rows based on leverage scores (L2 norms), then solves for the middle matrix `U` using pseudo‑inverses.
- **CUR‑MoE** applies the CUR algorithm to all expert weight matrices inside MoE modules.
- **MoDeGPT** concatenates the weights of a Q‑K pair (found by scanning module attributes), applies SVD, and splits back.

## Performance Notes

- CR and CUR retain the original weight values, which can be useful for interpretability.
- The compression ratio depends on the chosen rank.
- CUR‑MoE and MoDeGPT are structural compressors – they assume a specific model layout (MoE, Q/K pairs).

