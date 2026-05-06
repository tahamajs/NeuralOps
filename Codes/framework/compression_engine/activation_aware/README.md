
# 🧬 Advanced Compression Suite

This package provides **12 state‑of‑the‑art compressors** that go beyond simple SVD, incorporating activation awareness, orthogonal dictionaries, Fourier transforms, entropy budgeting, and joint pair compression.

All compressors inherit from `BaseCompressor`, return `(model, stats)`, and are registered in `COMPRESSOR_REGISTRY` for instant access via the CLI or Python scripts.

## 📦 Available Compressors

| Registry Key | Class | Core Technique |
|--------------|-------|---------------|
| `swift_svd` | `SwiftSVDCompressor` | Closed‑form SVD with rank ratio |
| `swift_svd_act` | `SwiftSVDActivationCompressor` | Cross‑covariance activation‑aware SVD |
| `saes_svd_v2` | `SAESSVDCompressorV2` | Accumulated error suppression via fixed rank ratio |
| `subspace_projection` | `SubspaceProjectionCompressor` | Activation covariance + randomized SVD |
| `mgie_budget` | `MGIESVDBudgetCompressor` | Entropy‑guided rank allocation with total budget |
| `lsc` | `LSCCompressor` | Fourier truncation of weight matrix |
| `lillama` | `LiLLaMACompressor` | Local distillation via cross‑covariance |
| `kvt_weight` | `KVTWeightCompressor` | PCA + quantisation on weight rows |
| `flat_llm` | `FLATLLMCompressor` | PCA on activation outputs to prune dimensions |
| `compot` | `COMPOTCompressor` | Orthogonal dictionary learning from calibration |
| `a3` | `A3Compressor` | Grouped QK/OV/MLP SVD compression |
| `slicegpt_pca` | `SliceGPTPCACompressor` | PCA‑based dimension slicing |

---

## 🚀 Quick Start

### Compress via Registry

```python
from framework.compression import COMPRESSOR_REGISTRY

# Swift linear SVD compression
compressor = COMPRESSOR_REGISTRY.get("swift_svd")(model, rank_ratio=0.5)
model, stats = compressor.compress()

# Entropy‑guided rank allocation
compressor = COMPRESSOR_REGISTRY.get("mgie_budget")(model, total_rank_budget=500)
model, stats = compressor.compress()
```

### Compress via CLI

```bash
python cli.py compress --method swift_svd --rank-ratio 0.5
python cli.py compress --method flat_llm --keep-ratio 0.7
python cli.py compress --method mgie_budget --total-rank-budget 500
```

---

## 🔬 Detailed Descriptions

### SwiftSVD (`swift_svd`)
- **What it does:** Applies simple truncated SVD with a rank‑ratio.
- **Parameters:** `rank_ratio` (default 0.5).
- **When to use:** Fastest baseline for low‑rank compression.

### SwiftSVD Activation‑Aware (`swift_svd_act`)
- **What it does:** Computes the cross‑covariance between input and output activations using calibration data, then performs SVD on this matrix to obtain a low‑rank weight.
- **Parameters:** `tokenizer`, `rank`, `calibration_texts`.
- **When to use:** When you want to align compression with the actual data distribution.

### SAES‑SVD v2 (`saes_svd_v2`)
- **What it does:** Suppresses accumulated error by keeping a higher rank (default 70%).
- **Parameters:** `rank_ratio`.
- **When to use:** When you need a simple, high‑quality SVD baseline.

### Subspace Projection (`subspace_projection`)
- **What it does:** Estimates the input activation covariance, then uses randomized SVD in the whitened space to obtain a low‑rank weight.
- **Parameters:** `calibration_loader`, `rank_ratio`, `power_iters`.
- **When to use:** For extremely large layers where full SVD is too costly.

### MGIE‑Budget (`mgie_budget`)
- **What it does:** Allocates a total rank budget across all layers proportionally to the entropy of their singular values.
- **Parameters:** `total_rank_budget` (sum of ranks across all layers).
- **When to use:** When you have a global compression budget and want to intelligently distribute it.

### LSC (`lsc`)
- **What it does:** Applies 2D FFT to the weight matrix, keeps the largest Fourier coefficients, then inverse‑FFTs back.
- **Parameters:** `rank` (used to define the number of kept coefficients).
- **When to use:** For weights with strong frequency‑domain structure.

### LiLLaMA (`lillama`)
- **What it does:** Local distillation: computes input‑output cross‑covariance via calibration, then uses SVD on that matrix to obtain a low‑rank weight that minimises local reconstruction error.
- **Parameters:** `tokenizer`, `rank`, `calibration_texts`.
- **When to use:** Lightweight, data‑driven compression.

### KVT Weight (`kvt_weight`)
- **What it does:** Applies PCA to the weight rows (or columns), then quantises the principal components.
- **Parameters:** `bits`, `keep_eig` (PCA energy retained).
- **When to use:** When you need both compression and quantisation.

### FLAT‑LLM (`flat_llm`)
- **What it does:** Runs a forward pass with calibration data, collects output activations of each linear layer, performs PCA on those activations, and prunes the least‑varying output dimensions (rows of the weight matrix).
- **Parameters:** `tokenizer`, `keep_ratio`, `calibration_texts`.
- **When to use:** For structured pruning that directly removes entire neurons based on their contribution.

### COMPOT (`compot`)
- **What it does:** Learns an orthogonal dictionary from calibration inputs, then projects the layer’s output onto this dictionary, effectively decomposing the weight into a dictionary component and a smaller coefficient matrix.
- **Parameters:** `tokenizer`, `rank`, `calibration_texts`.
- **When to use:** When you want a data‑dependent orthogonal factorisation.

### A3 (`a3`)
- **What it does:** Groups QK, OV, and MLP linear layers and applies SVD compression to each.
- **Parameters:** `rank`.
- **When to use:** Quick, targeted compression of attention and FFN blocks.

### SliceGPT PCA (`slicegpt_pca`)
- **What it does:** Collects activations from a chosen layer, runs PCA, and then slices the input dimension of all subsequent linear layers to the number of principal components that explain (1‑sparsity) of the variance.
- **Parameters:** `calibration_loader`, `sparsity`.
- **When to use:** Global dimension reduction across the model.

---

## ⚙️ Common Parameters

| Parameter | Typical default | Meaning |
|-----------|-----------------|---------|
| `rank` / `rank_ratio` | 64 / 0.5 | Target rank or fraction of dimensions kept |
| `tokenizer` / `calibration_loader` | required for activation‑aware methods | Source of calibration data |
| `calibration_texts` | `["The quick brown fox…"]` | Texts for calibration |
| `bits` | 4‑8 | Quantisation bit‑width |
| `keep_ratio` | 0.7 | Fraction of dimensions to keep (FLAT‑LLM) |
| `total_rank_budget` | 1000 | Sum of ranks across all layers (MGIE‑Budget) |

---

## 📊 Comparison

| Compressor | Calibration needed | Structure preserved | Speed |
|------------|--------------------|---------------------|-------|
| SwiftSVD | No | No (low‑rank) | Fast |
| SwiftSVD‑Act | Yes | No | Fast |
| SAES‑SVD v2 | No | No | Fast |
| Subspace Proj | Yes | No | Medium |
| MGIE‑Budget | No | No | Fast |
| LSC | No | No (Fourier) | Medium |
| LiLLaMA | Yes | No | Fast |
| KVT Weight | No | PCA + quant | Fast |
| FLAT‑LLM | Yes | **Yes** (prunes rows) | Fast |
| COMPOT | Yes | Orthogonal dict | Medium |
| A3 | No | No | Fast |
| SliceGPT PCA | Yes | **Yes** (global dim reduction) | Medium |

---

## 🔌 Integration

All compressors are importable from `framework.compression` and can be used interchangeably with any HuggingFace model:

```python
from framework.compression import FLATLLMCompressor
compressor = FLATLLMCompressor(model, tokenizer, keep_ratio=0.7)
model, stats = compressor.compress()
```

---

*Precision, speed, or structure – pick the compressor that fits your deployment.*