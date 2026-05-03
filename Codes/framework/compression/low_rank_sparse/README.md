# 🗜️ Compression Engine

A comprehensive suite of **20+ compressor classes** that reduce model size through low‑rank factorisation, pruning, quantisation, and more.  
All compressors inherit from `BaseCompressor`, return `(model, stats)`, and are registered in `COMPRESSOR_REGISTRY` for instant access via the CLI or Python scripts.

## 📦 Available Compressors

### Low‑Rank Factorisation

| Registry Key | Class | Description |
|--------------|-------|-------------|
| `svd` | *SVDLinearLayer* (standalone) | Truncated SVD per layer |
| `salr` | `SALRCompressor` | Sparse low‑rank + structured pruning |
| `saes` | `SAESCompressor` | Inter‑layer error‑compensated SVD |
| `mgie` | `MGIECompressor` | Entropy‑driven rank selection |
| `gfws` | `GFWSCompressor` | Fisher‑weighted SVD (uses gradients) |
| `erc` | `ERCCompressor` | Error‑controlled SVD |
| `dobi` | `DobiSVDCompressor` | Learnable rank ratio (differentiable) |
| `tt` | `TTCompressor` | Tensor Train decomposition |
| `soka` | `SoKACompressor` | Kronecker product (SoKA) approximation |
| `nmf` | `NMFCompressor` | Non‑negative Matrix Factorization |
| `rpca` | `RPCACompressor` | Robust PCA (low‑rank + sparse) |
| `gregodec` | `GreGoDecCompressor` | Greedy low‑rank + sparse decomposition |
| `frandi` | `FRanDICompressor` | Feature‑Regression aligned low‑rank |
| `enlighten` | `ENLightenCompressor` | Low‑rank SVD + 2:4 structured sparsity |

### Quantisation & Joint Methods

| Registry Key | Class | Description |
|--------------|-------|-------------|
| `uniql` | `UniQLCompressor` | SVD + uniform quantisation of factors |
| `mlorq` | `MLoRQCompressor` | Multi‑layer optimal rank & bit assignment |
| `kmeans` | `KMeansCompressor` | Weight clustering via k‑means |
| `faq` | `FAQQuantizer` | Future‑aware quantisation (activation‑driven) |
| `coopq` | `CoopQQuantizer` | Shapley‑driven per‑layer bit allocation |
| `astro` | `AstroQuantizer` | Simple uniform quantisation |
| `slim` | `SLiMCompressor` | One‑shot quantisation + 2:4 sparsity + low‑rank |

### Pruning & Structured Sparsity

| Registry Key | Class | Description |
|--------------|-------|-------------|
| `sparsegpt` | `SparseGPTCompressor` | One‑shot second‑order pruning (external) |
| `slicegpt` | `SliceGPTCompressor` | PCA‑based dimension reduction (external) |
| `come` | `CoMeCompressor` | Layer concatenation (external) |
| `flap` | `FLAPCompressor` | Fluctuation‑based adaptive structured pruning |
| `mamba_shedder` | `MambaShedderCompressor` | Sensitivity‑based structured pruning |
| `oats` | `OATSCompressor` | Outlier‑Aware Sparse Low‑Rank Decomposition |
| `proxsparse` | `ProxSparsePruner` | Proximal gradient for semi‑structured sparsity |
| `ast_pruner` | `ASTPruner` | 2:4 block sparsity |
| `armor_pruner` | `ARMORPruner` | 2:4 sparse + low‑rank error correction |
| `admm_pruner` | `ADMMPruner` | ADMM soft thresholding |
| `admm` | `ADMMCompressor` | ADMM‑based pruning + quantisation |
| `replaceme` | `ReplaceMeCompressor` | Remove last N layers with linear replacement |
| `bonsai` | `BonsaiPruner` | Iterative magnitude pruning (IMP) |

### MoE Conversion

| Registry Key | Class | Description |
|--------------|-------|-------------|
| `tomoe` | `ToMoECompressor` | Converts dense FFN → MoE via weight clustering |
| `cur_moe` | `CURMoECompressor` | CUR decomposition of expert layers |

### Utilities (standalone)

- `HuffmanCoder` – lossless entropy coding for weights
- `nli_softmax` – quantisation‑friendly softmax approximation
- `SVDLinearLayer` – drop‑in replacement for `nn.Linear` using truncated SVD
- `get_adaptive_ranks` / `apply_adaptive_svd` – pattern‑based rank assignment

---

## 🚀 Quick Start

### Compress via the Registry

```python
from framework.compression import COMPRESSOR_REGISTRY

# Low‑rank compression with automatic rank selection
CompressorClass = COMPRESSOR_REGISTRY.get("mgie")
compressor = CompressorClass(model)
compressed_model, stats = compressor.compress()
print(stats)
```

### Compress from the CLI

```bash
python cli.py compress --method mgie --model distilbert-base-uncased
python cli.py compress --method tt --target-rank 16 --model bert-base-uncased
python cli.py compress --method slim --bits 4 --target-rank 8
```

### Replace a single layer

```python
from framework.compression import SVDLinearLayer

svd_layer = SVDLinearLayer(original_linear_layer, rank=32)
```

---

## ⚙️ Common Parameters

| Parameter | Typical default | Meaning |
|-----------|-----------------|---------|
| `rank` / `target_rank` | 64 | Number of singular components to keep |
| `sparsity` | 0.5 | Fraction of weights to prune |
| `bits` | 8 | Quantisation bit‑width |
| `num_experts` | 4 | Number of experts for MoE conversion |
| `top_k` | 2 | Active experts per token |

---

## 📊 Parameter Savings (768×768 linear layer)

| Method | Rank/Bits | Parameters (approx) | Reduction |
|--------|-----------|--------------------|-----------|
| Original | – | 590,000 | 1× |
| SVD (rank 64) | 64 | 98,304 | 6× |
| TT (rank 16) | 16 | 24,576 | 24× |
| SoKA (rank 8) | 8 | 12,288 | 48× |
| SLiM (4‑bit, 2:4, rank 8) | 4/8 | ~12k effective | ~50× |
| NMF (rank 64) | 64 | 98,304 | 6× |

---

## 🔌 Integration

All compressors inherit from `BaseCompressor`, so they work identically:

```python
model, stats = compressor.compress()
# stats = {
#     'original_params': int,
#     'compressed_params': int,
#     'compression_ratio': float,
#     'parameters_saved': int,
#     'reduction_percentage': float
# }
```

They can be chained (compress then prune, etc.) and used within any training or inference script.

---

## 🧪 Adding Your Own Compressor

1. Create a new file in `framework/compression/`.
2. Inherit from `BaseCompressor` and implement `compress()` → `(model, stats)`.
3. Decorate the class with `@COMPRESSOR_REGISTRY.register("your_key")`.
4. Import it in `__init__.py`.

Your compressor will automatically appear in the CLI and in `COMPRESSOR_REGISTRY.keys()`.

---

*Compress smarter, not harder.*
