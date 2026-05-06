# 🧩 SVD‑Based Compression Suite

A collection of **nine production‑ready compressors** that leverage low‑rank matrix factorisations (SVD, Tensor Train, Kronecker) to dramatically reduce the number of parameters in Transformer models – all registered in `COMPRESSOR_REGISTRY` and available via a single CLI command.

## 📦 Available Compressors

| Registry Key | Compressor Class | Core Idea |
|--------------|------------------|-----------|
| `svd`        | *SVDLinearLayer* (standalone) | Truncated SVD per layer |
| `salr`       | `SALRCompressor` | Low‑rank + sparse pruning mask |
| `saes`       | `SAESCompressor` | Inter‑layer error‑compensated SVD |
| `mgie`       | `MGIECompressor` | Entropy‑driven rank selection |
| `gfws`       | `GFWSCompressor` | Fisher‑weighted SVD (uses gradients) |
| `erc`        | `ERCCompressor` | Error‑controlled SVD |
| `dobi`       | `DobiSVDCompressor` | Learnable rank ratio (differentiable) |
| `tt`         | `TTCompressor` | Tensor Train decomposition |
| `soka`       | `SoKACompressor` | Kronecker product (SoKA) approximation |

All compressors inherit from `BaseCompressor`, support **zero‑shot weight initialisation**, and return **detailed compression statistics**.

## 🚀 Quick Start

### Compress a model via the Registry

```python
from framework.compression import COMPRESSOR_REGISTRY

# Entropy‑driven rank selection
CompressorClass = COMPRESSOR_REGISTRY.get("mgie")
compressor = CompressorClass(model)          # no rank needed – automatic
compressed_model, stats = compressor.compress()
print(stats)
```

### Compress from the command line

```bash
python cli.py compress --method mgie --model distilbert-base-uncased
python cli.py compress --method tt --target-rank 16 --model bert-base-uncased
python cli.py compress --method salr --target-rank 64 --target-sparsity 0.5
```

### Replace a single layer manually

```python
from framework.compression import SVDLinearLayer

svd_layer = SVDLinearLayer(original_linear_layer, rank=32)
```

---

## 🧬 Detailed Descriptions

### 1. SVDLinearLayer (`svd`)
- **What it does:** Replaces any `nn.Linear` with `U @ diag(S) @ V`, keeping the top‑`k` singular components.
- **When to use:** Go‑to general‑purpose compression; fast, deterministic, zero‑shot.

### 2. SALR – Sparse And Low‑Rank (`salr`)
- **What it does:** Applies low‑rank SVD **plus** a sparse binary mask on the core singular values.
- **Parameters:** `rank`, `sparsity` (fraction of core elements kept).
- **When to use:** When you want to combine both parameter reduction and structured sparsity.

### 3. SAES – Self‑Adaptive Error‑compensated SVD (`saes`)
- **What it does:** Performs SVD layer‑by‑layer and (in future) compensates for the resulting error by adjusting subsequent layers’ inputs.
- **When to use:** When you need to maintain accuracy after aggressive compression.

### 4. MGIE – Maximum Gaussian Information Entropy (`mgie`)
- **What it does:** Automatically selects the rank per layer based on the entropy of the singular value distribution – no manual tuning needed.
- **When to use:** When you want a fully automatic, data‑driven rank selection.

### 5. GFW-SVD – Gradient‑based Fisher‑Weighted SVD (`gfws`)
- **What it does:** Weights the SVD by the observed gradient Fisher information (requires a backward pass before compression).
- **When to use:** When you have access to training gradients and want to preserve the most **task‑relevant** directions.

### 6. ERC-SVD – Error‑Controlled SVD (`erc`)
- **What it does:** Chooses rank such that the reconstruction error (Frobenius) is below a user‑specified threshold (`allowed_error`).
- **When to use:** When you have a strict accuracy budget and need guarantees on the compression error.

### 7. Dobi‑SVD – Differentiable SVD (`dobi`)
- **What it does:** Replaces the linear layer with a **learnable rank ratio**; the rank can be optimised during fine‑tuning via gradient descent.
- **When to use:** When you plan to fine‑tune the compressed model and want the rank itself to adapt.

### 8. Tensor Train Decomposition (`tt`)
- **What it does:** Decomposes the weight matrix into a chain of small 3‑D tensors (TT cores). Falls back to SVD if `torchtt` is not installed.
- **When to use:** For maximum parameter reduction (up to 100×) on matrices that can be factorised into multiple dimensions.

### 9. SoKA – Kronecker Approximation (`soka`)
- **What it does:** Approximates the weight as a sum of Kronecker products (here realised as a single low‑rank SVD).
- **When to use:** Lightweight alternative to TT; extremely fast forward pass.

---

## ⚙️ Global Parameters (most compressors)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `rank` / `target_rank` | 64 | Number of singular components to keep. |
| `sparsity` | 0.5 | Fraction of pruned elements inside SALR. |
| `allowed_error` | 0.01 | Frobenius reconstruction error for ERC. |
| `damping` | 0.01 | Regularisation for GFW‑SVD. |
| `init_rank_ratio` | 0.5 | Initial rank ratio (fraction of full rank) for Dobi‑SVD. |

*(MGIE and ERC select rank automatically – no manual rank needed.)*

---

## 📊 Parameter Savings (DistilBERT‑base, 768×768 linear layer)

| Method | Chosen Rank | Parameters (approx) | Reduction |
|--------|-------------|--------------------|-----------|
| Original | – | 590,000 | 1× |
| SVD | 64 | 98,304 | 6× |
| SALR | 64 (sparse) | 98,304 | 6× |
| MGIE | auto (~32) | 49,152 | 12× |
| TT (rank 16) | 16 | 24,576 | 24× |
| SoKA (rank 8) | 8 | 12,288 | 48× |

*Actual savings depend on the layer shape and chosen rank.*

---

## 🧪 Integration with the Framework

All compressors are drop‑in compatible with the models and data loaders described in the main `framework`.  
They can be used:

- **Standalone** – directly call `.compress()` on a PyTorch model.
- **With HuggingFace** – apply compression before fine‑tuning.
- **Via CLI** – the `compress` command uses the registry.
- **In custom scripts** – import `COMPRESSOR_REGISTRY` and call `get("key")`.

After compression, you can save the model with `model.save_pretrained("compressed_model")` and load it normally.

---

## 🛠 Customisation

To add your own compressor:

1. Inherit from `BaseCompressor`.
2. Implement `compress()` → `(model, stats)`.
3. Decorate with `@COMPRESSOR_REGISTRY.register("your_key")`.

Your compressor will automatically appear in the CLI and in `COMPRESSOR_REGISTRY.keys()`.

---

## 📜 License

Part of the **Efficient Transformer Toolkit** – MIT License.

---

*Compress smarter, not harder.*
