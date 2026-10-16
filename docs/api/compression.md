# Compression Engines API Reference

NeuralOps registers **72 canonical compression engines** unified under `neuralops.core.registry.COMPRESSOR_REGISTRY`.

All compressors subclass `neuralops.core.base.BaseCompressor` and implement calibrated low-rank factorization, structured pruning, quantization, or tensor network decompositions.

---

## 1. Algorithmic Taxonomy

| Family | Count | Primary Implementations | Typical Compression Ratio | Speedup |
| :--- | :---: | :--- | :---: | :---: |
| **Low-Rank Factorization (SVD)** | 25 | `fwsvd`, `adasvd_compressor`, `asvd_compressor`, `direct_svd`, `residual_svd_compressor`, `swift_svd`, `dobi_svd`, `erc_svd` | 4x – 24x | 1.3x – 2.4x |
| **Quantization & Integer Encoding** | 12 | `faq_quantizer`, `rotational_slicegpt_int4`, `uniql_compressor`, `astroquant`, `huffman_coder` | 4x – 8x | 1.4x – 2.0x |
| **Structured Pruning** | 15 | `slicegpt`, `slicegpt_pca`, `sparsegpt_engine`, `wanda_engine`, `admm_pruner`, `flap_pruner` | 2x – 4x | 1.2x – 1.8x |
| **KV-Cache Optimization** | 8 | `aatc_xkv_compressor`, `delta_kv`, `hybridkv_compressor`, `cachenotes`, `moe_deltakv_paged` | 2x – 6x | 1.5x – 3.2x |
| **Tensor & Matrix Decomposition** | 12 | `kronecker_tt_lora`, `tt_compressor`, `rpca_compressor`, `cur_moe_compressor`, `nmf_compressor` | 4x – 16x | 1.2x – 1.6x |

---

## 2. Using Compression Engines

### Standard Factory Initialization

```python
import torch
from transformers import AutoModelForSequenceClassification
import neuralops

model = AutoModelForSequenceClassification.from_pretrained("roberta-base")

# 1. Instantiate compressor with target rank or compression ratio
compressor = neuralops.create_compressor("fwsvd", model=model, rank=32)

# 2. Execute calibrated compression
compressed_model = compressor.compress()

# 3. Model parameters and shapes are updated in-place
print(f"Compressed Parameters: {sum(p.numel() for p in compressed_model.parameters()) / 1e6:.1f}M")
```

---

## 3. Recommended Rank Regimes

Empirical evaluation on GLUE SST-2 and WikiText-2 establishes the optimal compressor dispatch regimes:

- **$r \le 64$ (Extreme Compression Regime)**: Use **`fwsvd`** (Fisher-Weighted SVD) or **`adasvd_compressor`** (Adaptive Energy SVD) to maintain accuracy $>80\%$.
- **$r \in [64, 128]$ (Moderate Compression Regime)**: Use **`sparsegpt_engine`** or **`asvd_compressor`** for high structural preservation.
- **$r \ge 128$ (High-Rank Fine-Tuning)**: Use **`hnc_svd`** (Higher-Order Nullspace Continuation SVD) or **`direct_svd`**.
