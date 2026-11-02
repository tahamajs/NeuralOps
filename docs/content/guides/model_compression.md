# Model Compression Guide

NeuralOps unifies **72 canonical model compression engines** across **8 foundational algorithmic paradigms** under `neuralops.core.registry.COMPRESSOR_REGISTRY`.

The platform provides end-to-end support for low-rank matrix decomposition, structured channel/head pruning, integer/floating-point quantization, rotational invariant slicing, dynamic rank allocation, and KV-cache compression.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-ffcc00)](https://huggingface.co/datasets/Hooshaai/neuralops-results)

---

## 1. The 8 Compression Paradigms

| # | Paradigm | Count | Primary Implementations | Typical Ratio | Speedup | Algorithmic Mechanism |
| :-: | :--- | :-: | :--- | :---: | :---: | :--- |
| 1 | **Low-Rank SVD Factorization (`svd`)** | 16 | `fwsvd_compressor`, `adasvd_compressor`, `direct_svd`, `residual_svd_compressor`, `swift_svd`, `dobi_svd`, `erc_svd`, `saes_svd_v2` | 2x – 6x | 1.3x – 2.4x | SVD truncation with activation whitening and covariance pullback ($W \approx U \Sigma V^T$) |
| 2 | **Adaptive & Dynamic Rank Allocation (`dynamic_rank`)** | 11 | `adaptive_ranks`, `dynamic_rank`, `come_compressor`, `grecodec_compressor`, `sdrm_compressor`, `entropy_budget_allocation` | 2x – 5x | 1.2x – 2.0x | Per-layer singular value spectral energy budgeting and entropy allocation |
| 3 | **Structured Pruning & Sparsity (`pruning`)** | 10 | `admm_pruner`, `admm_compressor`, `sparsegpt`, `wanda_engine`, `flap`, `mamba_shedder`, `proxsparse`, `armor_pruner` | 2x – 4x | 1.2x – 1.8x | ADMM optimization, second-order Taylor importance, and token-shedding |
| 4 | **Subspace & Residual Projection (`subspace`)** | 9 | `optimal_subspace_projection`, `subspace_projection`, `drone_compressor`, `enlighten_compressor`, `liftr`, `oats_compressor` | 3x – 8x | 1.2x – 1.6x | Orthogonal residual subspace projection and principal component bases |
| 5 | **Tensor Network Decompositions (`tensor_decomp`)** | 8 | `tt_compressor`, `kronecker_compressor`, `kronecker_tt_lora`, `cur_compressor`, `cur_moe_compressor`, `rpca_compressor` | 4x – 16x | 1.2x – 1.6x | Tensor-Train (TT), Kronecker factorizations, CUR matrix decomposition |
| 6 | **Integer & Floating-Point Quantization (`quant`)** | 7 | `faq_quantizer`, `rotational_slicegpt_int4`, `uniql_compressor`, `astro_quantizer`, `coopq_quantizer`, `huffman_coder` | 4x – 8x | 1.4x – 2.0x | Calibration-based INT8/INT4/FP4 quantization with scale factor optimization |
| 7 | **KV-Cache Optimization (`kv_cache`)** | 6 | `aatc_xkv_compressor`, `delta_kv`, `hybridkv_compressor`, `cachenotes_compressor`, `kvtc_compressor`, `zerokv_neo_compressor` | 2x – 6x | 1.5x – 3.2x | Dynamic token eviction, adaptive attention budgets, delta KV caching |
| 8 | **Rotational Invariance Transforms (`rotational`)** | 5 | `slicegpt`, `slicegpt_pca`, `flat_llm`, `hashed_net_compressor`, `roast_hashed_linear` | 2x – 4x | 1.3x – 1.7x | Haar-random orthogonal rotation matrices spreading outliers across dimensions |

---

## 2. Using `neuralops.compress`

NeuralOps provides a high-level API to compress transformer models in a single line of code:

```python
import torch
from transformers import AutoModelForCausalLM
import neuralops

model = AutoModelForCausalLM.from_pretrained("gpt2")

# 1. Fisher-Weighted SVD (FWSVD)
compressed_model = neuralops.compress(
    model=model,
    method="fwsvd",
    rank=32,
)

# 2. SliceGPT with 25% sparsity
slicegpt_model = neuralops.compress(
    model=model,
    method="slicegpt",
    sparsity=0.25,
)
```

---

## 3. Instantiating via `neuralops.create_compressor`

For fine-grained control over calibration datasets, target modules, and tolerance thresholds:

```python
from neuralops.core.registry import create_compressor

compressor = create_compressor(
    "fwsvd_compressor",
    model=model,
    rank=48,
    target_modules=["c_attn", "c_proj"],
    dtype=torch.float16,
)

# Run calibration and compression
compressed_model = compressor.compress()
```

---

## 4. Grounded Pareto Frontiers

All compression engines are evaluated on physical hardware:
- **Testbed**: Dedicated **NVIDIA GeForce RTX 4070 (Ada Lovelace SM 8.9, AD104 silicon)**.
- **Metrics Tracked**: Parameter reduction (%), WikiText-2 Perplexity ($\Delta \text{PPL}$), Zero-shot GLUE retention (%), Peak VRAM allocation (MB), Latency speedup ($x$).
- **Master Results Ledger**: Mirrored and synchronized at  
  👉 **[https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results)**
