# Compression Engines API Reference

NeuralOps registers **exactly 72 canonical model compression engines** unified under `neuralops.core.registry.COMPRESSOR_REGISTRY` across **8 foundational algorithmic paradigms**.

All compression engines implement calibrated low-rank matrix decomposition, structured pruning, integer/floating-point quantization, tensor networks, or dynamic KV-cache compression to drastically reduce VRAM footprints and accelerate inference while preserving model accuracy.

[![Release](https://img.shields.io/badge/release-v2.0.6-blue.svg)](https://github.com/tahamajs/NeuralOps/releases/tag/v2.0.6)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-ffcc00)](https://huggingface.co/datasets/Hooshaai/neuralops-results)
[![Hardware Verified](https://img.shields.io/badge/Hardware%20Verified-RTX%204070%20(SM%208.9)-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/datasets/Hooshaai/neuralops-results)

---

## 1. Algorithmic Taxonomy (8 Foundational Paradigms)

The 72 compression engines span 8 foundational algorithmic paradigms, totaling exactly 72 registered engines:

| # | Paradigm Key | Paradigm Name | Count | Typical Ratio | Speedup | Representative Engine Identifiers |
| :-: | :--- | :--- | :-: | :---: | :---: | :--- |
| 1 | `svd` | **Low-Rank SVD Factorization** | 16 | 2x – 6x | 1.3x – 2.4x | `fwsvd_compressor`, `adasvd_compressor`, `direct_svd`, `residual_svd_compressor`, `swift_svd`, `dobi_svd`, `erc_svd` |
| 2 | `dynamic_rank` | **Adaptive & Dynamic Rank Allocation** | 11 | 2x – 5x | 1.2x – 2.0x | `adaptive_ranks`, `come_compressor`, `dynamic_rank`, `entropy_budget_allocation`, `grecodec_compressor`, `sdrm_compressor` |
| 3 | `pruning` | **Structured Pruning & Sparsity** | 10 | 2x – 4x | 1.2x – 1.8x | `admm_pruner`, `sparsegpt`, `wanda_engine`, `flap`, `mamba_shedder`, `proxsparse`, `armor_pruner`, `bonsai_pruner` |
| 4 | `subspace` | **Subspace & Residual Projection** | 9 | 3x – 8x | 1.2x – 1.6x | `optimal_subspace_projection`, `subspace_projection`, `drone_compressor`, `enlighten_compressor`, `liftr`, `oats_compressor` |
| 5 | `tensor_decomp` | **Tensor Network Decompositions** | 8 | 4x – 16x | 1.2x – 1.6x | `tt_compressor`, `kronecker_compressor`, `kronecker_tt_lora`, `cur_compressor`, `cur_moe_compressor`, `rpca_compressor` |
| 6 | `quant` | **Integer & Floating-Point Quantization** | 7 | 4x – 8x | 1.4x – 2.0x | `faq_quantizer`, `rotational_slicegpt_int4`, `uniql_compressor`, `astro_quantizer`, `coopq_quantizer`, `huffman_coder` |
| 7 | `kv_cache` | **KV-Cache Optimization & Compression** | 6 | 2x – 6x | 1.5x – 3.2x | `aatc_xkv_compressor`, `cachenotes_compressor`, `delta_kv`, `hybridkv_compressor`, `kvtc_compressor`, `zerokv_neo_compressor` |
| 8 | `rotational` | **Rotational Invariance Transforms** | 5 | 2x – 4x | 1.3x – 1.7x | `slicegpt`, `slicegpt_pca`, `flat_llm`, `hashed_net_compressor`, `roast_hashed_linear` |

> [!NOTE]
> Summing across all 8 paradigms: $16 + 11 + 10 + 9 + 8 + 7 + 6 + 5 = \mathbf{72}$ canonical compression engines.
> All empirical compression Pareto curves, WikiText-2 perplexity benchmarks, and GLUE retention matrices are available on Hugging Face:  
> 👉 **[https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results)**

---

## 2. Paradigm Breakdown & Mathematical Foundations

### 2.1 Low-Rank SVD Factorization (`svd` — 16 Engines)
- **Module IDs**: `a3_compressor`, `aa_svd_compressor`, `adasvd_compressor`, `direct_svd`, `dobi_svd`, `erc_svd`, `frandi_compressor`, `fwsvd_compressor`, `hybrid_svd_layer`, `residual_svd_compressor`, `saes_svd_v2`, `salr`, `svd_linear`, `svd_llm_compressor`, `swift_svd`, `swift_svd_act`
- **Mathematical Principle**:
  Given weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and activation covariance $C_X = \mathbb{E}[X^\top X]$, whitening transforms the optimization space:
  $$\tilde{W} = W C_X^{1/2} = U \Sigma V^\top$$
  Truncating to rank $r$ minimizes empirical reconstruction error $\|(W - \hat{W}) X\|_F^2$:
  $$\hat{W} = U_r \Sigma_r V_r^\top C_X^{-1/2} = A B, \quad A \in \mathbb{R}^{d_{\text{out}} \times r}, \; B \in \mathbb{R}^{r \times d_{\text{in}}}$$

### 2.2 Adaptive & Dynamic Rank Allocation (`dynamic_rank` — 11 Engines)
- **Module IDs**: `adaptive_ranks`, `come_compressor`, `dynamic_rank`, `entropy_budget_allocation`, `gfws`, `grecodec_compressor`, `mgie_budget`, `mlorq_compressor`, `modegpt_compressor`, `sdrm_compressor`, `soka_compressor`
- **Mathematical Principle**:
  Instead of uniform rank $r$ across all layers, dynamically allocates rank budgets based on layer-wise singular value entropy $\mathcal{H}_l = -\sum_i \bar{\sigma}_{l,i} \log \bar{\sigma}_{l,i}$ and Hessian eigenspectrum curvature, allocating higher rank to sensitive early layers.

### 2.3 Structured Pruning & Sparsity (`pruning` — 10 Engines)
- **Module IDs**: `admm_compressor`, `admm_pruner`, `armor_pruner`, `ast_pruner`, `bonsai_pruner`, `flap`, `mamba_shedder`, `proxsparse`, `slim_compressor`, `sparsegpt`
- **Mathematical Principle**:
  Solves constrained $\ell_0$ or $\ell_{2,1}$ group-sparse minimization via Alternating Direction Method of Multipliers (ADMM) or second-order Taylor approximations:
  $$\min_W \mathcal{L}(W) + \frac{\rho}{2} \|W - Z + U\|_F^2 \quad \text{s.t.} \quad \|Z_{\text{cols}}\|_0 \le k$$

### 2.4 Subspace & Residual Projection (`subspace` — 9 Engines)
- **Module IDs**: `cr_compressor`, `drone_compressor`, `enlighten_compressor`, `liftr`, `lsc_compressor`, `oats_compressor`, `optimal_subspace_projection`, `subspace_projection`, `tomoe_compressor`
- **Mathematical Principle**:
  Projects layer activations into orthogonal principal subspaces, decomposing operations into a shared invariant basis plus small residual corrections.

### 2.5 Tensor Network Decompositions (`tensor_decomp` — 8 Engines)
- **Module IDs**: `compot_compressor`, `cur_compressor`, `cur_moe_compressor`, `kronecker_compressor`, `kronecker_tt_lora`, `nmf_compressor`, `rpca_compressor`, `tt_compressor`
- **Mathematical Principle**:
  Decomposes high-order weight tensors into Tensor-Train (TT) cores ($W = G_1 G_2 \cdots G_d$) or Kronecker matrix products ($W = A \otimes B$), unlocking exponential theoretical parameter reduction.

### 2.6 Integer & Floating-Point Quantization (`quant` — 7 Engines)
- **Module IDs**: `astro_quantizer`, `coopq_quantizer`, `faq_quantizer`, `huffman_coder`, `kmeans_quantize`, `rotational_slicegpt_int4`, `uniql_compressor`
- **Mathematical Principle**:
  Maps continuous FP16/FP32 weights to discrete low-bit grids (INT8, INT4, FP4 MX format) using optimal asymmetric scales and zero-points:
  $$W_q = \text{clamp}\left(\left\lfloor \frac{W}{S} \right\rceil + Z, -2^{b-1}, 2^{b-1}-1\right)$$

### 2.7 KV-Cache Optimization & Compression (`kv_cache` — 6 Engines)
- **Module IDs**: `aatc_xkv_compressor`, `cachenotes_compressor`, `delta_kv`, `hybridkv_compressor`, `kvtc_compressor`, `zerokv_neo_compressor`
- **Mathematical Principle**:
  Reduces KV-cache memory footprints during generation via delta compression ($K_t - K_{t-1}$), attention sink retention, and importance-driven token shedding.

### 2.8 Rotational Invariance Transforms (`rotational` — 5 Engines)
- **Module IDs**: `flat_llm`, `hashed_net_compressor`, `roast_hashed_linear`, `slicegpt`, `slicegpt_pca`
- **Mathematical Principle**:
  Applies orthogonal rotation matrices $Q \in \mathbb{O}(D)$ (such as randomized Hadamard transforms) to eliminate activation outliers across channels, enabling aggressive 25%–40% column slicing without perplexity collapse.

---

## 3. Core Functional API Signatures

### `neuralops.create_compressor`

```python
def create_compressor(
    name: str,
    model: Any = None,
    rank: int | None = None,
    **kwargs: Any,
) -> Any
```

Instantiates any of the 72 registered compression engines from `COMPRESSOR_REGISTRY`.

**Parameters:**
- `name` (*str*): Canonical compressor name (e.g., `"fwsvd_compressor"`, `"adasvd_compressor"`, `"sparsegpt"`, `"slicegpt"`).
- `model` (*nn.Module*, optional): Model instance to compress.
- `rank` (*int*, optional): Target low-rank bottleneck or compression rank.
- `**kwargs`: Hyperparameters forwarded to the compressor constructor.

**Example:**
```python
import neuralops

# Create Fisher-Weighted SVD compressor targeting rank 32
compressor = neuralops.create_compressor(
    "fwsvd_compressor",
    model=model,
    rank=32,
)
compressed_model = compressor.compress()
```

---

### `neuralops.compress`

```python
def compress(
    model: nn.Module,
    method: str = "svd",
    rank: int = 32,
    **kwargs: Any,
) -> nn.Module
```

Convenience function to execute full-model compression in a single function call.

---

## 4. Grounded Hardware Verification & Telemetry

All 72 compression engines are verified on physical silicon:
- **Primary Hardware**: **NVIDIA GeForce RTX 3090 (Ampere SM 8.6, GA102 silicon)**.
- **Hardware Specs**: 5,888 CUDA cores, 46 SMs, 184 Tensor Cores (4th gen), 160.2 TFLOPS boost FP16, 504.0 GB/s GDDR6X, 12 GB VRAM.
- **Official Ledger Repository**: [https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results).
