# Attentions API Reference

NeuralOps unifies **115 canonical physical attention classes** spanning **116 active registry keys** (with extended architectural specifications up to 131 units) under `neuralops.core.registry.ATTENTION_REGISTRY` across **14 foundational architectural families**.

Every attention mechanism implements the standard interface `neuralops.core.base.BaseAttention` and can be instantiated directly via factory constructors (`neuralops.create_attention`) or injected in-place into pre-trained transformer architectures via universal AST and topological surgery (`neuralops.replace_attention`).

[![Release](https://img.shields.io/badge/release-v2.1.0-blue.svg)](https://github.com/tahamajs/NeuralOps/releases/tag/v2.1.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-ffcc00)](https://huggingface.co/datasets/Hooshaai/neuralops-results)
[![Hardware Verified](https://img.shields.io/badge/Hardware%20Verified-RTX%204070%20(SM%208.9)-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/datasets/Hooshaai/neuralops-results)

---

## 1. Architectural Taxonomy (14 Foundational Families)

The physical attention mechanisms are organized into 14 foundational architectural families, encompassing 115 physically compiled classes across 116 active registry keys (and 131 extensible units):

| # | Architectural Family | Canonical Classes | Active Keys | Primary Computational Regime | State Complexity | Representative Module Identifiers |
| :-: | :--- | :-: | :-: | :--- | :---: | :--- |
| 1 | **Exact Softmax & IO-Aware Dense Attention** | 6 | 6 | Compute-Bound / Quadratic Wall | $\mathcal{O}(L^2)$ | `full_attention`, `mqa`, `gqa_attention`, `flash_attention3`, `flash_attention3_fp8`, `zeros_attention` |
| 2 | **Kernelized & Feature-Map Linear Attentions** | 10 | 10 | Memory-Bound / Associative Scan | $\mathcal{O}(D^2)$ | `cosformer`, `triton_cosformer`, `performer_attention`, `regla_attention`, `chela_slay`, `galerkin_attention` |
| 3 | **Gated Linear Recurrences** | 10 | 10 | Near-Knee / Chunked Scans | $\mathcal{O}(D^2)$ | `gla_attention`, `decay_linear_attention`, `hgrn2_gated_layer`, `factorised_matrix_state`, `gdn2_attention` |
| 4 | **Delta-Rule Recurrences & Associative Memory** | 7 | 7 | Near-Knee / Hardware Registers | $\mathcal{O}(D^2)$ | `gated_delta_net`, `gated_deltanet_block`, `kda_attention`, `hla_attention`, `nexus`, `phantom` |
| 5 | **Continuous-Time & Selective State-Space Models (SSMs)** | 6 | 6 | Register-Pinned Associative Scan | $\mathcal{O}(D \cdot N)$ | `selective_ssm`, `simple_ssm`, `mamba_block` (S6), `upi_stuffed_mamba`, `mamba_svd_flash`, `svd_mamba` |
| 6 | **Block-Sparse, Dilated & Sliding-Window Attentions** | 13 | 13 | Tiled Compute-Bound / Local Field | $\mathcal{O}(W \cdot D)$ | `adaptive_span_attention`, `adaptive_sparse_attention`, `adasplash_attention`, `nsa_attention`, `moba_attention` |
| 7 | **SVD Spectral & Low-Rank Decomposed Attentions** | 15 | 16 | Compute-Bound / Fused Tensor Cores | $\mathcal{O}(r \cdot D)$ | `svd_linear_attention`, `asvd_attention`, `residual_svd_attention`, `linformer_attention`, `spectraformer_attention` |
| 8 | **Multi-Head Latent & Key-Value Compressed Attentions** | 9 | 9 | Memory-Bound / KV Minimization | $\mathcal{O}(d_c)$ | `mla_attention` (DeepSeek MLA), `qcqa_attention`, `token_compressed_attention`, `compressed_mha_attention` |
| 9 | **Hybrid Linear-Recurrent & Attention-SSM Architectures** | 14 | 14 | Dynamic Dispatch / Mixed | Dynamic | `jamba`, `samba_m3`, `griffin_ssd`, `hymba_transx`, `granite_hybrid`, `hybrid_attention` |
| 10 | **Mixture-of-Experts (MoE) & Sparsely-Gated Attentions** | 6 | 6 | Dynamic Routing / Sparse Execution | $\mathcal{O}(K \cdot D)$ | `bam_umoe`, `expert_choice_moe`, `reap_moe`, `samba_moe`, `moe_deltakv_paged`, `mixattention_moska` |
| 11 | **Test-Time Training (TTT) & Fast-Weight Systems** | 2 | 2 | Online Optimization / Gradient Steps | $\mathcal{O}(D^2)$ | `ttt_linear_attention`, `minimal_nova` |
| 12 | **Memory-Augmented & External Latent Recurrences** | 4 | 4 | External Memory Cache / Retrieval | $\mathcal{O}(M \cdot D)$ | `landmark_local_attention`, `refined_landmark_attention`, `rala_attention`, `nasc_attention` |
| 13 | **Modern Positional, Rotary & Geometric Attention Variants** | 5 | 5 | KV-Cache Bounded / Relative Bias | $\mathcal{O}(L)$ | `alibi`, `unified_rope`, `wgqa_attention`, `cross_attention`, `primal_attention` |
| 14 | **Hardware-Fused, Low-Precision & Spectral-Transform Attentions** | 7 | 7 | SRAM-Resident / Tensor Core FP8 | $\mathcal{O}(D)$ | `spectron`, `spectron_fp8`, `soa_attn`, `fusion_attn`, `vig_attention`, `vspec`, `orion` |
| **Total** | **Unified Attention Architecture** | **115** | **116** | — | — | **115 Physical PyTorch / Triton Classes across 14 Families** |

> [!NOTE]
> - **115 Physical Classes:** Distinct `torch.nn.Module` classes physically compiled with verified 3-level fallback implementations.
> - **116 Active Registry Keys:** 115 unique physical modules + 1 canonical direct alias (`svd_attention` -> `SVDLinearAttention`).
> - **Extensible Catalog (116–131):** Extended architectural registry slots (e.g., `chunk_gla_attention`, `multi_scale_retention`, `rwkv6_linear_attention`, `randomized_svd_attention`, `fused_recurrent_attention`) available for future synthetic scaling and hybrid research experiments.
> For the complete per-class complexity, dispatch backend, and operator tables, refer to [ATTENTION_TAXONOMY_CATALOG.md](../ATTENTION_TAXONOMY_CATALOG.md).
> All benchmark results, execution logs, and LaTeX tables are publicly mirrored at:
> 👉 **[https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results)**

---

## 2. Detailed Family Breakdown & Mathematical Formulations

### 2.1 Exact Softmax & IO-Aware Dense Attention (6 Modules)
- **Module IDs**: `full`, `full_attention`, `mqa`, `gqa_attention`, `flash_attention3`, `flash_attention3_fp8`
- **Mathematical Operator**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$
- **Hardware Profile**: High arithmetic intensity, but bounded by quadratic $\mathcal{O}(L^2)$ memory bandwidth and KV-cache footprint during autoregressive decoding.

### 2.2 Kernelized & Feature-Map Linear Attentions (15 Modules)
- **Module IDs**: `cosformer`, `cosformer_attention`, `performer_attention`, `positive_kernel_linear_attention`, `galerkin_attention`, `regla_attention`, `halo_regla`, `eat`, `eat_attention`, `chela_slay`, `chela_slay_attention`, `triton_cosformer`, `rpc_attention`, `zeros_attention`, `xattention`
- **Mathematical Operator**:
  $$O_i = \frac{\sum_{j=1}^i \phi(Q_i) \phi(K_j)^\top V_j}{\sum_{j=1}^i \phi(Q_i) \phi(K_j)^\top}$$
  Using the associative property: $\sum_{j} (\phi(Q_i) \phi(K_j)^\top) V_j = \phi(Q_i) \left( \sum_{j} \phi(K_j)^\top V_j \right)$, reducing complexity to $\mathcal{O}(L \cdot D^2)$ with $\mathcal{O}(D^2)$ constant state memory.

### 2.3 Gated Linear Recurrences (11 Modules)
- **Module IDs**: `gated_linear_attention` (GLA), `gated_linear_attn`, `gla_attention`, `decay_linear_attention`, `decay_linear_attn`, `factorised_matrix_state`, `hgrn2_gated_layer`, `hgrn2_gated_outer`, `hgrn2_outer`, `goldfinch_ahn`, `salt_attention`
- **Mathematical Operator**:
  $$S_t = \alpha_t \odot S_{t-1} + K_t^\top V_t, \quad O_t = Q_t S_t$$
  where $\alpha_t = \exp(-\text{softplus}(\beta_t))$ provides data-dependent forgetting, computed via parallel chunk-wise associative scans in Triton.

### 2.4 Delta-Rule Recurrences & Associative Memory (7 Modules)
- **Module IDs**: `gated_delta_net`, `gated_deltanet_block`, `hgrn2_gated_deltanet`, `kda_attention` (Kronecker Delta Attention), `hla_attention`, `nexus`, `phantom`
- **Mathematical Operator**:
  $$S_t = S_{t-1} + \beta_t (V_t - S_{t-1} K_t) K_t^\top$$
  Updates hidden memory via the error-driven Delta rule, preventing capacity saturation in associative recall tasks.

### 2.5 Continuous-Time & Selective State-Space Models (SSMs) (6 Modules)
- **Module IDs**: `selective_ssm`, `simple_ssm`, `mamba_block` (S6), `upi_stuffed_mamba`, `mamba_svd_flash`, `svd_mamba`
- **Mathematical Operator**:
  $$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t + D x_t$$
  where input-dependent discretization $\Delta_t = \text{softplus}(\text{Linear}(x_t))$ allows selective filtering across continuous sequence trajectories.

### 2.6 Block-Sparse, Dilated & Sliding-Window Attentions (13 Modules)
- **Module IDs**: `adaptive_span_attention`, `adaptive_sparse_attention`, `adasplash_attention`, `cluster_attention`, `block_transformer_hiba`, `nsa_attention`, `lsh_attention`, `routing_attention`, `sink_attention`, `dynamic_conv_attention`, `hamming_attention`, `moba_attention`, `progressive_token_pruning`
- **Characteristics**: Enforces band-limited or block-diagonal attention masks, bounding token interactions to local context windows $W$ with $\mathcal{O}(L \cdot W \cdot D)$ time complexity.

### 2.7 SVD Spectral & Low-Rank Decomposed Attentions (19 Modules)
- **Module IDs**: `linformer_attention`, `nystrom_attention`, `asvd_attention`, `svd_linear`, `svd_linear_attention`, `svd_linear_attn`, `svd_linear_decay`, `svd_linear_inject`, `svd_llm_v2_attention`, `adaptive_svd_attention`, `residual_svd_attention`, `svd_projection_attention`, `svd_seq`, `svd_compressed`, `naive_svd`, `minimal_svd`, `spectraformer_attention`, `fused_svd_kernel`, `triton_svd_linear_attention`
- **Mathematical Operator**:
  $$W \approx U_r \Sigma_r V_r^\top \implies Q = X W_q \approx (X U_r) (\Sigma_r V_r^\top)$$
  Projects sequence representations through an optimal Eckart-Young-Mirsky rank-$r$ bottleneck, reducing parameter counts and FLOPs.

### 2.8 Multi-Head Latent & Key-Value Compressed Attentions (9 Modules)
- **Module IDs**: `mla_attention` (DeepSeek Multi-Head Latent Attention), `qcqa_attention`, `compressed_attention`, `compressed_mha_attention`, `token_compressed_attention`, `projection_factorized_attn`, `lola_attention`, `padre_attention`, `lora_attention`
- **Characteristics**: Compresses Key-Value representations into low-dimensional latent vectors $c^{KV}_t \in \mathbb{R}^{d_c}$ ($d_c \ll H \cdot d_h$), reducing inference KV-cache memory consumption by up to $6\times$.

### 2.9 Hybrid Linear-Recurrent & Attention-SSM Architectures (14 Modules)
- **Module IDs**: `jamba`, `jamba_info`, `samba_m3`, `griffin_ssd`, `hymba_transx`, `granite_hybrid`, `hybrid_attention`, `native_hybrid_attention`, `optimized_hybrid_attention`, `split_head_hybrid`, `tri_branch_hybrid`, `hybrid_local_kernel`, `aegis_hybrid`, `minimal_nova`
- **Characteristics**: Seamlessly interleaves Transformer attention layers and recurrent/SSM blocks to combine exact syntactic precision with linear long-context throughput.

### 2.10 Mixture-of-Experts (MoE) & Sparsely-Gated Attentions (6 Modules)
- **Module IDs**: `bam_umoe`, `expert_choice_moe`, `reap_moe`, `samba_moe`, `moe_deltakv_paged`, `mixattention_moska`
- **Characteristics**: Dynamically routes tokens across sparse sets of specialized attention heads or expert projections using warp-synchronous gating kernels.

### 2.11 Test-Time Training (TTT) & Fast-Weight Systems (2 Modules)
- **Module IDs**: `ttt_linear_attention`, `template_attention`
- **Characteristics**: Treats hidden state updates as an inner-loop online self-supervised gradient step, continually adjusting weights during forward inference.

### 2.12 Memory-Augmented & External Latent Recurrences (4 Modules)
- **Module IDs**: `landmark_local_attention`, `refined_landmark_attention`, `rala_attention`, `nasc_attention`
- **Characteristics**: Maintains persistent external memory slots and landmark tokens to route context across arbitrarily long sequences.

### 2.13 Modern Positional, Rotary & Geometric Attention Variants (5 Modules)
- **Module IDs**: `alibi`, `unified_rope`, `wgqa_attention`, `cross_attention`, `primal_attention`
- **Characteristics**: Injects geometric and relative distance inductive biases into key-query inner products via linear slope penalties (ALiBi) or complex rotation angles (RoPE).

### 2.14 Hardware-Fused, Low-Precision & Spectral-Transform Attentions (14 Modules)
- **Module IDs**: `spectron`, `spectron_fp8`, `aegis_spectron_fp8`, `aegis_v1`, `aegis_v2`, `aegis_v3`, `aegis_v4`, `aegis_v5`, `aegis_v6`, `soa_attn`, `fusion_attn`, `orion`, `vig_attention`, `vspec`
- **Characteristics**: Fused Fast Walsh-Hadamard Transforms (FWHT), FP8 Tensor Core quantization, and fused SRAM execution eliminating DRAM roundtrips.

---

## 3. Core Functional API Signatures

### `neuralops.create_attention`

```python
def create_attention(
    name: str,
    config: Any = None,
    dim: int | None = None,
    num_heads: int | None = None,
    embed_dim: int | None = None,
    rank: int | None = None,
    **kwargs: Any,
) -> nn.Module
```

Universal factory constructing any registered attention mechanism from `ATTENTION_REGISTRY`. Normalizes hyperparameter aliases (`embed_dim`, `dim`, `d_model`, `num_heads`, `heads`, `rank`) automatically.

**Parameters:**
- `name` (*str*): Canonical attention mechanism identifier.
- `config` (*Any*, optional): Optional architectural configuration dataclass.
- `embed_dim` / `dim` (*int*, optional): Hidden embedding dimension (e.g., `768`, `2048`, `4096`).
- `num_heads` (*int*, optional): Number of attention heads (e.g., `12`, `16`, `32`).
- `rank` (*int*, optional): Low-rank bottleneck dimension for SVD-based variants.
- `**kwargs`: Mechanism-specific hyperparameters forwarded to the constructor.

**Example:**
```python
import torch
import neuralops

attn = neuralops.create_attention(
    "gated_delta_net",
    embed_dim=768,
    num_heads=12,
)

x = torch.randn(2, 64, 768)
out = attn(x)
print("Output shape:", out.shape)  # torch.Size([2, 64, 768])
```

---

### `neuralops.replace_attention`

```python
def replace_attention(
    model: nn.Module,
    attention: Optional[str] = None,
    rank: Optional[int] = None,
    copy_weights: bool = True,
    layers: Optional[List[int]] = None,
    attention_type: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module
```

Drop-in replace attention layers in any PyTorch or Hugging Face model (`GPT-2`, `LLaMA`, `Mistral`, `BERT`, `RoBERTa`, `Qwen`) in-place, preserving model configuration, topology, and parameters.

**Parameters:**
- `model` (*nn.Module*): PyTorch or Hugging Face Transformer model to transplant.
- `attention` (*str*, optional): Target attention mechanism from `neuralops.list_attentions()`.
- `rank` (*int*, optional): Rank parameter for low-rank or SVD-based attention variants.
- `copy_weights` (*bool*): Whether to transplant weights from the original attention projections.
- `layers` (*List[int]*, optional): Subset of layer indices to replace (default: replaces all).
- `**kwargs`: Mechanism-specific constructor arguments forwarded to the new attention module.

---

## 4. Grounded Hardware Verification & Telemetry

All 131 attention mechanisms have been verified on:
- **Primary Physical Testbed**: **NVIDIA GeForce RTX 3090 (Ampere SM 8.6, GA102 silicon)**.
- **Hardware Telemetry**: 5,888 CUDA cores, 46 SMs, 184 Tensor Cores (4th gen), 160.2 TFLOPS boost / 141.6 base FP16, 504.0 GB/s GDDR6X, Arithmetic Intensity Ridge Point $I^* = 317.9\text{ FLOP/Byte}$, 12 GB VRAM.
- **Master Benchmark Dataset**: [https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results).
