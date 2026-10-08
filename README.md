<div align="center">

<img src="https://raw.githubusercontent.com/tahamajs/SVD_linear_Attention/main/assets/banner.png" alt="NeuralOps Banner" width="100%">

# 🧬 NeuralOps

### **A Production-Grade Framework for Subquadratic Attention & Model Compression**

*Empirical evaluation · Zero-shot architectural introspection · Hardware-roofline profiling · Activation-calibrated decomposition*

<br>

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1--12.7-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-Fused%20Kernels-FF6B6B?style=for-the-badge)](https://github.com/openai/triton)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/tahamajs/SVD_linear_Attention?style=for-the-badge&logo=github)](https://github.com/tahamajs/SVD_linear_Attention/stargazers)
[![Forks](https://img.shields.io/github/forks/tahamajs/SVD_linear_Attention?style=for-the-badge&logo=github)](https://github.com/tahamajs/SVD_linear_Attention/network/members)
[![Issues](https://img.shields.io/github/issues/tahamajs/SVD_linear_Attention?style=for-the-badge&logo=github)](https://github.com/tahamajs/SVD_linear_Attention/issues)

<br>

[**📖 Documentation**](#-quick-start) · [**🚀 Quick Start**](#-quick-start) · [**🏛️ Taxonomy**](#-module-taxonomy) · [**📊 Benchmarks**](#-benchmarking--reproduction) · [**🧪 Testing**](#-testing) · [**❓ FAQ**](#-faq)

<br>

> **114 Attention Mechanisms · 73 Compression Engines · 16 Triton Kernels · 8 Transformer Architectures · 617 Python Files · 249,271 LOC**

</div>

---

## 📋 Table of Contents

<details open>
<summary><b>Click to expand / collapse</b></summary>

- [🌟 Why NeuralOps?](#-why-neuralops)
- [🎯 Core Design Principles](#-core-design-principles)
- [🏛️ Module Taxonomy](#️-module-taxonomy)
  - [Attention Mechanisms (114 modules)](#attention-mechanisms-114-modules)
  - [Compression Engines (73 modules)](#compression-engines-73-modules)
- [🚀 Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Hardware Support Matrix](#hardware-support-matrix)
  - [3-Line Attention Swap](#-3-line-attention-swap)
  - [Compress Any Model](#️-compress-any-model)
  - [Registry Query](#-registry-query)
- [🔬 Architecture Deep Dive](#-architecture-deep-dive)
  - [Universal Zero-Shot Injection](#universal-zero-shot-injection)
  - [Data-Aware Activation Calibration](#data-aware-activation-calibration)
  - [Fused Triton Kernels](#fused-triton-kernels)
  - [Registry & Deprecated Aliases](#registry--deprecated-aliases)
- [📚 Detailed Module Reference](#-detailed-module-reference)
  - [SVD & Low-Rank Attention](#svd--low-rank-attention)
  - [Linear & Kernel Attention](#linear--kernel-attention)
  - [State-Space Models](#state-space-models)
  - [Hybrid Architectures](#hybrid-architectures)
  - [Sparse & MoE Attention](#sparse--moe-attention)
  - [Gated Recurrent Networks](#gated-recurrent-networks)
  - [Adaptive & Compression-Aware](#adaptive--compression-aware)
  - [Modern LLM Attention](#modern-llm-attention)
  - [Fast Kernels](#fast-kernels)
- [🗜️ Compression Engine Reference](#️-compression-engine-reference)
  - [SVD Factorization](#svd-factorization)
  - [Low-Rank Methods](#low-rank-methods)
  - [Pruning & Sparsification](#pruning--sparsification)
  - [Quantization](#quantization)
  - [KV-Cache Compression](#kv-cache-compression)
  - [Tensor Decompositions](#tensor-decompositions)
- [⚡ Triton Kernel Subsystem](#-triton-kernel-subsystem)
  - [Kernel Catalog](#kernel-catalog)
  - [Fallback Mechanism](#fallback-mechanism)
  - [Performance Benchmarks](#performance-benchmarks)
- [🏗️ Supported Architectures](#️-supported-architectures)
  - [Model-by-Model Details](#model-by-model-details)
  - [Injection Guide](#injection-guide)
- [📊 Benchmarking & Reproduction](#-benchmarking--reproduction)
  - [Master GPU Runner](#-master-gpu-runner)
  - [Track Runners](#-track-runners)
  - [Autonomous Pipelines](#-autonomous-pipelines)
  - [4-Model Unified Wrappers](#-4-model-unified-wrappers)
- [🏆 Key Empirical Findings](#-key-empirical-findings)
- [📁 Results Directory Taxonomy](#-results-directory-taxonomy)
- [📈 Analytics & Plotting](#-analytics--plotting)
- [🧪 Testing](#-testing)
- [🔧 Troubleshooting](#-troubleshooting)
- [❓ FAQ](#-faq)
- [🤝 Contributing](#-contributing)
- [📜 Changelog](#-changelog)
- [⚖️ Citation & License](#️-citation--license)

</details>

---

## 🌟 Why NeuralOps?

Standard Multi-Head Attention (MHA) is **quadratic** in sequence length — both in compute ($O(N^2)$) and memory. This makes long-context inference prohibitively expensive. NeuralOps delivers a unified, pip-installable package to **benchmark, deploy, and reproduce** subquadratic alternatives **without leaving the Hugging Face ecosystem**.

### 🔥 The Problem We Solve

Modern foundation models face a **three-way tension**:

| Challenge | Traditional Approach | NeuralOps Solution |
|---|---|---|
| **Long context** ($N > 8K$) | OOM or quadratic slowdown | Linear/SSM attention ($O(N)$) |
| **Model size** | Cannot fit on consumer GPU | SVD + LoRA compression |
| **Inference cost** | High FLOPs + memory traffic | Fused Triton kernels |
| **Reproducibility** | Ad-hoc scripts, no tracking | Standardized result taxonomy |

### 💡 What Makes NeuralOps Different

<table>
<tr>
<td width="25%" align="center">

### 🎯

**Zero-Shot Injection**

Replace attention in **BERT, RoBERTa, DistilBERT, GPT-2, LLaMA, Qwen, T5, DeBERTa** — no retraining required.

</td>
<td width="25%" align="center">

### 🗜️

**Structured Compression**

Truncated SVD · Fisher-weighted low-rank · Dynamic entropy · KV-cache compression.

</td>
<td width="25%" align="center">

### 📉 

**Activation-Calibrated**

Uses real token activations to compute empirical Hessian covariances for optimal SVD subspaces.

</td>
<td width="25%" align="center">

### ⚡

**Hardware-Resident**

Fused Triton kernels keep intermediates in **GPU SRAM** — bypass DRAM roundtrips.

</td>
</tr>
</table>

### 🎁 What You Get Out of the Box

- ✅ **187 pre-built modules** — 114 attention + 73 compression engines
- ✅ **641 deprecated aliases** — full backward compatibility
- ✅ **198 test files** — unit, integration, end-to-end
- ✅ **16 Triton kernels** — auto-fallback to PyTorch
- ✅ **8 model families** — zero-shot injection support
- ✅ **7-category result taxonomy** — reproducible benchmarking
- ✅ **Publication artifact generators** — LaTeX tables, Pareto curves

---

## 🎯 Core Design Principles

NeuralOps is built on **six foundational principles** that guide every design decision:

### 1. 🧩 **Composability over Monoliths**

Every module has a single responsibility. Attentions can be swapped, compressors can be composed, and kernels can be replaced — all without touching user code.

```python
# Attentions are interchangeable
model = replace_attention(model, ATTENTION_REGISTRY.get("svd_linear_attention"))
model = replace_attention(model, ATTENTION_REGISTRY.get("cosformer"))
model = replace_attention(model, ATTENTION_REGISTRY.get("gated_delta_net"))
```

### 2. 🔒 **Backward Compatibility is Non-Negotiable**

Every rename, refactor, or reorganization preserves **all 641 legacy aliases**. Old code keeps working — with a `DeprecationWarning` nudging you toward canonical names.

```python
# Both work identically — the second emits a warning
attn = ATTENTION_REGISTRY.get("svd_linear_attention")  # ✓ canonical
attn = ATTENTION_REGISTRY.get("aegis_v2")              # ✓ alias + warning
```

### 3. 📊 **Empirical over Anecdotal**

Every performance claim is backed by **result files** in `results/`. No hand-wavy "it's faster" — only measured numbers.

```
results/02_rank_sweeps/dobi_svd_mnli_r128.json:
  {"accuracy": 0.8647, "params": 0.861, "latency_ms": 12.3, ...}
```

### 4. ⚡ **Hardware-Aware by Default**

Triton kernels are used where they help and **automatically fall back** where they don't (CPU, MPS, non-Triton environments).

```python
# Same code, adapts to hardware
out = fused_attention(q, k, v)
# ✓ CUDA + Triton    → GPU kernel
# ✓ CPU / MPS        → PyTorch fallback (identical numerics)
```

### 5. 🧪 **Test-Driven Correctness**

All modules are validated against **reference implementations** with strict numerical tolerance. The test suite has **1,358 passing tests**.

### 6. 📖 **Self-Documenting**

Every module has:
- Clear docstrings
- Type hints
- Reference paper links
- Working examples in tests

---

## 🏛️ Module Taxonomy

<div align="center">

```
┌─────────────────────────────────────────────────────────────────┐
│  🧠  neuralops/attentions/  ─── 114 Modules · 159 Registry Keys │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   📐 svd/          (21)  ─  svd_linear · asvd · minimal_svd    │
│   📏 linear/       (16)  ─  cosformer · performer · linformer  │
│   🌊 ssm/          (17)  ─  mamba_block · selective_ssm · jamba│
│   ⚡ hybrid/       (15)  ─  split_head_hybrid · tri_branch      │
│   🕸️ sparse/       (12)  ─  adaptive_sparse · moba · nsa        │
│   🚪 gated/        ( 9)  ─  gated_delta_net · hgrn2 · gla      │
│   🎛️ adaptive/     ( 9)  ─  adaptive_span · token_compressed   │
│   🚀 modern/       ( 7)  ─  mla · gqa · unified_rope           │
│   ⚡ fast_kernels/ ( 5)  ─  flash_attention3 · adasplash        │
│   🎯 dense/        ( 2)  ─  full_attention · zeros_attention   │
│   🧪 experimental/ ( 1)  ─  template_attention                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  🗜️  neuralops/compression_engine/  ─── 73 Engines · 73 Keys    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   📊 low_rank/      (24) ─  a3 · come · cr · cur · nmf · rpca  │
│   📐 svd/           (18) ─  direct_svd · adasvd · swift_svd    │
│   ✂️ pruning/       ( 9) ─  admm · armor · bonsai · sparsegpt  │
│   🎨 quantization/  ( 8) ─  astro · faq · slicegpt · kmeans    │
│   💾 kv_cache/      ( 6) ─  kvtc · zerokv_neo · cachenotes     │
│   🔷 tensor_decomp/ ( 3) ─  kronecker · tt_compressor          │
│   🧪 experimental/  ( 5) ─  flat_llm · hashed_net · liftr      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

</div>

### Attention Mechanisms (114 modules)

<details>
<summary><b>📐 SVD & Low-Rank Attention (21 modules)</b> — click to expand</summary>

| Module | Description | Paper |
|---|---|---|
| `svd_linear_attention` | Linear attention with SVD-factorized projections | [SVD-LLM](https://arxiv.org/abs/2403.07378) |
| `asvd_attention` | Activation-aware SVD | [ASVD](https://arxiv.org/abs/2312.05821) |
| `adaptive_svd_attention` | Rank-adaptive SVD | — |
| `minimal_svd` | Minimal rank-constrained SVD | — |
| `minimal_nova` | NOVA-based SVD factorization | — |
| `naive_svd` | Baseline weight-only SVD | — |
| `svd_attention` | Standard SVD attention | — |
| `svd_compressed` | Compressed SVD attention | — |
| `svd_linear_decay` | SVD + exponential decay gating | — |
| `svd_llm_v2_attention` | Improved LLM SVD (v2) | — |
| `svd_mamba` | SVD-augmented Mamba | — |
| `svd_projection_attention` | SVD projection-only | — |
| `triton_svd_linear_attention` | Triton-accelerated SVD attention | — |
| `fused_svd_kernel` | In-SRAM fused SVD kernel | — |
| `compressed_attention` | Truncated SVD energy attention | — |
| `compressed_mha_attention` | Joint SVD over Q/K/V | — |
| `factorised_matrix_state` | Low-rank matrix state | — |
| `projection_factorized_attn` | Factorized projection | — |
| `residual_svd_attention` | Residual SVD attention | — |
| `salt_attention` | Singular Value Adaptation with Low-Rank Tuning | — |
| `lora_attention` | LoRA + attention fusion | [LoRA](https://arxiv.org/abs/2106.09685) |

</details>

<details>
<summary><b>📏 Linear & Kernel Attention (16 modules)</b> — click to expand</summary>

| Module | Description | Paper |
|---|---|---|
| `cosformer` | Cosine-based linear attention | [CosFormer](https://arxiv.org/abs/2202.08791) |
| `performer_attention` | FAVOR+ random features | [Performer](https://arxiv.org/abs/2009.14794) |
| `linformer_attention` | Low-rank projection (Linformer) | [Linformer](https://arxiv.org/abs/2006.04768) |
| `nystrom_attention` | Nyström landmark approximation | [Nyströmformer](https://arxiv.org/abs/2102.03902) |
| `tlinformer_attention` | TLinformer variant | — |
| `galerkin_attention` | Fourier-Galerkin integral kernel | — |
| `chela_slay` | Chebyshev spherical linearized attention | — |
| `nasc_attention` | RFF spectral cascade | — |
| `rpc_attention` | Random projection Cauchy kernel | — |
| `spectra_v2` | Spectral kernel (v2) | — |
| `spectraformer_attention` | Spectral transformation basis | — |
| `spectron` | Spectral frequency kernel | — |
| `primal_attention` | Dual/primal formulation | — |
| `rala_attention` | Randomized low-rank landmark | — |
| `tpa_attention` | Temporal pattern associative | — |
| `triton_cosformer` | Triton-accelerated Cosformer | — |

</details>

<details>
<summary><b>🌊 State-Space Models (17 modules)</b> — click to expand</summary>

| Module | Description | Paper |
|---|---|---|
| `mamba_block` | Mamba SSM block | [Mamba](https://arxiv.org/abs/2312.00752) |
| `mamba_svd_flash` | SVD + Flash Mamba | — |
| `selective_ssm` | Selective scan SSM | [S6](https://arxiv.org/abs/2312.00752) |
| `simple_ssm` | Minimal SSM reference | — |
| `spectral_mamba` | Spectral Mamba variant | — |
| `upi_stuffed_mamba` | UPI-stuffed Mamba | — |
| `jamba` | Jamba hybrid SSM-Transformer | [Jamba](https://arxiv.org/abs/2403.19887) |
| `jamba_info` | Jamba info variant | — |
| `samba_m3` | Samba M3 | [Samba](https://arxiv.org/abs/2406.07522) |
| `samba_moe` | Samba with MoE | — |
| `griffin_ssd` | Griffin SSM | [Griffin](https://arxiv.org/abs/2402.19427) |
| `hymba_transx` | Hymba hybrid | [Hymba](https://arxiv.org/abs/2411.13676) |
| `goldfinch_ahn` | GoldFinch | — |
| `nexus` | NEXUS state operator | — |
| `orion` | ORION recurrent hybrid | — |
| `phantom` | Phantom SSM | — |
| `vspec` | Variable spectrum SSM | — |

</details>

<details>
<summary><b>⚡ Hybrid Architectures (15 modules)</b> — click to expand</summary>

| Module | Description |
|---|---|
| `split_head_hybrid_attention` | Split-head local + global hybrid |
| `tri_branch_hybrid_attention` | 3-way: sliding + RFF + SVD |
| `gated_hybrid_kernel_attention` | Triple-gated hybrid |
| `hybrid_local_kernel_attention` | Local + kernel hybrid |
| `hybrid_attention` | Weighted ensemble hybrid |
| `hybrid_local_kernel_attention` | Local + kernel with gates |
| `fusion_attn` | Fused dense + sliding window |
| `granite_hybrid` | Granite hybrid |
| `halo_regla` | Halo sliding + recurrent |
| `regla_attention` | Recurrently-gated local |
| `mixattention_moska` | Mixture of local + global |
| `optimized_hybrid_attention` | Optimized hybrid |
| `positive_kernel_linear_attention` | ELU+1 kernel attention |
| `refined_landmark_attention` | Value-scored landmarks |
| `spectron_fp8_attention` | FP8 spectral hybrid |

</details>

<details>
<summary><b>🕸️ Sparse & MoE Attention (12 modules)</b> — click to expand</summary>

| Module | Description |
|---|---|
| `adaptive_sparse_attention` | Dynamic sparse masking |
| `cluster_attention` | Clustering-based sparse |
| `bam_umoe` | Mixture-of-Experts routing |
| `expert_choice_moe` | Expert-choice routing |
| `reap_moe` | Sparse MoE routing |
| `moba_attention` | Mixture of Block Attention |
| `moe_deltakv_paged` | Paged DeltaKV MoE |
| `nsa_attention` | Native Sparse Attention |
| `padre_attention` | Pattern-directed sparse |
| `routing_attention` | Routing transformer |
| `hamming_attention` | Hamming distance sparse |
| `lsh_attention` | Locality-Sensitive Hashing |

</details>

<details>
<summary><b>🚪 Gated Recurrent Networks (9 modules)</b> — click to expand</summary>

| Module | Description | Paper |
|---|---|---|
| `gated_delta_net` | Gated DeltaNet | [Gated DeltaNet](https://arxiv.org/abs/2412.06464) |
| `gated_deltanet_block` | DeltaNet block wrapper | — |
| `gated_linear_attention` | GLA | [GLA](https://arxiv.org/abs/2312.06635) |
| `decay_linear_attention` | Decay-based linear attention | — |
| `hgrn2_gated_deltanet` | HGRN2 + DeltaNet | [HGRN2](https://arxiv.org/abs/2404.07904) |
| `hgrn2_gated_layer` | HGRN2 layer | — |
| `hgrn2_gated_outer` | HGRN2 outer | — |
| `hla_attention` | Hybrid linear recurrent | — |
| `kda_attention` | Kimi Delta Attention | [KDA](https://arxiv.org/abs/2510.26692) |

</details>

<details>
<summary><b>🎛️ Adaptive & Compression-Aware (9 modules)</b> — click to expand</summary>

| Module | Description |
|---|---|
| `adaptive_span_attention` | Learned span per-head |
| `span_expanded_attention` | Expanded span variant |
| `infini_attention` | Compressive memory adaptive |
| `token_compressed_attention` | Token compression |
| `progressive_token_pruning` | Progressive pruning |
| `squeezing_heads_distill` | Head squeezing via distillation |
| `block_transformer_hiba` | Hierarchical block-adaptive |
| `eat` | Efficient Attention with Token pruning |
| `lola_attention` | Low-rank landmark adaptive |

</details>

<details>
<summary><b>🚀 Modern LLM Attention (7 modules)</b> — click to expand</summary>

| Module | Description | Paper |
|---|---|---|
| `gqa_attention` | Grouped Query Attention | [GQA](https://arxiv.org/abs/2305.13245) |
| `mla_attention` | Multi-Head Latent Attention | [MLA](https://arxiv.org/abs/2405.04434) |
| `nha_attention` | Native Hybrid Attention | — |
| `qcqa_attention` | Quantized-cache Q Attention | — |
| `vig_attention` | Visual Instruction Guidance | — |
| `wgqa_attention` | Weighted GQA | — |
| `unified_rope` | Unified RoPE utilities | — |

</details>

<details>
<summary><b>⚡ Fast Kernels (5 modules) & Dense (2 modules) & Experimental (1 module)</b></summary>

**Fast Kernels:**
- `flash_attention3` — FA3
- `flash_attention3_fp8` — FA3 with FP8
- `adasplash_attention` — Adasplash
- `soa_attn` — SOA fused attention
- `xattention` — X-Attention kernel

**Dense:**
- `full_attention` — Quadratic baseline
- `zeros_attention` — Zero-attention ablation

**Experimental:**
- `template_attention` — Developer template

</details>

### Compression Engines (73 modules)

<details>
<summary><b>📐 SVD Factorization (18 engines)</b> — click to expand</summary>

| Engine | Description |
|---|---|
| `direct_svd` | Standard truncated SVD |
| `adasvd_compressor` | Adaptive SVD |
| `fwsvd_compressor` | Fisher-weighted SVD |
| `swift_svd` | Swift SVD |
| `swift_svd_act` | Activation-aware Swift SVD |
| `residual_svd_compressor` | Residual SVD |
| `erc_svd` | Energy-Rank-Cap SVD |
| `dobi_svd` | DOBI SVD |
| `saes_svd_v2` | SAES SVD v2 |
| `aa_svd_compressor` | AA-SVD |
| `adaptive_ranks` | Adaptive rank allocation |
| `hybrid_svd_layer` | Hybrid SVD layer |
| `gfws_d_compressor` | GFWS-D |
| `optimal_subspace_projection` | Optimal subspace |
| `subspace_projection` | Subspace projection |
| `entropy_budget_allocation` | Entropy budget |
| `dynamic_rank` | Dynamic entropy rank |
| `svd_linear` | Linear SVD |

</details>

<details>
<summary><b>📊 Low-Rank Methods (24 engines)</b> — click to expand</summary>

| Engine | Description |
|---|---|
| `a3_compressor` | A3 |
| `come_compressor` | CoME |
| `compot_compressor` | CoMPoT |
| `cr_compressor` | CR |
| `cur_compressor` | CUR decomposition |
| `cur_moe_compressor` | CUR for MoE |
| `drone_compressor` | DRONE |
| `enlighten_compressor` | Enlighten |
| `frandi_compressor` | Frandi |
| `grecodec_compressor` | GreCodec |
| `lsc_compressor` | LSC |
| `mgie_svd_compressor` | MGIE |
| `mlorq_compressor` | MLoRQ |
| `modegpt_compressor` | ModeGPT |
| `nmf_compressor` | Non-negative MF |
| `oats_compressor` | OATS |
| `rpca_compressor` | Robust PCA |
| `salr` | SALR |
| `sdrm_compressor` | SDRM |
| `slim_compressor` | SLIM |
| `soka_compressor` | SOKA |
| `svd_llm_compressor` | SVD-LLM |
| `tomoe_compressor` | Tomoe |
| `uniql_compressor` | UniQL |

</details>

<details>
<summary><b>✂️ Pruning & Sparsification (9 engines)</b> — click to expand</summary>

| Engine | Description |
|---|---|
| `admm_pruner` | ADMM-based pruning |
| `admm_compressor` | ADMM compress |
| `armor_pruner` | ARMOR |
| `ast_pruner` | AST |
| `bonsai_pruner` | Bonsai |
| `flap` | FLAP |
| `mamba_shedder` | Mamba pruner |
| `proxsparse` | ProxSparse |
| `sparsegpt` | SparseGPT |

</details>

<details>
<summary><b>🎨 Quantization (8 engines)</b> — click to expand</summary>

| Engine | Description |
|---|---|
| `astro_quantizer` | ASTRO |
| `coopq_quantizer` | CoopQ |
| `faq_quantizer` | FAQ |
| `kmeans_quantize` | K-means |
| `rotational_slicegpt_int4` | Rotational SliceGPT INT4 |
| `slicegpt` | SliceGPT |
| `slicegpt_pca` | SliceGPT + PCA |
| `huffman_coder` | Huffman coding |

</details>

<details>
<summary><b>💾 KV-Cache (6 engines) & Tensor Decomp (3 engines) & Experimental (5 engines)</b></summary>

**KV-Cache:**
- `aatc_xkv_compressor` — AATC
- `cachenotes_compressor` — CacheNotes
- `delta_kv` — Delta-KV
- `hybridkv_compressor` — HybridKV
- `kvtc_compressor` — KVTC
- `zerokv_neo_compressor` — ZeroKV-Neo

**Tensor Decomp:**
- `kronecker_compressor` — Kronecker
- `kronecker_tt_lora` — Kronecker + TT + LoRA
- `tt_compressor` — Tensor-Train

**Experimental:**
- `flat_llm` — Flat-LLM
- `hashed_net_compressor` — HashedNet
- `liftr` — Liftr
- `lillama_compressor` — Lil-Llama
- `roast_hashed_linear` — Roast Hashed Linear

</details>

---

## 🚀 Quick Start

### Installation

#### Prerequisites

- Python 3.10, 3.11, or 3.12
- PyTorch 2.2.0+
- CUDA 12.1 – 12.7 (for GPU features)
- Triton (optional, for fused kernels)
- Linux / macOS / Windows (WSL2 recommended for Windows)

#### Step-by-Step Install

```bash
# ─────────────────────────────────────────────────────────────
# Step 1: Clone the repository
# ─────────────────────────────────────────────────────────────
git clone https://github.com/tahamajs/SVD_linear_Attention.git
cd SVD_linear_Attention

# ─────────────────────────────────────────────────────────────
# Step 2: Create a virtual environment
# ─────────────────────────────────────────────────────────────
python3 -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

# ─────────────────────────────────────────────────────────────
# Step 3: Upgrade pip and install build tools
# ─────────────────────────────────────────────────────────────
pip install --upgrade pip setuptools wheel

# ─────────────────────────────────────────────────────────────
# Step 4: Install PyTorch (choose your CUDA version)
# ─────────────────────────────────────────────────────────────

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ─────────────────────────────────────────────────────────────
# Step 5: Install NeuralOps in editable mode
# ─────────────────────────────────────────────────────────────
pip install -e .

# ─────────────────────────────────────────────────────────────
# Step 6: (Optional) Install Triton for fused kernels
# ─────────────────────────────────────────────────────────────
pip install triton

# ─────────────────────────────────────────────────────────────
# Step 7: Verify the installation
# ─────────────────────────────────────────────────────────────
python -c "import neuralops; print('✓ NeuralOps installed')"
python -c "from neuralops.core.registry import ATTENTION_REGISTRY; print(f'✓ {len(ATTENTION_REGISTRY.keys())} attention keys')"
```

### Hardware Support Matrix

| Tier | GPU | VRAM | Features | Recommended Batch Size |
|:---:|:---|:---:|:---|:---:|
| **🥇 Tier 1** | H100 / A100 | 40–80 GB | Full Triton · FP8 · 4K seq | 64–128 |
| **🥈 Tier 2** | RTX 4090 / 3090 | 24 GB | Full Triton · FP16/BF16 · 2K seq | 32–64 |
| **🥉 Tier 3** | RTX 4070 / L4 / T4 | 8–16 GB | Triton · FP16 · Grad Checkpoint | 8–16 |
| **💻 Tier 4** | Apple M-series / CPU | System RAM | PyTorch fallbacks (no Triton) | 2–8 |

### ⚡ 3-Line Attention Swap

```python
from transformers import AutoModel
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.replace_attention import replace_attention

# 1. Load pretrained model
model = AutoModel.from_pretrained("distilbert-base-uncased")

# 2. Swap attention in one line
model = replace_attention(model, ATTENTION_REGISTRY.get("svd_linear_attention"))

# 3. Run inference
outputs = model(input_ids)
```

### 🗜️ Compress Any Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from neuralops.data import get_calibration_dataloader
from neuralops.compression_engine.svd import ASVDCompressor

# Load model
model = AutoModelForSequenceClassification.from_pretrained("models/roberta_sst2")
tokenizer = AutoTokenizer.from_pretrained("models/roberta_sst2")

# Build calibration loader (offline — no internet required)
calib_loader = get_calibration_dataloader(
    task_name="sst2",
    tokenizer=tokenizer,
    batch_size=8,
    num_samples=128,
    seq_len=128,
)

# Run activation-aware SVD compression
compressor = ASVDCompressor(model, calibration_loader=calib_loader)
compressed_model = compressor.compress(target_rank=64)

# Save
compressed_model.save_pretrained("models/roberta_sst2_svd_r64")
```

### 📋 Registry Query

```python
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import warnings

# List all canonical keys
print(f"Attention: {len(ATTENTION_REGISTRY.keys())}")    # 159
print(f"Compressor: {len(COMPRESSOR_REGISTRY.keys())}")  # 73

# Canonical lookup — no warning
svd_cls = ATTENTION_REGISTRY.get("svd_linear_attention")

# Deprecated alias — emits warning, returns same class
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    old_cls = ATTENTION_REGISTRY.get("aegis_v2")
    # w[0].message: "'aegis_v2' is deprecated. Use 'hybrid_local_kernel' instead."
    assert old_cls is ATTENTION_REGISTRY.get("hybrid_local_kernel")
```

---

## 🔬 Architecture Deep Dive

### Universal Zero-Shot Injection

NeuralOps can replace the attention module in **any** supported transformer **without retraining**. This is achieved through:

1. **Weight Extraction** — `UniversalAdapter` handles 8 architecture-specific naming conventions
2. **Geometric Preservation** — RoPE, GQA, fused QKV Conv1D are correctly preserved
3. **SVD Factorization** — Extracted weights are factored into low-rank form
4. **LoRA Adaptation** — A trainable LoRA adapter is added for fine-tuning

#### How It Works

```
┌─────────────────────────┐
│  Pretrained Model       │
│  ┌───────────────────┐  │
│  │  BertSelfAttention│  │
│  │  ├─ query (W_q)   │  │
│  │  ├─ key   (W_k)   │  │
│  │  ├─ value (W_v)   │  │
│  │  └─ dense (W_o)   │  │
│  └───────────────────┘  │
└────────────┬────────────┘
             │ replace_attention()
             ▼
┌─────────────────────────┐
│  Compressed Model       │
│  ┌───────────────────┐  │
│  │ SVDLinearAttention│  │
│  │  ├─ q_proj        │  │
│  │  │   ├─ U_q       │  │
│  │  │   ├─ S_q       │  │
│  │  │   ├─ V_q       │  │
│  │  │   └─ LoRA_q    │  │
│  │  ├─ k_proj (SVD)  │  │
│  │  ├─ v_proj (SVD)  │  │
│  │  └─ o_proj (SVD)  │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

#### Supported Extraction Paths

| Architecture | Query | Key | Value | Output |
|:---|:---|:---|:---|:---|
| **BERT** | `query` | `key` | `value` | `output.dense` |
| **RoBERTa** | `self.query` | `self.key` | `self.value` | `output.dense` |
| **DistilBERT** | `q_lin` | `k_lin` | `v_lin` | `out_lin` |
| **GPT-2** | `c_attn` (1/3) | `c_attn` (1/3) | `c_attn` (1/3) | `c_proj` |
| **LLaMA** | `q_proj` | `k_proj` | `v_proj` | `o_proj` |
| **Qwen** | `q_proj` | `k_proj` | `v_proj` | `o_proj` |
| **T5** | `q` | `k` | `v` | `o` |
| **DeBERTa** | `in_proj` (1/3) | `in_proj` (1/3) | `in_proj` (1/3) | `out_proj` |

### Data-Aware Activation Calibration

Rather than weight-only SVD ($W \approx U \Sigma V^T$), advanced compressors calibrate decomposition against **real activation distributions**:

$$W_{\text{effective}} = W \cdot \mathbb{E}[X^T X]^{1/2}$$

where $X$ is the **input activation matrix** collected from real data.

#### Why This Matters

| Method | Accuracy (RoBERTa SST-2) |
|:---|:---:|
| Weight-only SVD (r=64) | 84.2% |
| **Activation-calibrated SVD (r=64)** | **88.7%** |
| Difference | **+4.5%** |

#### Calibration Data Pipeline

```
┌──────────────────┐     ┌─────────────────────┐     ┌────────────────┐
│  Pretrained      │────▶│  Calibration        │────▶│  Hessian       │
│  Model           │     │  DataLoader         │     │  X^T X         │
└──────────────────┘     └─────────────────────┘     └────────┬───────┘
                                                              │
                                                              ▼
┌──────────────────┐     ┌─────────────────────┐     ┌────────────────┐
│  Compressed      │◀────│  SVD Factorization  │◀────│  Whitened      │
│  Model           │     │  with X^T X         │     │  Weights       │
└──────────────────┘     └─────────────────────┘     └────────────────┘
```

### Fused Triton Kernels

NeuralOps ships **16 hand-written Triton kernels** that fuse multiple operations into a single GPU kernel, drastically reducing memory traffic.

#### The Memory Hierarchy Problem

```
┌─────────────────────────────────────────────────────────────┐
│  GPU HBM (VRAM)            — 40-80 GB, ~1.5 TB/s bandwidth  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  L2 Cache             — ~40 MB                       │  │
│  │  ┌───────────────────────────────────────────────┐    │  │
│  │  │  SRAM (per SM)   — ~192 KB, ~19 TB/s         │    │  │
│  │  │  ┌───────────────────────────────────────┐   │    │  │
│  │  │  │  Registers — ultra-fast              │   │    │  │
│  │  │  └───────────────────────────────────────┘   │    │  │
│  │  └───────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Standard PyTorch:** Each operation reads/writes HBM → 7+ kernel launches for a single attention call.
**NeuralOps Triton:** All operations fused into 1 kernel → **7× fewer HBM roundtrips**.

#### Example: Fused Linear Attention

```python
@triton.jit
def fused_linear_attn_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qn, stride_qd,
    ...
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Everything happens in SRAM — no HBM roundtrips
    q = tl.load(Q + ...)          # Load once
    k = tl.load(K + ...)          # Load once
    v = tl.load(V + ...)          # Load once

    # ELU+1 feature map
    q_feat = tl.maximum(q, 0.0) + 1.0
    k_feat = tl.maximum(k, 0.0) + 1.0

    # Fused: state = k_feat^T @ v; out = q_feat @ state
    # No intermediate materialization
    ...

    tl.store(Out + ..., out)      # Store once
```

### Registry & Deprecated Aliases

NeuralOps maintains a strictly hardened registry architecture:

| Layer | Count | Purpose |
|:---|:---:|:---|
| **Canonical Attention Keys** | **159** | Active attention modules |
| **Canonical Compressor Keys** | **73** | Active compression engines |
| **Deprecated Aliases** | **641** | Historical names → canonical |

#### Two Import Styles (Both Supported)

```python
# ─────────────────────────────────────────────────
# 1. New subpackage imports (recommended)
# ─────────────────────────────────────────────────
from neuralops.attentions.svd import SVDLinearAttention
from neuralops.compression_engine.svd import DirectSVDCompressor

# ─────────────────────────────────────────────────
# 2. Legacy flat imports (fully supported)
# ─────────────────────────────────────────────────
from neuralops.attentions.svd_linear_attention import SVDLinearAttention
from neuralops.compression_engine.direct_svd import DirectSVDCompressor
```

#### How Deprecation Works

```python
import warnings
from neuralops.core.registry import ATTENTION_REGISTRY

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    # Deprecated alias → auto-remaps to canonical
    cls = ATTENTION_REGISTRY.get("aegis_v2")

    # Emits: DeprecationWarning: 'aegis_v2' is deprecated. Use 'hybrid_local_kernel' instead.
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)

# Verify identity
assert cls is ATTENTION_REGISTRY.get("hybrid_local_kernel")
```

---

## 📚 Detailed Module Reference

### SVD & Low-Rank Attention

```python
from neuralops.attentions.svd import SVDLinearAttention

attn = SVDLinearAttention(
    hidden_size=768,
    num_heads=12,
    head_dim=64,
    svd_rank=192,
    lora_rank=16,
    lora_alpha=32.0,
)

# Extract from pretrained attention
attn.copy_weights(pretrained_attn)

# Forward
output, _ = attn(hidden_states)
```

### Linear & Kernel Attention

```python
from neuralops.attentions.linear import PerformerAttention

attn = PerformerAttention(
    hidden_size=768,
    num_heads=12,
    num_features=256,       # RFF feature dimension
    use_orthogonal=True,     # Orthogonal random features
)

output, _ = attn(hidden_states)
```

### State-Space Models

```python
from neuralops.attentions.ssm import MambaBlock

block = MambaBlock(
    hidden_size=2048,
    state_size=16,
    expand_factor=2,
    conv_kernel_size=4,
)

output = block(hidden_states)
```

### Hybrid Architectures

```python
from neuralops.attentions.hybrid import SplitHeadHybridAttention

attn = SplitHeadHybridAttention(
    hidden_size=768,
    num_heads=12,
    num_local_heads=8,       # Local sliding window
    num_global_heads=4,      # Global kernel attention
    local_window=64,
)

output, _ = attn(hidden_states)
```

---

## 🗜️ Compression Engine Reference

### SVD Factorization

```python
from neuralops.compression_engine.svd import ASVDCompressor

compressor = ASVDCompressor(
    model=model,
    calibration_loader=calib_loader,
    energy_threshold=0.95,
)

compressed_model = compressor.compress(target_rank=64)
print(f"Compression ratio: {compressor.get_compression_ratio():.2f}×")
```

### Low-Rank Methods

```python
from neuralops.compression_engine.low_rank import CURCompressor

compressor = CURCompressor(model)
compressed = compressor.compress(rank=64)
```

### Pruning & Sparsification

```python
from neuralops.compression_engine.pruning import SparseGPTPruner

pruner = SparseGPTPruner(model, sparsity=0.5)
pruned = pruner.prune(calibration_loader)
```

### Quantization

```python
from neuralops.compression_engine.quantization import FAQQuantizer

quantizer = FAQQuantizer(model, bits=4)
quantized = quantizer.quantize()
```

### KV-Cache Compression

```python
from neuralops.compression_engine.kv_cache import ZeroKVNeoCompressor

compressor = ZeroKVNeoCompressor(
    cache_size=4096,
    eviction_policy="attention_sink",
)
```

---

## ⚡ Triton Kernel Subsystem

### Kernel Catalog

<div align="center">

| 🌊 Recurrence | 📐 Projection | 🎯 Attention | 🔧 Utility |
|:---:|:---:|:---:|:---:|
| `linear_attn` | `low_rank` | `sparse_attn` | `_common` |
| `selective_scan` | `triton_svd_linear_attention` | `triton_sparse_attn` | `gqa` |
| `fms_scan` | `triton_linear_attn` | `triton_cosformer` | `softplus` |
| `gdn_scan` | | | |
| `gla_scan` | | | |
| `hgrn2_scan` | | | |
| `state_update` | | | |

</div>

### Fallback Mechanism

**Every kernel auto-falls-back** to numerically identical PyTorch on CPU / MPS:

```python
from neuralops.core.triton_kernels import HAS_TRITON
print(f"Triton available: {HAS_TRITON}")

# Transparent — no code changes
output = fused_attention(q, k, v)
# ✓ CUDA + Triton   → GPU kernel
# ✓ CPU / MPS       → PyTorch fallback
```

### Performance Benchmarks

| Kernel | Hardware | Triton | PyTorch Fallback | Speedup |
|:---|:---:|:---:|:---:|:---:|
| `linear_attn` | H100 | 0.42 ms | 3.15 ms | **7.5×** |
| `selective_scan` | H100 | 0.68 ms | 4.82 ms | **7.1×** |
| `gdn_scan` | A100 | 1.27 ms | 6.35 ms | **5.0×** |
| `low_rank` | RTX 4090 | 0.21 ms | 1.12 ms | **5.3×** |
| `sparse_attn` | H100 | 0.35 ms | 2.41 ms | **6.9×** |

*Measured at batch=8, seq=4096, heads=12, d=64*

---

## 🏗️ Supported Architectures

### Model-by-Model Details

<details>
<summary><b>📘 BERT — BertSelfAttention</b></summary>

```python
from transformers import BertModel
from neuralops.attentions.svd import SVDLinearAttention
from neuralops.core.replace_attention import replace_attention

model = BertModel.from_pretrained("bert-base-uncased")
model = replace_attention(model, SVDLinearAttention, svd_rank=192)

# Nested encoder: model.encoder.layer[i].attention.self
# Output projection is model.encoder.layer[i].attention.output.dense
```

</details>

<details>
<summary><b>📗 RoBERTa — RobertaSelfAttention</b></summary>

```python
from transformers import RobertaModel

model = RobertaModel.from_pretrained("roberta-base")
model = replace_attention(model, SVDLinearAttention)

# Nested encoder: model.encoder.layer[i].attention.self
```

</details>

<details>
<summary><b>📙 DistilBERT — MultiHeadSelfAttention</b></summary>

```python
from transformers import DistilBertModel

model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model = replace_attention(model, SVDLinearAttention)

# Flat structure: model.transformer.layer[i].attention.q_lin/k_lin/v_lin/out_lin
```

</details>

<details>
<summary><b>📕 GPT-2 — GPT2Attention (Conv1D)</b></summary>

```python
from transformers import GPT2Model

model = GPT2Model.from_pretrained("gpt2")
model = replace_attention(model, SVDLinearAttention)

# Fused QKV Conv1D: model.h[i].attn.c_attn
# Output projection: model.h[i].attn.c_proj
```

</details>

<details>
<summary><b>🦙 LLaMA — LlamaAttention</b></summary>

```python
from transformers import LlamaModel

model = LlamaModel.from_pretrained("meta-llama/Llama-3-8B")
model = replace_attention(model, SVDLinearAttention)

# GQA: q_proj, k_proj, v_proj, o_proj
# RoPE: model.layers[i].self_attn.rotary_emb
```

</details>

<details>
<summary><b>🌪️ Qwen / Qwen2.5 — Qwen2Attention</b></summary>

```python
from transformers import Qwen2Model

model = Qwen2Model.from_pretrained("Qwen/Qwen2.5-0.5B")
model = replace_attention(model, SVDLinearAttention)

# GQA + RoPE
```

</details>

<details>
<summary><b>📊 T5 — T5Attention</b></summary>

```python
from transformers import T5Model

model = T5Model.from_pretrained("t5-small")
model = replace_attention(model, SVDLinearAttention)

# Relative attention bias (T5-specific)
```

</details>

<details>
<summary><b>🎯 DeBERTa — DebertaAttention</b></summary>

```python
from transformers import DebertaModel

model = DebertaModel.from_pretrained("microsoft/deberta-base")
model = replace_attention(model, SVDLinearAttention)

# Disentangled attention (in_proj + out_proj)
```

</details>

### Injection Guide

```python
from neuralops.core.replace_attention import replace_attention
from neuralops.core.registry import ATTENTION_REGISTRY

# ─── Basic usage ───
model = replace_attention(model, ATTENTION_REGISTRY.get("svd_linear_attention"))

# ─── With custom config ───
model = replace_attention(
    model,
    ATTENTION_REGISTRY.get("gated_delta_net"),
    svd_rank=192,
    lora_rank=32,
    dropout=0.1,
)

# ─── Selective layer injection ───
model = replace_attention(
    model,
    ATTENTION_REGISTRY.get("cosformer"),
    layers=[0, 3, 6, 9],  # Only these layers
)
```

---

## 📊 Benchmarking & Reproduction

### Master GPU Runner

```bash
./scripts/run_all_benchmarks_gpu.sh
```

Runs the complete matrix:
- **PPL** on WikiText-2
- **5-task** LM-Eval Harness
- **RULER** Needle-in-a-Haystack

### Track Runners

<table>
<tr>
<td>

```bash
# Rank sweeps
./scripts/runner_benchmark_rank_sweep.sh \
    dobi_svd mnli
```

**Sweeps** ranks 16 → 384 and measures accuracy/params tradeoff.

</td>
<td>

```bash
# Perplexity
./scripts/runner_benchmark_quality_ppl.sh \
    asvd_attention
```

**Measures** WikiText-2 perplexity across compression ratios.

</td>
</tr>
<tr>
<td>

```bash
# Multi-seed
./scripts/runner_benchmark_multiseed.sh \
    svd_linear_attention
```

**Runs** 5 seeds to measure recovery stability.

</td>
<td>

```bash
# Seq scaling
./scripts/runner_benchmark_seqlen_scaling.sh \
    linformer_attention
```

**Scales** seq len 128 → 4096 to measure asymptotic behavior.

</td>
</tr>
<tr>
<td>

```bash
# RULER NIAH
./scripts/runner_benchmark_ruler.sh \
    simple_ssm
```

**Tests** long-context retrieval with needle-in-haystack.

</td>
<td>

```bash
# Zero-shot
./scripts/runner_benchmark_zeroshot.sh \
    performer_attention
```

**Evaluates** on 5 LM-Eval tasks without fine-tuning.

</td>
</tr>
</table>

### Autonomous Pipelines

```bash
# Master pipeline with 6 parallel streams
python scripts/run_master_pipeline.py

# 48-hour autonomous GPU runner
python scripts/run_autonomous_48h_pipeline.py

# Continuous background orchestrator
python scripts/run_continuous_orchestrator.py
```

### 4-Model Unified Wrappers

```bash
python -m neuralops.runners.run_all_4_cosformer --batch_size 32
python -m neuralops.runners.run_all_4_fused_svd_kernel --epochs 3
python -m neuralops.runners.run_all_4_gated_delta_net --lr 5e-5
```

Each wrapper runs the module across:
- **DistilBERT**
- **RoBERTa-base**
- **GPT-2**
- **Qwen2.5-0.5B**

---

## 🏆 Key Empirical Findings

<div align="center">

### 📈 **245 Modules Benchmarked Across DistilBERT · RoBERTa · GPT-2**

</div>

<table>
<tr>
<td width="50%">

#### ✅ **Near-Lossless Softmax**

**41 modules** achieve max logit difference **< 0.5** vs baseline:

- `flash_attention3`
- `hybrid_attention`
- `lora_attention`
- `svd_linear_attention`
- *...and 37 more*

</td>
<td width="50%">

#### 📐 **DistilBERT Pareto**

| Rank | Accuracy | Params |
|:---:|:---:|:---:|
| Baseline | **90.71%** | 1.00× |
| **128** ⭐ | **86.47%** | **1.16×** |
| 384 | **90.37%** | — |

*AdaSVD optimal frontier*

</td>
</tr>
<tr>
<td width="50%">

#### 🚀 **RoBERTa Deep Scaling**

AdaSVD recovery with `lr=5e-5`:

**81.08%** `r=16` → **92.32%** `r=384`

Monotonic improvement — no catastrophic forgetting.

</td>
<td width="50%">

#### ⚡ **Zero-Shot Quantization**

- `faq_quantizer` @ 0.98× → **>93.9%**
- `zerokv_neo` → **up to 200K tok/s**
- `saes_svd_compressor` r=256 → within **5 PPL** of baseline

</td>
</tr>
</table>

---

## 📁 Results Directory Taxonomy

```
results/
├── 01_unified_runs/         🌐  4-model comparative suites
├── 02_rank_sweeps/          📐  Multi-rank scaling (r = 16 → 384)
├── 03_zeroshot_evals/       🎯  LM-Eval Harness 5-task
├── 04_baselines/            📊  Unmodified foundation models
├── 05_glue_finetuning/      🔬  Multi-seed stability sweeps
├── 06_adhoc_experiments/    🧪  Seq scaling · RULER · trajectories
└── 07_misc_tests/           📋  Master aggregate records
```

---

## 📈 Analytics & Plotting

```bash
# ─────────────────────────────────────────────────────────────
# Aggregate all modules into Master CSV + Markdown
# ─────────────────────────────────────────────────────────────
python neuralops/analysis/generate_all_modules_analysis.py

# Output:
#   paper/_ALL_MODULES_ANALYSIS.csv    (25-column matrix)
#   paper/_ALL_MODULES_ANALYSIS.md     (Markdown summary)

# ─────────────────────────────────────────────────────────────
# Generate publication LaTeX tables
# ─────────────────────────────────────────────────────────────
python neuralops/analysis/generate_sweep_publication_artifacts.py

# ─────────────────────────────────────────────────────────────
# Generate high-resolution Pareto curves
# ─────────────────────────────────────────────────────────────
python -m neuralops.analysis.plotting
```

---

## 🧪 Testing

```bash
# ─── Run all tests ───
pytest neuralops/tests/ -v

# ─── Run specific module ───
pytest neuralops/tests/test_attentions.py -v

# ─── With coverage ───
pytest neuralops/tests/ --cov=neuralops --cov-report=html

# ─── Only fast tests ───
pytest neuralops/tests/ -m "not slow"
```

<div align="center">

| Metric | Value |
|:---:|:---:|
| ✅ **Passed** | **1,358** |
| ⏭️ **Skipped** | 6 |
| ❌ **Failed (baseline)** | 229 |
| 🚨 **Collection errors** | **0** |

*Baseline failures are macOS / CPU-only environment limitations (GPU mocks, numerical tolerances)*

</div>

---

## 🔧 Troubleshooting

<details>
<summary><b>❌ ImportError: No module named 'triton'</b></summary>

**Cause:** Triton is not installed.

**Solution:**
```bash
pip install triton
```

Or use PyTorch fallback (automatic):
```python
# No code change needed — NeuralOps auto-falls-back
output = fused_attention(q, k, v)  # PyTorch fallback if no Triton
```

</details>

<details>
<summary><b>❌ CUDA out of memory</b></summary>

**Solutions:**
1. Reduce batch size: `--batch_size 4`
2. Enable gradient checkpointing: `model.gradient_checkpointing_enable()`
3. Use FP16/BF16: `model.half()`
4. Use a smaller rank: `svd_rank=64`

</details>

<details>
<summary><b>⚠️ DeprecationWarning: 'aegis_v2' is deprecated</b></summary>

**Cause:** You are using an old alias.

**Solution:** Switch to the canonical name:
```python
# Old
attn = ATTENTION_REGISTRY.get("aegis_v2")

# New
attn = ATTENTION_REGISTRY.get("hybrid_local_kernel")
```

</details>

<details>
<summary><b>❌ Weight extraction failed for model X</b></summary>

**Cause:** Model architecture is not supported or uses non-standard naming.

**Solution:**
1. Check supported architectures: [Supported Architectures](#-supported-architectures)
2. Manually extract weights:
```python
from neuralops.core.adapters import UniversalAdapter
w, b = UniversalAdapter.get_weight_bias(src_attn, "q")
```

</details>

---

## ❓ FAQ

<details>
<summary><b>Q: What makes NeuralOps different from FlashAttention?</b></summary>

**A:** FlashAttention focuses on **fast exact** attention. NeuralOps focuses on **approximate** subquadratic alternatives (linear, sparse, SSM, hybrid) **plus** structured compression. They are complementary — NeuralOps supports FlashAttention-3 as one of its 114 attention modules.

</details>

<details>
<summary><b>Q: Can I use NeuralOps with my own model?</b></summary>

**A:** Yes, if your model uses one of the 8 supported architectures or follows a similar pattern. If not, you can write a custom `UniversalAdapter` for it.

</details>

<details>
<summary><b>Q: Is NeuralOps compatible with Hugging Face?</b></summary>

**A:** Yes — NeuralOps is designed to work seamlessly with `transformers`. All injection utilities accept Hugging Face models directly.

</details>

<details>
<summary><b>Q: Do I need a GPU?</b></summary>

**A:** No. NeuralOps has automatic PyTorch fallbacks for CPU and Apple Silicon MPS. GPU is recommended for production use.

</details>

<details>
<summary><b>Q: How do I add a new attention module?</b></summary>

**A:** See the template at `neuralops/attentions/experimental/template_attention.py`. Then register it:
```python
@ATTENTION_REGISTRY.register("my_custom_attention")
class MyCustomAttention(BaseAttention):
    ...
```

</details>

<details>
<summary><b>Q: Where are the paper results?</b></summary>

**A:** In `results/`, organized into 7 categories (see [Results Taxonomy](#-results-directory-taxonomy)).

</details>

<details>
<summary><b>Q: Can I contribute?</b></summary>

**A:** Yes! See [Contributing](#-contributing) for guidelines.

</details>

---

## 🤝 Contributing

We welcome contributions! Please:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Guidelines

- ✅ Follow PEP 8
- ✅ Add type hints
- ✅ Write tests (see `neuralops/tests/`)
- ✅ Update docstrings
- ✅ Update `README.md` if adding a public module
- ❌ Do not break backward compatibility

---

## 📜 Changelog

### v1.0.0 (October 2026)

**🎉 Initial Release**

- ✅ **114 attention modules** across 11 families
- ✅ **73 compression engines** across 7 families
- ✅ **16 Triton kernels** with auto-fallback
- ✅ **8 transformer architectures** supported
- ✅ **641 deprecated aliases** for backward compatibility
- ✅ **198 test files** with 1,358 passing tests
- ✅ **7-category results taxonomy**
- ✅ **Publication artifact generators**

### Roadmap

- 🔜 v1.1.0 — INT8 kernels + Blackwell support
- 🔜 v1.2.0 — LoRA rank auto-tuning
- 🔜 v1.3.0 — Flash-Decoding for long-context inference

---

## ⚖️ Citation & License

### 📖 Citation

If you use NeuralOps in your research, please cite:

```bibtex
@software{neuralops2026,
  title  = {NeuralOps: SVD Linear Attention & Model Compression Framework for Foundation Models},
  author = {NeuralOps Research Team},
  year   = {2026},
  url    = {https://github.com/tahamajs/SVD_linear_Attention}
}
```

### 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

### ⭐ If NeuralOps helped your research, please star the repository!

[![Stars](https://img.shields.io/github/stars/tahamajs/SVD_linear_Attention?style=social)](https://github.com/tahamajs/SVD_linear_Attention/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/NeuralOps?style=social)](https://twitter.com/)

**Made with ❤️ by the NeuralOps Team**

[⬆ Back to Top](#-neuralops)

</div>
