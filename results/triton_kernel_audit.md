# Comprehensive Audit: Triton Kernel Requirements for Attention and Compression Modules

**Date:** 2026-09-30

## 1. Executive Summary

- **Total Attention Module Files Scanned:** 114
- **Total Compression Module Files Scanned:** 76
- **Total Modules with Sequential Token Recurrence Loops:** 21 (all in `attentions/`, 0 in `compression_engine/`)
- **Triton Kernel Status for Recurrent Modules:**
  - ✅ **Fully Verified & Integrated:** 3 modules (`mamba_block`, `hgrn2_layer`, `gated_delta_net`)
  - ⚠️ **Partial Integration (Forward exists, backward missing):** 6 modules (`gated_linear_attention`, `decay_linear_attention`, `svd_linear_attention`, `triton_svd_linear_attention`, `cosformer`, `triton_cosformer`)
  - ❌ **No Triton Kernel (Python loop fallback):** 12 modules (`factorised_matrix_state`, `chela_slay`, `goldfinch_ahn`, `granite_hybrid`, `hgrn2_outer`, `hybrid_attention`, `infini_attention`, `kda_attention`, `orion`, `rala_attention`, `regla_attention`, `samba_m3`)
- **Pure Matmul / Parallel Attention & Compression Modules:** 169 modules (already optimal via cuBLAS / PyTorch SDPA / FlashAttention)

## 2. Priority List for Kernel Development

| Rank | Module | Recurrence Type | Pretrained Transfer | Complexity | Estimated Time | Rationale |
|---|---|---|---|---|---|---|
| **1** | `factorised_matrix_state.py` | Associative matrix memory with constant decay | ✅ Yes | Low (constant decay) | 2 hours | Simplest recurrent module: scalar decay per head. 10x-50x speedup expected. |
| **2** | `gated_linear_attention.py` | Data-dependent gated linear attention | ✅ Yes | Medium (dynamic decay $\lambda_t$) | 2.5 hours | Forward kernel already in `linear_attn.py`. Need backward pass adjoints. |
| **3** | `decay_linear_attention.py` / `svd_linear_attention.py` | Shared linear attention with power/exponential decay | ✅ Yes | Medium (shared kernel with GLA) | 1 hour | Can reuse the completed `linear_attn.py` backward kernel directly. |
| **4** | `hgrn2_outer.py` / `granite_hybrid.py` | Outer product / hybrid gated recurrence | ✅ Yes | Medium | 3 hours | High potential for low-rank and state-space hybrid models. |
| **5** | `infini_attention.py` | Compressive memory matrix recurrence | ✅ Yes | High (memory retrieval + ELU kernel) | 3.5 hours | Matrix memory updates across long contexts. |

## 3. Master Audit Table

| Module File | Directory | Python Loop? | Loop Line(s) | Triton Import? | HAS_TRITON | Status |
|---|---|:---:|---|---|:---:|---|
| `chela_slay.py` | `attentions/` | ✅ | 789 | `❌` | ❌ | ❌ Need Triton |
| `cosformer.py` | `attentions/` | ✅ | 846 | `triton` | ⚠️ Partial | ⚠️ Partial (Bwd missing) |
| `decay_linear_attention.py` | `attentions/` | ✅ | 561 | `linear_attn` | ⚠️ Partial | ⚠️ Partial (Bwd missing) |
| `factorised_matrix_state.py` | `attentions/` | ✅ | 524 | `❌` | ❌ | ❌ Priority 1 (No kernel) |
| `gated_delta_net.py` | `attentions/` | ✅ | 69, 142 | `gdn_scan` | ✅ True | ✅ Ready (Triton scan) |
| `gated_linear_attention.py` | `attentions/` | ✅ | 646 | `linear_attn` | ⚠️ Partial | ⚠️ Partial (Bwd missing) |
| `goldfinch_ahn.py` | `attentions/` | ✅ | 166 | `❌` | ❌ | ❌ Need Triton |
| `granite_hybrid.py` | `attentions/` | ✅ | 151 | `❌` | ❌ | ❌ Need Triton |
| `hgrn2_layer.py` | `attentions/` | ✅ | 352 | `hgrn2_scan` | ✅ True | ✅ Ready (Triton scan) |
| `hgrn2_outer.py` | `attentions/` | ✅ | 486 | `❌` | ❌ | ❌ Need Triton |
| `hybrid_attention.py` | `attentions/` | ✅ | 400 | `❌` | ❌ | ❌ Need Triton |
| `infini_attention.py` | `attentions/` | ✅ | 672 | `❌` | ❌ | ❌ Need Triton |
| `kda_attention.py` | `attentions/` | ✅ | 516 | `❌` | ❌ | ❌ Need Triton |
| `mamba_block.py` | `attentions/` | ✅ | 508, 520 | `selective_scan` | ✅ True | ✅ Ready (Triton SSM) |
| `orion.py` | `attentions/` | ✅ | 680 | `❌` | ❌ | ❌ Need Triton |
| `performer_attention.py` | `attentions/` | ✅ | 962 | `triton` | ⚠️ | ❌ Need Triton |
| `rala_attention.py` | `attentions/` | ✅ | 535 | `❌` | ❌ | ❌ Need Triton |
| `regla_attention.py` | `attentions/` | ✅ | 580 | `❌` | ❌ | ❌ Need Triton |
| `samba_m3.py` | `attentions/` | ✅ | 527 | `❌` | ❌ | ❌ Need Triton |
| `svd_linear_attention.py` | `attentions/` | ✅ | 726, 778, 844 | `linear_attn` | ⚠️ Partial | ⚠️ Partial (Bwd missing) |
| `svd_mamba.py` | `attentions/` | ✅ | 650 | `❌` | ❌ | ❌ Need Triton |
| `triton_cosformer.py` | `attentions/` | ✅ | 360, 448 | `triton` | ⚠️ Partial | ⚠️ Partial (Bwd missing) |
| `triton_svd_linear_attention.py` | `attentions/` | ✅ | 451, 557 | `linear_attn` | ⚠️ Partial | ⚠️ Partial (Bwd missing) |
| `upi_stuffed_mamba.py` | `attentions/` | ✅ | 521 | `selective_scan` | ⚠️ | ❌ Need Triton |
| `aegis_v6.py` | `attentions/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `aegis_v5.py` | `attentions/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `compressed_attention.py` | `attentions/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `eat.py` | `attentions/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `flash_attention3.py` | `attentions/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `full_attention.py` | `attentions/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `gqa_attention.py` | `attentions/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `leka_v3.py` | `attentions/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `linformer_attention.py` | `attentions/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `mla_attention.py` | `attentions/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `direct_svd.py` | `compression_engine/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `aa_svd_compressor.py` | `compression_engine/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `adasvd_compressor.py` | `compression_engine/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `sparsegpt.py` | `compression_engine/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `slicegpt.py` | `compression_engine/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `tt_compressor.py` | `compression_engine/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| `kronecker_compressor.py` | `compression_engine/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul) |
| *(Remaining 152 modules)* | `attentions/` & `compression_engine/` | ❌ | — | ❌ | N/A | ✅ Ready (parallel matmul / SVD / prune) |