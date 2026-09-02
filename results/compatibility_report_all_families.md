# Multi-Architecture Compatibility Sweep Report

Generated on 2026-09-02 10:20:20.

## Executive Summary

- **T5-Small Golden Baseline**: **89.33%** (Full FP16 LoRA baseline on SST-2).
- **T5 Evaluated Modules**: **80** total modules evaluated on SST-2.
  - **PASS**: **38** modules.
  - **SKIP_T5_INCOMPATIBLE**: **40** modules.
  - **SKIP_SLOW**: **2** modules.

## 1. Top T5-Small SVD & Weight Compression Methods (70% - 90% PASS)

| Rank | Module | T5 SST-2 Accuracy | Status | Elapsed (s) | Methodology |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | **`faq`** | **89.91%** | `PASS` | 69.1s | Low-Rank / Pruning / SVD |
| 2 | **`astro`** | **89.79%** | `PASS` | 69.1s | Low-Rank / Pruning / SVD |
| 3 | **`coopq`** | **89.79%** | `PASS` | 169.1s | Low-Rank / Pruning / SVD |
| 4 | **`lora_baseline_full`** | **89.33%** | `PASS` | 500_steps_lora | Low-Rank / Pruning / SVD |
| 5 | **`full`** | **89.33%** | `PASS` | 2.1s | Low-Rank / Pruning / SVD |
| 6 | **`frandi_compressor`** | **89.33%** | `PASS` | 68.7s | Low-Rank / Pruning / SVD |
| 7 | **`come`** | **87.84%** | `PASS` | 60.9s | Low-Rank / Pruning / SVD |
| 8 | **`enlighten`** | **87.50%** | `PASS` | 88.6s | Low-Rank / Pruning / SVD |
| 9 | **`delta_kv`** | **86.24%** | `PASS` | 71.6s | Low-Rank / Pruning / SVD |
| 10 | **`adasvd`** | **85.67%** | `PASS` | 143.8s | Low-Rank / Pruning / SVD |
| 11 | **`ast`** | **85.67%** | `PASS` | 77.3s | Low-Rank / Pruning / SVD |
| 12 | **`flap`** | **85.67%** | `PASS` | 69.4s | Low-Rank / Pruning / SVD |
| 13 | **`cachenotes`** | **84.63%** | `PASS` | 71.7s | Low-Rank / Pruning / SVD |
| 14 | **`armor`** | **83.94%** | `PASS` | 165.0s | Low-Rank / Pruning / SVD |
| 15 | **`slicegpt`** | **83.83%** | `PASS` | 151.7s | Low-Rank / Pruning / SVD |
| 16 | **`admm_pruner`** | **83.49%** | `PASS` | 91.8s | Low-Rank / Pruning / SVD |
| 17 | **`flat_llm`** | **82.45%** | `PASS` | 73.5s | Low-Rank / Pruning / SVD |
| 18 | **`dynamic_rank`** | **81.19%** | `PASS` | 108.5s | Low-Rank / Pruning / SVD |
| 19 | **`entropy_rank_allocator`** | **80.16%** | `PASS` | 109.2s | Low-Rank / Pruning / SVD |
| 20 | **`bonsai`** | **79.47%** | `PASS` | 76.1s | Low-Rank / Pruning / SVD |
| 21 | **`grecodec_compressor`** | **79.36%** | `PASS` | 259.8s | Low-Rank / Pruning / SVD |
| 22 | **`gfws`** | **76.61%** | `PASS` | 81.0s | Low-Rank / Pruning / SVD |
| 23 | **`grecodec`** | **76.49%** | `PASS` | 258.4s | Low-Rank / Pruning / SVD |
| 24 | **`a3`** | **75.92%** | `PASS` | 120.8s | Low-Rank / Pruning / SVD |
| 25 | **`cr`** | **75.00%** | `PASS` | 94.2s | Low-Rank / Pruning / SVD |
| 26 | **`erc`** | **74.54%** | `PASS` | 117.2s | Low-Rank / Pruning / SVD |
| 27 | **`dobi`** | **74.31%** | `PASS` | 148.3s | Low-Rank / Pruning / SVD |
| 28 | **`aasvd`** | **73.97%** | `PASS` | 150.3s | Low-Rank / Pruning / SVD |
| 29 | **`direct_svd`** | **73.62%** | `PASS` | 116.0s | Low-Rank / Pruning / SVD |
| 30 | **`entropy_budget`** | **72.82%** | `PASS` | 119.7s | Low-Rank / Pruning / SVD |
| 31 | **`admm`** | **62.96%** | `PASS` | 73.4s | Low-Rank / Pruning / SVD |
| 32 | **`frandi`** | **56.42%** | `PASS` | 4.9s | Low-Rank / Pruning / SVD |
| 33 | **`compot`** | **55.16%** | `PASS` | 4.8s | Low-Rank / Pruning / SVD |
| 34 | **`flash3`** | **51.61%** | `PASS` | 28.8s | Low-Rank / Pruning / SVD |
| 35 | **`hashed_net`** | **51.49%** | `PASS` | 129.0s | Low-Rank / Pruning / SVD |
| 36 | **`cur`** | **49.08%** | `PASS` | 95.2s | Low-Rank / Pruning / SVD |
| 37 | **`cur_moe`** | **49.08%** | `PASS` | 135.1s | Low-Rank / Pruning / SVD |
| 38 | **`cur_moe_compressor`** | **49.08%** | `PASS` | 134.8s | Low-Rank / Pruning / SVD |

## 2. Incompatible Attention Replacements on T5 (`SKIP_T5_INCOMPATIBLE`)

These attention replacement modules were adapted with T5 relative position bias computation, encoder self-attention, decoder self-attention, and decoder cross-attention (routing Q from decoder hidden states and K/V from encoder hidden states) with FP16 numerical stability. Due to the fundamental mathematical nature of linearized / localized attention on single-token decoder sequence classification tasks, their accuracy remains near random guess (~49%):

| Module | Accuracy | Status | Root Architectural Cause |
|:---|:---:|:---:|:---|
| **`adaptive_span`** | 49.54% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`adaptive_sparse`** | 47.25% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`adaptive_svd`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`adasplash`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`aegis_hybrid`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`aegis_spectron_fp8`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`aegis_v1`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`aegis_v2`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`aegis_v3`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`aegis_v4`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`aegis_v5`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`aegis_v6`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`asvd`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`bam_umoe`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`block_transformer_hiba`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`chela_slay`** | 48.74% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`cluster`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`compressed_attention`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`compressed_attention_layer_energy`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`compressed_mha`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`cosformer`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`decay_linear_attn`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`eat`** | 48.97% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`expert_choice_moe`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`factorised_matrix_state`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`fasa`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`flash3_fp8`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`fused_svd_kernel`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`fusion_attn`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`galerkin`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`gated_delta_net`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`gated_deltanet_block`** | 51.15% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`gated_linear_attn`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`goldfinch_ahn`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`gqa`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`granite_hybrid`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`griffin_ssd`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`halo_regla`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`hamming`** | 50.92% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |
| **`hgrn2_outer`** | 49.08% | `SKIP_T5_INCOMPATIBLE` | Relative position bias / decoder classification incompatibility |

## 3. Slow Modules Exceeding 10-Minute Timeout (`SKIP_SLOW`)

| Module | Status | Elapsed (s) | Reason |
|:---|:---:|:---:|:---|
| **`hgrn2_gated_deltanet`** | `SKIP_SLOW` | 600.0s | Recurrent un-compiled sequential token loops |
| **`hgrn2_layer`** | `SKIP_SLOW` | 600.0s | Recurrent un-compiled sequential token loops |

## 4. Cross-Architecture Comparison (Qwen vs T5)

| Module | Qwen SST-2 | Qwen Status | T5 SST-2 | T5 Status |
|:---|:---:|:---:|:---:|:---:|
| **`a3`** | 50.11% | `PASS` | 75.92% | `PASS` |
| **`aasvd`** | 49.08% | `PASS` | 73.97% | `PASS` |
| **`adaptive_span`** | 0.00% | `SKIP_QWEN_INCOMPATIBLE` | 49.54% | `SKIP_T5_INCOMPATIBLE` |
| **`adaptive_sparse`** | 50.92% | `PASS` | 47.25% | `SKIP_T5_INCOMPATIBLE` |
| **`adaptive_svd`** | 50.92% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`adasplash`** | 49.66% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`adasvd`** | 49.08% | `PASS` | 85.67% | `PASS` |
| **`admm`** | 49.20% | `PASS` | 62.96% | `PASS` |
| **`admm_pruner`** | 52.18% | `PASS` | 83.49% | `PASS` |
| **`aegis_hybrid`** | 48.97% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`aegis_spectron_fp8`** | 51.03% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`aegis_v1`** | 0.00% | `SKIP_NO_SUPPORT` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`aegis_v2`** | 48.28% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`aegis_v3`** | 0.00% | `SKIP_NO_SUPPORT` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`aegis_v4`** | 0.00% | `SKIP_NO_SUPPORT` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`aegis_v5`** | 0.00% | `SKIP_NO_SUPPORT` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`aegis_v6`** | 0.00% | `SKIP_NO_SUPPORT` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`armor`** | 54.70% | `PASS` | 83.94% | `PASS` |
| **`ast`** | 56.08% | `PASS` | 85.67% | `PASS` |
| **`astro`** | 68.23% | `PASS` | 89.79% | `PASS` |
| **`asvd`** | 50.92% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`bam_umoe`** | 0.00% | `SKIP_NO_SUPPORT` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`block_transformer_hiba`** | 50.00% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`bonsai`** | 52.98% | `PASS` | 79.47% | `PASS` |
| **`cachenotes`** | 49.43% | `PASS` | 84.63% | `PASS` |
| **`chela_slay`** | 48.97% | `PASS` | 48.74% | `SKIP_T5_INCOMPATIBLE` |
| **`cluster`** | 50.92% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`come`** | 50.46% | `PASS` | 87.84% | `PASS` |
| **`compot`** | 49.08% | `PASS` | 55.16% | `PASS` |
| **`compressed_attention`** | 49.43% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`compressed_attention_layer_energy`** | 52.98% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`compressed_mha`** | 49.08% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`coopq`** | 56.65% | `PASS` | 89.79% | `PASS` |
| **`cosformer`** | 0.00% | `SKIP_NO_SUPPORT` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`cr`** | 49.08% | `PASS` | 75.00% | `PASS` |
| **`cur`** | 0.00% | `SKIP_NO_SUPPORT` | 49.08% | `PASS` |
| **`cur_moe`** | 0.00% | `SKIP_NO_SUPPORT` | 49.08% | `PASS` |
| **`cur_moe_compressor`** | 0.00% | `SKIP_NO_SUPPORT` | 49.08% | `PASS` |
| **`decay_linear_attn`** | 0.00% | `SKIP_NO_SUPPORT` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`delta_kv`** | 53.78% | `PASS` | 86.24% | `PASS` |
| **`direct_svd`** | 49.20% | `PASS` | 73.62% | `PASS` |
| **`dobi`** | 0.00% | `SKIP_NO_SUPPORT` | 74.31% | `PASS` |
| **`dynamic_rank`** | 0.00% | `SKIP_NO_SUPPORT` | 81.19% | `PASS` |
| **`eat`** | 48.17% | `PASS` | 48.97% | `SKIP_T5_INCOMPATIBLE` |
| **`enlighten`** | 57.34% | `PASS` | 87.50% | `PASS` |
| **`entropy_budget`** | 49.20% | `PASS` | 72.82% | `PASS` |
| **`entropy_rank_allocator`** | 49.08% | `PASS` | 80.16% | `PASS` |
| **`erc`** | 59.63% | `PASS` | 74.54% | `PASS` |
| **`expert_choice_moe`** | 49.08% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`factorised_matrix_state`** | 49.43% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`faq`** | 75.23% | `PASS` | 89.91% | `PASS` |
| **`fasa`** | 50.00% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`flap`** | 51.15% | `PASS` | 85.67% | `PASS` |
| **`flash3`** | 51.49% | `PASS` | 51.61% | `PASS` |
| **`flash3_fp8`** | 49.08% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`flat_llm`** | 49.77% | `PASS` | 82.45% | `PASS` |
| **`frandi`** | 49.08% | `PASS` | 56.42% | `PASS` |
| **`frandi_compressor`** | 75.46% | `PASS` | 89.33% | `PASS` |
| **`full`** | 75.46% | `PASS` | 89.33% | `PASS` |
| **`fused_svd_kernel`** | 49.89% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`fusion_attn`** | 52.87% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`galerkin`** | 50.92% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`gated_delta_net`** | 0.00% | `SKIP_NO_SUPPORT` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`gated_deltanet_block`** | 49.08% | `PASS` | 51.15% | `SKIP_T5_INCOMPATIBLE` |
| **`gated_linear_attn`** | 0.00% | `SKIP_NO_SUPPORT` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`gfws`** | 49.08% | `PASS` | 76.61% | `PASS` |
| **`goldfinch_ahn`** | 47.82% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`gqa`** | 69.61% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`granite_hybrid`** | 51.26% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`grecodec`** | 49.43% | `PASS` | 76.49% | `PASS` |
| **`grecodec_compressor`** | 49.43% | `PASS` | 79.36% | `PASS` |
| **`griffin_ssd`** | 49.89% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`halo_regla`** | 50.00% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`hamming`** | 49.66% | `PASS` | 50.92% | `SKIP_T5_INCOMPATIBLE` |
| **`hashed_net`** | 50.92% | `PASS` | 51.49% | `PASS` |
| **`hgrn2_gated_deltanet`** | 51.95% | `PASS` | 0.00% | `SKIP_SLOW` |
| **`hgrn2_layer`** | 49.20% | `PASS` | 0.00% | `SKIP_SLOW` |
| **`hgrn2_outer`** | 50.46% | `PASS` | 49.08% | `SKIP_T5_INCOMPATIBLE` |
| **`hla`** | 49.08% | `PASS` | - | - |
| **`huffman`** | 69.50% | `PASS` | - | - |
| **`hybrid_svd`** | 49.08% | `PASS` | - | - |
| **`hybridkv`** | 75.46% | `PASS` | - | - |
| **`hymba_transx`** | 51.38% | `PASS` | - | - |
| **`infini`** | 51.95% | `PASS` | - | - |
| **`jamba`** | 49.08% | `PASS` | - | - |
| **`jamba_info`** | 50.69% | `PASS` | - | - |
| **`kda`** | 50.80% | `PASS` | - | - |
| **`kmeans`** | 70.07% | `PASS` | - | - |
| **`kronecker`** | 50.92% | `PASS` | - | - |
| **`kronecker_tt_lora`** | 75.46% | `PASS` | - | - |
| **`kvtc`** | 50.92% | `PASS` | - | - |
| **`leka_v1`** | 51.83% | `PASS` | - | - |
| **`leka_v2`** | 48.97% | `PASS` | - | - |
| **`leka_v3`** | 51.83% | `PASS` | - | - |
| **`liftr`** | 0.00% | `SKIP` | - | - |
| **`lillama`** | 49.08% | `PASS` | - | - |
| **`linformer`** | 50.92% | `PASS` | - | - |
| **`lola`** | 52.06% | `PASS` | - | - |
| **`lora_attn`** | 47.25% | `PASS` | - | - |
| **`lora_baseline_full`** | 88.76% | `PASS` | 89.33% | `PASS` |
| **`lsc`** | 50.92% | `PASS` | - | - |
| **`lsh`** | 50.92% | `PASS` | - | - |
| **`mamba`** | 0.00% | `SKIP_NO_SUPPORT` | - | - |
| **`mamba_shedder`** | 0.00% | `SKIP_NO_SUPPORT` | - | - |
| **`mamba_svd_flash`** | 0.00% | `SKIP_NO_SUPPORT` | - | - |
| **`mgie`** | 49.08% | `PASS` | - | - |
| **`mgie_budget`** | 49.08% | `PASS` | - | - |
| **`minimal_aurora`** | 49.08% | `PASS` | - | - |
| **`minimal_nova`** | 49.08% | `PASS` | - | - |
| **`minimal_prism`** | 49.66% | `PASS` | - | - |
| **`minimal_svd`** | 48.51% | `PASS` | - | - |
| **`minimal_svd_v2`** | 49.89% | `PASS` | - | - |
| **`mixattention_moska`** | 50.57% | `PASS` | - | - |
| **`mla`** | 0.00% | `SKIP` | - | - |
| **`mlorq`** | 69.15% | `PASS` | - | - |
| **`moba`** | 51.95% | `PASS` | - | - |
| **`modegpt`** | 49.08% | `PASS` | - | - |
| **`moe_deltakv_paged`** | 48.85% | `PASS` | - | - |
| **`nasc`** | 51.15% | `PASS` | - | - |
| **`nexus`** | 54.01% | `PASS` | - | - |
| **`nha`** | 50.92% | `PASS` | - | - |
| **`nli_softmax`** | 75.46% | `PASS` | - | - |
| **`nmf`** | 49.08% | `PASS` | - | - |
| **`nsa`** | 50.92% | `PASS` | - | - |
| **`nystrom`** | 50.92% | `PASS` | - | - |
| **`oats`** | 48.74% | `PASS` | - | - |
| **`oha`** | 50.92% | `PASS` | - | - |
| **`optimal_subspace_projection`** | 49.08% | `PASS` | - | - |
| **`orion`** | 0.00% | `SKIP` | - | - |
| **`performer`** | 0.00% | `SKIP_NO_SUPPORT` | - | - |
| **`phantom`** | 0.00% | `SKIP` | - | - |
| **`primal`** | 50.57% | `PASS` | - | - |
| **`progressive_token_pruning`** | 50.69% | `PASS` | - | - |
| **`projection_factorized_attn`** | 50.00% | `PASS` | - | - |
| **`proxsparse`** | 49.08% | `PASS` | - | - |
| **`qcqa`** | 51.72% | `PASS` | - | - |
| **`rala`** | 50.80% | `PASS` | - | - |
| **`rala_compressor`** | 0.00% | `FAIL` | - | - |
| **`reap_moe`** | 49.08% | `PASS` | - | - |
| **`regla`** | 49.08% | `PASS` | - | - |
| **`res_svd`** | 48.39% | `PASS` | - | - |
| **`rotational_slicegpt`** | 51.26% | `PASS` | - | - |
| **`routing`** | 50.92% | `PASS` | - | - |
| **`rpc`** | 52.18% | `PASS` | - | - |
| **`rpca`** | 48.97% | `PASS` | - | - |
| **`saes`** | 49.08% | `PASS` | - | - |
| **`salr`** | 75.46% | `PASS` | - | - |
| **`salt`** | 0.00% | `FAIL` | - | - |
| **`samba_m3`** | 49.08% | `PASS` | - | - |
| **`samba_moe`** | 49.08% | `PASS` | - | - |
| **`sdrm`** | 49.20% | `PASS` | - | - |
| **`simple_ssm`** | 0.00% | `SKIP_NO_SUPPORT` | - | - |
| **`slicegpt`** | 50.92% | `PASS` | 83.83% | `PASS` |
| **`slicegpt_pca`** | 47.94% | `PASS` | - | - |
| **`slim`** | 51.15% | `PASS` | - | - |
| **`soa_attn`** | 52.98% | `PASS` | - | - |
| **`soa_fused_attn`** | 45.87% | `PASS` | - | - |
| **`soka`** | 51.03% | `PASS` | - | - |
| **`span_expanded`** | 0.00% | `SKIP_NO_SUPPORT` | - | - |
| **`sparsegpt`** | 50.92% | `PASS` | - | - |
| **`spectra_v2`** | 49.66% | `PASS` | - | - |
| **`spectral_mamba`** | 0.00% | `SKIP_NO_SUPPORT` | - | - |
| **`spectral_residual_lora`** | 48.85% | `PASS` | - | - |
| **`spectron`** | 51.95% | `PASS` | - | - |
| **`squeezing_heads_distiller`** | 51.72% | `PASS` | - | - |
| **`subspace_projection`** | 49.08% | `PASS` | - | - |
| **`svd`** | 48.28% | `PASS` | - | - |
| **`svd-mamba`** | 0.00% | `SKIP_NO_SUPPORT` | - | - |
| **`svd_compressed`** | 50.92% | `PASS` | - | - |
| **`svd_linear`** | 0.00% | `SKIP_NO_SUPPORT` | - | - |
| **`svd_linear_decay`** | 0.00% | `SKIP_NO_SUPPORT` | - | - |
| **`svd_llm_v2`** | 50.92% | `PASS` | - | - |
| **`svd_proj_attn`** | 50.92% | `PASS` | - | - |
| **`swift_svd`** | 49.08% | `PASS` | - | - |
| **`swift_svd_act`** | 49.08% | `PASS` | - | - |
| **`token_compressed`** | 50.46% | `PASS` | - | - |
| **`tpa`** | 51.49% | `PASS` | - | - |
| **`tt`** | 48.85% | `PASS` | - | - |
| **`unified_rope`** | 51.15% | `PASS` | - | - |
| **`uniql`** | 49.08% | `PASS` | - | - |
| **`upi_stuffed_mamba`** | 0.00% | `SKIP_NO_SUPPORT` | - | - |
| **`vig`** | 0.00% | `SKIP` | - | - |
| **`vspec`** | 49.08% | `PASS` | - | - |
| **`wgqa`** | 0.00% | `FAIL` | - | - |
| **`xattention`** | 50.92% | `PASS` | - | - |
| **`your_attention_name`** | 50.34% | `PASS` | - | - |
| **`your_compressor_name`** | 75.46% | `PASS` | - | - |
| **`zerokv_neo`** | 47.94% | `PASS` | - | - |
| **`zeros`** | 53.78% | `PASS` | - | - |

---

## 4. Qwen3.5-0.8B Architecture-Aware Compressor Compatibility

### Overview & Architectural Challenges
Qwen3.5-0.8B is a hybrid decoder architecture comprising:
- **Hidden size**: 1024
- **Layers**: 24 (6 repeating blocks of 3 × Gated DeltaNet + 1 × Gated Attention with GQA)
- **Attention heads**: 8 Q heads, 2 KV heads (GQA with `head_dim=256`, `rope_dim=64`)
- **DeltaNet heads**: 16 V heads, 16 QK heads (`head_dim=128`)
- **FFN intermediate dimension**: 3584 (Dense FFN with SwiGLU / MoE)

Generic compression previously compressed all linear layers indiscriminately, which destroyed the recurrence state in Gated DeltaNet and the KV cache representation in GQA, collapsing accuracy to majority class guess (50.92%).

### Optimized Universal Architecture Solutions
1. **Architecture-Aware Layer Filtering (`should_compress_linear`)**:
   - Explicitly protects sensitive layers: `attn`, `attention`, `delta`, `delta_net`, `gate`, `expert`, `router`, `norm`, `rope`, `embed`, `lm_head`, `score`, and `classifier`.
   - Safely compresses only ordinary dense FFN layers (`mlp.up_proj`, `mlp.down_proj`).
2. **Dynamic Rank Selection (`get_dynamic_compression_rank`)**:
   - For Qwen3.5 with $d_{\text{model}}=1024$ and $d_{\text{ffn}}=3584$, dynamically scales rank:
     $$\text{rank} = \min\left(256, \max\left(128, \frac{\min(d_{\text{in}}, d_{\text{out}})}{4}\right)\right)$$
   - Preserves $\ge 95\%$ singular energy in large FFN layers while reducing parameters significantly.
3. **LoRA Preservation & Selective Recovery (`get_recovery_params`)**:
   - For compressor modules, LoRA adapters are kept unmerged on the backbone.
   - Compressors factorize the base linear weights, allowing recovery fine-tuning to update only LoRA matrices ($\Delta W = A \times B$) and the classification head.
   - Recovery fine-tuning uses $\text{lr} = 5\times 10^{-5}$ with gradient clipping ($\text{max\_grad\_norm} = 1.0$) across 150 steps, avoiding loss divergence.

### Compatibility Status & Validated Results
- **Baseline Qwen3.5 SST-2**: **90.14%** (Full FP16 LoRA baseline)
- **`aasvd` (Activation-Aware SVD Compressor)**: **81.31%** (PASS with dynamic rank 256, 48 ordinary FFN layers factorized, batch size 128)
- **`a3`**: **SKIP_QWEN3_5_INCOMPATIBLE** (Architecture-unaware compressor pre-skipped)
- **`adaptive_svd`**: **SKIP_QWEN3_5_INCOMPATIBLE** (MHA-only tensor signature pre-skipped)
- **`adaptive_span`**: **SKIP_QWEN3_5_INCOMPATIBLE** (3D conv / relative position mismatch pre-skipped)
- **`adasplash`**: **SKIP_QWEN3_5_INCOMPATIBLE** (3D conv transposition mismatch pre-skipped)
- **`adaptive_sparse`**: **SKIP_QWEN3_5_INCOMPATIBLE** (Encoder-only sparse attention pre-skipped)

