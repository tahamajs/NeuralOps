# Per-Module Execution Status Report: Phase 1 Efficiency Sweep

**Audit Date:** 2026-08-26  
**Evaluated Artifact:** `results/gpu_server/snapshot_2026_08_26/benchmark_summary.csv`  
**Target Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU (CUDA 12.7 / Driver 566.07)  

---

## 1. Aggregate Execution Summary

| Metric | Count | Percentage |
| :--- | :---: | :---: |
| **Total Modules in Codebase / Registry** | **226** | 100.0% |
| **Total Modules Evaluated (RUN)** | **113** | 50.0% |
| **Total Modules Skipped / Blacklisted** | **0** | 0.0% |
| **Total Modules Not Run (Deferred)** | **113** | 50.0% |

---

## 2. Per-Module Execution Status

| Module Name | Category | Execution Status | Combinations Run | PASS | FAIL | TIMEOUT | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `SOAFusedAttention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `SVD-Mamba` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `a3_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `aa_svd_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `aasvd` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `adaptive_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `adaptive_ranks` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `adaptive_span_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `adaptive_sparse_attention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `adaptive_svd_attention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `adasplash_attention` | attention | **`RUN`** | 6 | 0 | 6 | 0 | 6 FAIL (value cannot be converted to type at::Ha...) |
| `adasvd_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `admm_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `admm_pruner` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `aegis_hybrid` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `aegis_spectron_fp8` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `aegis_v1` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `aegis_v2` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `aegis_v3` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `aegis_v4` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `aegis_v5` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `aegis_v6` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `armor_pruner` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `ast_pruner` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `astro_quantizer` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `asvd_attention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `bam_umoe` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `block_transformer_hiba` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `bonsai_pruner` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `cachenotes_compressor` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `chela_slay` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `cluster_attention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `come` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `compot_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `compressed_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `compressed_mha_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `coopq_quantizer` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `cosformer` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `cr_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `cur_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `cur_linear` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `cur_moe` | compression | **`RUN`** | 12 | 12 | 0 | 0 | 12 PASS |
| `cur_moe_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `cure` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `decay_linear_attention` | attention | **`RUN`** | 12 | 12 | 0 | 0 | 12 PASS |
| `delta_kv` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `direct_svd` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `dobi_svd` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `dynamic_rank` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `eat` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `enlighten_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `entropy_budget_allocation` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `erc_svd` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `expert_choice_moe` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `factorised_matrix_state` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `faq_quantizer` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `fasa` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `flap` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `flash_attention3` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `flash_attention3_fp8` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `flat_llm` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `frandi` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `frandi_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `full_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `fused_svd_kernel` | attention | **`RUN`** | 6 | 0 | 6 | 0 | 6 FAIL (at 54:17:

        # Load tile of X: [BL...) |
| `fusion_attn` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `galerkin_attention` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `gated_delta_net` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `gated_deltanet_block` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `gated_linear_attention` | attention | **`RUN`** | 12 | 12 | 0 | 0 | 12 PASS |
| `gfws_d_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `goldfinch_ahn` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `gqa_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `granite_hybrid` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `grecodec` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `grecodec_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `griffin_ssd` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `halo_regla` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `hamming_attention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `hashed_linear` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `hashed_net_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `hgrn2_gated_deltanet` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `hgrn2_layer` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `hgrn2_outer` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `hla_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `huffman_coder` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `hybrid_svd_layer` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `hybridkv` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `hybridkv_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `hymba_transx` | attention | **`RUN`** | 6 | 0 | 6 | 0 | 6 FAIL (mat1 and mat2 shapes cannot be multiplie...) |
| `infini_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `jamba` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `jamba_info` | attention | **`RUN`** | 6 | 0 | 6 | 0 | 6 FAIL (mat1 and mat2 shapes cannot be multiplie...) |
| `kda_attention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `kmeans_quantize` | compression | **`RUN`** | 6 | 0 | 6 | 0 | 6 FAIL (expected mat1 and mat2 to have the same ...) |
| `kronecker_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `kronecker_linear` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `kronecker_tt_lora` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `kvtc` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `kvtc_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `leka_v1` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `leka_v2` | attention | **`RUN`** | 6 | 0 | 6 | 0 | 6 FAIL (value cannot be converted to type at::Ha...) |
| `leka_v3` | attention | **`RUN`** | 6 | 0 | 6 | 0 | 6 FAIL (value cannot be converted to type at::Ha...) |
| `liftr` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `lillama_compressor` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `linformer_attention` | attention | **`RUN`** | 12 | 12 | 0 | 0 | 12 PASS |
| `lola_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `lora_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `lsc_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `lsh_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `mamba` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `mamba_inject` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `mamba_shedder` | compression | **`RUN`** | 12 | 12 | 0 | 0 | 12 PASS |
| `mamba_svd_flash` | attention | **`RUN`** | 6 | 0 | 0 | 6 | 6 TIMEOUT |
| `mgie_svd` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `mgie_svd_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `minimal_aurora` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `minimal_nova` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `minimal_prism` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `minimal_svd` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `minimal_svd_v2` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `mixattention_moska` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `mla_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `mlorq_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `moba` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `moba_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `modegpt` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `modegpt_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `moe_deltakv_paged` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `nasc_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `nexus` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `nha_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `nha_compressor` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `nli_softmax` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `nmf_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `nsa_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `nystrom_attention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `oats` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `oats_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `optimal_subspace_projection` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `optimized_hybrid_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `orion` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `padre_attention` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `performer_attention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `phantom` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `primal_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `progressive_token_pruning` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `projection_factorized_attn` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `proxsparse` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `proxsparse_pruner` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `pruning_utils` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `qcqa_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `rala_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `rala_compressor` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `reap_moe` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `regla_attention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `replace` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `res_svd` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `roast_hashed_linear` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `rotational_slicegpt_int4` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `routing_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `rpc_attention` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `rpca` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `rpca_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `saes_svd` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `saes_svd_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `saes_svd_v2` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `salr` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `salt` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `salt_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `samba_m3` | attention | **`RUN`** | 6 | 0 | 6 | 0 | 6 FAIL (mat1 and mat2 shapes cannot be multiplie...) |
| `samba_moe` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `sdrm` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `sdrm_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `simple_ssm` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `slicegpt` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `slicegpt_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `slicegpt_pca` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `slim_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `soa_attn` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `soka` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `soka_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `span_expanded_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `sparsegpt` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `spectra_v2` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `spectraformer_attention` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `spectral_mamba` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `spectron` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `squeezing_heads_distill` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `subspace_projection` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `svd_attention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `svd_compressed` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `svd_linear` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `svd_linear_attention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `svd_linear_decay` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `svd_llm_v2` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `svd_llm_v2_attention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `svd_proj_attn` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `svd_projection_attention` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `swift_svd` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `swift_svd_act` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `swift_svd_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `template_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `template_compressor` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `test_all_attentions_team` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `test_syntax` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `tlinformer_attention` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `token_compressed` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `token_compressed_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `tomoe_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `tpa_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `tt` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `tt_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `tt_linear` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `unified_rope` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `uniql_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `upi_stuffed_mamba` | attention | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `vig_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `vspec` | attention | **`RUN`** | 6 | 0 | 6 | 0 | 6 FAIL (value cannot be converted to type at::Ha...) |
| `vspec_attention` | attention | **`RUN`** | 6 | 0 | 6 | 0 | 6 FAIL (value cannot be converted to type at::Ha...) |
| `weighted_ensemble` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `wgqa_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `xattention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `zerokv_neo` | compression | **`RUN`** | 6 | 6 | 0 | 0 | 6 PASS |
| `zerokv_neo_compressor` | compression | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |
| `zeros_attention` | attention | `NOT_RUN` | 0 | 0 | 0 | 0 | Source file present in codebase; not included in Phase 1 720-matrix sweep |

---

## 3. Explanatory Notes & Classification Methodology

1. **`RUN`**:
   The module was explicitly instantiated and executed on the GPU across the model/dataset matrix (e.g. BERT, RoBERTa, T5 on SST-2 and MRPC).
   All PASS runs captured exact forward latency, backward latency, total step latency, and parameter counts.
   
2. **`SKIPPED_BLACKLISTED`**:
   The module exceeded the execution timeout budget during an earlier stage of the benchmark sweep and was automatically blacklisted to avoid stalling the hardware run.

3. **`NOT_RUN`**:
   The module is fully implemented in `Codes/framework/attentions/` or `Codes/framework/compression_engine/`, passed local syntax and AST extraction tests, but was not part of the active 720-combination snapshot sweep on the GPU server. These modules are scheduled for the next extended sweep cycle.
