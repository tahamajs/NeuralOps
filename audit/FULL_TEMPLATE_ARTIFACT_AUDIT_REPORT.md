# FULL_TEMPLATE_ARTIFACT_AUDIT_REPORT.md

**Audit Execution Timestamp:** 2026-09-18T10:40:00Z  
**Target Scope:** Complete forensic audit across all 217 modules in `paper/PerModulePaper/` on forensic branch `templated_2026_09_18_forensic`.  

---

## SECTION 1 — Executive Summary (Expanded)

- **Total modules audited:** 217
- **Modules with ZERO source citations in 04_method.md:** 217 (100.0%)
- **Modules with ZERO source citations anywhere:** 71 (32.7%)
- **Modules with identical SHA256 for 04_method.md (after normalization):** 146 (67.3%)
- **Modules with "0.85" repeated > 10 times:** 66 (30.4%)
- **Total artifacts detected across all modules:** 82019
- **Total Python syntax errors in code blocks:** 543
- **Total unsubstituted placeholder occurrences:** 41675
- **Total double-character artifact occurrences:** 1796
- **Average word count of 04_method.md:** 1168.99
- **Standard deviation of word count:** 139.36
- **Number of modules at exactly 1,221 words:** 146

---

## SECTION 2 — Category 1: Double-Character Artifacts (FULL TABLE)

### Pattern 1: double_percent
- **Regex:** `%%`
- **Modules affected:** 44 (20.3%)
- **Total occurrences:** 870
- **Complete list of every module that has this artifact:**
  - `a3` (count: 18)
  - `aasvd` (count: 12)
  - `aegis_v5` (count: 4)
  - `dobi_svd` (count: 14)
  - `eat` (count: 34)
  - `expert_choice_moe` (count: 46)
  - `flash3` (count: 6)
  - `flash_attention3` (count: 20)
  - `flash_attention3_fp8` (count: 18)
  - `fusion` (count: 12)
  - `gated_delta_net` (count: 22)
  - `gated_deltanet_block` (count: 4)
  - `goldfinch_ahn` (count: 26)
  - `hgrn2_gated_deltanet` (count: 42)
  - `hgrn2_layer` (count: 38)
  - `hymba_transx` (count: 28)
  - `infini` (count: 38)
  - `jamba_info` (count: 26)
  - `leka_v1` (count: 18)
  - `leka_v2` (count: 18)
  - `leka_v3` (count: 26)
  - `linformer_attention` (count: 22)
  - `lora_attention` (count: 16)
  - `minimal_aurora` (count: 2)
  - `minimal_nova` (count: 10)
  - `minimal_prism` (count: 12)
  - `minimal_svd` (count: 14)
  - `mixattention_moska` (count: 18)
  - `mla_attention` (count: 6)
  - `moba` (count: 6)
  - `nasc_attention` (count: 10)
  - `nsa` (count: 18)
  - `nystrom_attention` (count: 12)
  - `performer_attention` (count: 16)
  - `phantom` (count: 6)
  - `rala_attention` (count: 38)
  - `reap_moe` (count: 8)
  - `res_svd` (count: 2)
  - `routing_attention` (count: 26)
  - `rpc` (count: 38)
  - `svd_linear_attention` (count: 2)
  - `svd_llm_v2_attention` (count: 18)
  - `svd_mamba` (count: 40)
  - `token_compressed_attention` (count: 60)
- **First 10 examples with file:line:**
  - `a3/07_results.md:72: %%`
  - `a3/07_results.md:73: %%`
  - `a3/07_results.md:74: %%`
  - `aasvd/07_results.md:68: %%`
  - `aasvd/07_results.md:69: %%`
  - `aasvd/07_results.md:70: %%`
  - `aegis_v5/09_discussion.md:56: %%`
  - `aegis_v5/09_discussion.md:56: %%`
  - `aegis_v5/contents.md:1474: %%`
  - `dobi_svd/07_results.md:30: %%`

### Pattern 2: double_cross
- **Regex:** `××`
- **Modules affected:** 0 (0.0%)
- **Total occurrences:** 0
- **Complete list of every module that has this artifact:**
  - *(None)*
- **First 10 examples with file:line:**
  - *(None)*

### Pattern 3: duplicate_MB
- **Regex:** `\bMB\s+MB\b`
- **Modules affected:** 65 (30.0%)
- **Total occurrences:** 858
- **Complete list of every module that has this artifact:**
  - `Galerkin` (count: 14)
  - `Primal` (count: 12)
  - `SOAFusedAttention` (count: 12)
  - `a3` (count: 14)
  - `aasvd` (count: 16)
  - `adaptive_svd_attention` (count: 12)
  - `adasplash` (count: 6)
  - `aegis_v5` (count: 12)
  - `asvd` (count: 16)
  - `bam_umoe` (count: 17)
  - `chela_slay` (count: 20)
  - `cluster` (count: 2)
  - `compressed_mha` (count: 14)
  - `cosformer` (count: 20)
  - `decay_linear_attention` (count: 7)
  - `dobi_svd` (count: 14)
  - `eat` (count: 16)
  - `expert_choice_moe` (count: 20)
  - `flash3` (count: 8)
  - `flash_attention3` (count: 12)
  - `flash_attention3_fp8` (count: 12)
  - `fusion` (count: 8)
  - `gated_delta_net` (count: 34)
  - `gated_deltanet_block` (count: 8)
  - `goldfinch_ahn` (count: 14)
  - `gqa` (count: 16)
  - `hgrn2_gated_deltanet` (count: 6)
  - `hgrn2_layer` (count: 14)
  - `hymba_transx` (count: 14)
  - `infini` (count: 16)
  - `jamba_info` (count: 8)
  - `leka_v1` (count: 2)
  - `leka_v2` (count: 2)
  - `leka_v3` (count: 14)
  - `linformer_attention` (count: 12)
  - `lora_attention` (count: 6)
  - `minimal_aurora` (count: 16)
  - `minimal_nova` (count: 20)
  - `minimal_prism` (count: 20)
  - `minimal_svd` (count: 20)
  - `mixattention_moska` (count: 14)
  - `mla_attention` (count: 6)
  - `moba` (count: 14)
  - `moe_deltakv_paged` (count: 12)
  - `nasc_attention` (count: 14)
  - `nexus` (count: 12)
  - `nha` (count: 18)
  - `nsa` (count: 12)
  - `nystrom_attention` (count: 18)
  - `orion` (count: 10)
  - `padre_attention` (count: 12)
  - `performer_attention` (count: 10)
  - `phantom` (count: 16)
  - `ptp` (count: 18)
  - `rala_attention` (count: 6)
  - `reap_moe` (count: 16)
  - `res_svd` (count: 16)
  - `routing_attention` (count: 6)
  - `rpc` (count: 16)
  - `salt` (count: 4)
  - `svd_linear_attention` (count: 16)
  - `svd_llm_v2_attention` (count: 20)
  - `svd_mamba` (count: 16)
  - `token_compressed_attention` (count: 16)
  - `unified_rope` (count: 14)
- **First 10 examples with file:line:**
  - `Galerkin/07_results.md:48: MB MB`
  - `Galerkin/07_results.md:49: MB MB`
  - `Galerkin/07_results.md:50: MB MB`
  - `Primal/07_results.md:17: MB MB`
  - `Primal/07_results.md:19: MB MB`
  - `Primal/07_results.md:21: MB MB`
  - `SOAFusedAttention/07_results.md:58: MB MB`
  - `SOAFusedAttention/07_results.md:59: MB MB`
  - `SOAFusedAttention/07_results.md:60: MB MB`
  - `a3/07_results.md:83: MB MB`

### Pattern 4: duplicate_KB
- **Regex:** `\bKB\s+KB\b`
- **Modules affected:** 0 (0.0%)
- **Total occurrences:** 0
- **Complete list of every module that has this artifact:**
  - *(None)*
- **First 10 examples with file:line:**
  - *(None)*

### Pattern 5: nonsense_units
- **Regex:** `\d+\s*x\s+s\b`
- **Modules affected:** 41 (18.9%)
- **Total occurrences:** 250
- **Complete list of every module that has this artifact:**
  - `dobi_svd` (count: 6)
  - `eat` (count: 8)
  - `expert_choice_moe` (count: 4)
  - `flash3` (count: 4)
  - `flash_attention3_fp8` (count: 4)
  - `goldfinch_ahn` (count: 8)
  - `hymba_transx` (count: 6)
  - `infini` (count: 10)
  - `jamba_info` (count: 4)
  - `leka_v3` (count: 6)
  - `linformer_attention` (count: 6)
  - `lora_attention` (count: 4)
  - `minimal_aurora` (count: 8)
  - `minimal_nova` (count: 6)
  - `minimal_prism` (count: 6)
  - `minimal_svd` (count: 6)
  - `mixattention_moska` (count: 8)
  - `mla_attention` (count: 4)
  - `moba` (count: 6)
  - `moe_deltakv_paged` (count: 6)
  - `nasc_attention` (count: 6)
  - `nexus` (count: 6)
  - `nha` (count: 6)
  - `nsa` (count: 6)
  - `nystrom_attention` (count: 6)
  - `orion` (count: 6)
  - `padre_attention` (count: 6)
  - `performer_attention` (count: 8)
  - `phantom` (count: 6)
  - `ptp` (count: 6)
  - `rala_attention` (count: 4)
  - `reap_moe` (count: 6)
  - `res_svd` (count: 8)
  - `routing_attention` (count: 6)
  - `rpc` (count: 8)
  - `salt` (count: 2)
  - `svd_linear_attention` (count: 8)
  - `svd_llm_v2_attention` (count: 6)
  - `svd_mamba` (count: 8)
  - `token_compressed_attention` (count: 2)
  - `unified_rope` (count: 10)
- **First 10 examples with file:line:**
  - `dobi_svd/07_results.md:65: 42x s`
  - `dobi_svd/09_discussion.md:68: 42x s`
  - `dobi_svd/C_data.md:22: 42x s`
  - `eat/07_results.md:17: 42x s`
  - `eat/07_results.md:44: 42x s`
  - `eat/09_discussion.md:65: 42x s`
  - `expert_choice_moe/07_results.md:53: 42x s`
  - `expert_choice_moe/C_data.md:13: 42x s`
  - `expert_choice_moe/contents.md:1088: 42x s`
  - `flash3/09_discussion.md:55: 42x s`

### Pattern 6: placeholder_xx
- **Regex:** `\d+\.?\d*xx`
- **Modules affected:** 1 (0.5%)
- **Total occurrences:** 6
- **Complete list of every module that has this artifact:**
  - `fusion` (count: 6)
- **First 10 examples with file:line:**
  - `fusion/13_references.md:49: 2406.xx`
  - `fusion/13_references.md:56: 2502.xx`
  - `fusion/contents.md:1433: 2406.xx`

### Pattern 7: unsubstituted_var
- **Regex:** `\b[A-Z_]{4,}(?=[×x\*]|\b)`
- **Modules affected:** 217 (100.0%)
- **Total occurrences:** 41675
- **Complete list of every module that has this artifact:**
  - `AdaptiveSVD` (count: 602)
  - `AdaptiveSpanAttention` (count: 429)
  - `AdaptiveSparseAttention` (count: 454)
  - `Galerkin` (count: 397)
  - `HALO` (count: 470)
  - `OH` (count: 355)
  - `Primal` (count: 402)
  - `SOAFusedAttention` (count: 167)
  - `SqueezingHeadsAttentionDistillationLayer` (count: 330)
  - `a3` (count: 521)
  - `aa_svd_compressor` (count: 116)
  - `aasvd` (count: 758)
  - `adaptive_compressor` (count: 112)
  - `adaptive_ranks` (count: 116)
  - `adaptive_span_attention` (count: 116)
  - `adaptive_sparse_attention` (count: 114)
  - `adaptive_svd_attention` (count: 161)
  - `adasplash` (count: 489)
  - `adasvd_compressor` (count: 122)
  - `admm_compressor` (count: 132)
  - `admm_pruner` (count: 148)
  - `aegis_hybrid` (count: 112)
  - `aegis_spectron_fp8` (count: 112)
  - `aegis_v1` (count: 112)
  - `aegis_v2` (count: 112)
  - `aegis_v3` (count: 112)
  - `aegis_v4` (count: 112)
  - `aegis_v5` (count: 276)
  - `aegis_v6` (count: 112)
  - `armor_pruner` (count: 130)
  - `ast_pruner` (count: 128)
  - `astro_quantizer` (count: 132)
  - `asvd` (count: 610)
  - `bam_umoe` (count: 194)
  - `block_transformer_hiba` (count: 230)
  - `bonsai_pruner` (count: 128)
  - `cachenotes_compressor` (count: 132)
  - `chela_slay` (count: 356)
  - `cluster` (count: 159)
  - `come` (count: 124)
  - `compot_compressor` (count: 108)
  - `compressed_attention` (count: 112)
  - `compressed_mha` (count: 201)
  - `coopq_quantizer` (count: 128)
  - `cosformer` (count: 371)
  - `cr_compressor` (count: 108)
  - `cur_compressor` (count: 120)
  - `cur_linear` (count: 116)
  - `cur_moe` (count: 120)
  - `cure` (count: 116)
  - `decay_linear_attention` (count: 269)
  - `delta_kv` (count: 120)
  - `direct_svd` (count: 120)
  - `dobi` (count: 116)
  - `dobi_svd` (count: 444)
  - `dynamic_entropy` (count: 116)
  - `dynamic_rank` (count: 120)
  - `eat` (count: 403)
  - `enlighten_compressor` (count: 120)
  - `entropy_budget_allocation` (count: 120)
  - `erc_svd` (count: 120)
  - `expert_choice_moe` (count: 143)
  - `factorised_matrix_state` (count: 112)
  - `faq_quantizer` (count: 128)
  - `flap` (count: 122)
  - `flash3` (count: 408)
  - `flash_attention3` (count: 271)
  - `flash_attention3_fp8` (count: 573)
  - `flat_llm` (count: 120)
  - `frandi` (count: 120)
  - `full_attention` (count: 106)
  - `fused_svd_kernel` (count: 112)
  - `fusion` (count: 286)
  - `fusion_attn` (count: 112)
  - `galerkin_attention` (count: 124)
  - `gated_delta_net` (count: 430)
  - `gated_deltanet_block` (count: 371)
  - `gated_linear_attention` (count: 112)
  - `gfws_d_compressor` (count: 108)
  - `goldfinch_ahn` (count: 373)
  - `gqa` (count: 457)
  - `granite_hybrid` (count: 112)
  - `grecodec` (count: 120)
  - `griffin_ssd` (count: 112)
  - `halo_regla` (count: 112)
  - `hamming_attention` (count: 112)
  - `hashed_linear` (count: 116)
  - `hashed_net_compressor` (count: 108)
  - `hgrn2_gated_deltanet` (count: 258)
  - `hgrn2_layer` (count: 194)
  - `hgrn2_outer` (count: 112)
  - `hla_attention` (count: 124)
  - `huffman_coder` (count: 116)
  - `hybrid_svd_layer` (count: 116)
  - `hybridkv` (count: 120)
  - `hymba_transx` (count: 166)
  - `infini` (count: 379)
  - `jamba` (count: 112)
  - `jamba_info` (count: 173)
  - `kda_attention` (count: 112)
  - `kmeans_quantize` (count: 120)
  - `kronecker_compressor` (count: 120)
  - `kronecker_linear` (count: 116)
  - `kronecker_tt_lora` (count: 120)
  - `kvtc` (count: 120)
  - `leka_v1` (count: 152)
  - `leka_v2` (count: 164)
  - `leka_v3` (count: 180)
  - `liftr` (count: 120)
  - `lillama_compressor` (count: 128)
  - `linformer_attention` (count: 150)
  - `lola_attention` (count: 112)
  - `lora_attention` (count: 186)
  - `lsc_compressor` (count: 128)
  - `lsh_attention` (count: 124)
  - `mamba` (count: 112)
  - `mamba_inject` (count: 112)
  - `mamba_shedder` (count: 120)
  - `mamba_svd_flash` (count: 112)
  - `mgie_svd` (count: 112)
  - `minimal_aurora` (count: 219)
  - `minimal_nova` (count: 440)
  - `minimal_prism` (count: 465)
  - `minimal_svd` (count: 489)
  - `minimal_svd_v2` (count: 112)
  - `mixattention_moska` (count: 558)
  - `mla_attention` (count: 188)
  - `mlorq_compressor` (count: 120)
  - `moba` (count: 156)
  - `modegpt` (count: 120)
  - `moe_deltakv_paged` (count: 159)
  - `nasc_attention` (count: 479)
  - `nexus` (count: 186)
  - `nha` (count: 354)
  - `nli_softmax` (count: 120)
  - `nmf_compressor` (count: 128)
  - `nsa` (count: 333)
  - `nystrom_attention` (count: 368)
  - `oats` (count: 120)
  - `optimal_subspace_projection` (count: 120)
  - `optimized_hybrid_attention` (count: 114)
  - `orion` (count: 491)
  - `padre_attention` (count: 125)
  - `performer_attention` (count: 455)
  - `phantom` (count: 508)
  - `progressive_token_pruning` (count: 112)
  - `projection_factorized_attn` (count: 112)
  - `proxsparse` (count: 120)
  - `proxsparse_pruner` (count: 136)
  - `pruning_utils` (count: 106)
  - `ptp` (count: 361)
  - `qcqa_attention` (count: 112)
  - `rala_attention` (count: 471)
  - `reap_moe` (count: 268)
  - `regla_attention` (count: 112)
  - `replace` (count: 112)
  - `replaceme_compressor` (count: 120)
  - `res_svd` (count: 183)
  - `roast_hashed_linear` (count: 112)
  - `rotational_slicegpt_int4` (count: 120)
  - `routing_attention` (count: 325)
  - `rpc` (count: 374)
  - `rpca` (count: 120)
  - `saes_svd` (count: 106)
  - `saes_svd_v2` (count: 120)
  - `salr` (count: 120)
  - `salt` (count: 492)
  - `samba_m3` (count: 112)
  - `samba_moe` (count: 112)
  - `sdrm` (count: 122)
  - `simple_ssm` (count: 112)
  - `slicegpt` (count: 120)
  - `slicegpt_pca` (count: 120)
  - `slim_compressor` (count: 128)
  - `smart_weighted_ensemble` (count: 112)
  - `soa_attn` (count: 112)
  - `soa_fused_module` (count: 112)
  - `soa_module` (count: 112)
  - `soka` (count: 120)
  - `span_expanded_attention` (count: 112)
  - `sparsegpt` (count: 120)
  - `spectra_v2` (count: 112)
  - `spectraformer_attention` (count: 124)
  - `spectral_mamba` (count: 112)
  - `spectron` (count: 112)
  - `squeezing_heads` (count: 112)
  - `squeezing_heads_distill` (count: 112)
  - `squeezing_heads_distiller` (count: 112)
  - `subspace_projection` (count: 120)
  - `svd_attention` (count: 112)
  - `svd_compressed` (count: 112)
  - `svd_linear_attention` (count: 413)
  - `svd_linear_decay` (count: 112)
  - `svd_llm_v2_attention` (count: 451)
  - `svd_mamba` (count: 401)
  - `svd_proj_attn` (count: 112)
  - `svd_projection_attention` (count: 106)
  - `svd_sequence_attention` (count: 124)
  - `swift_svd` (count: 120)
  - `swift_svd_act` (count: 120)
  - `template_attention` (count: 112)
  - `tlinformer_attention` (count: 124)
  - `token_compressed_attention` (count: 209)
  - `tomoe` (count: 114)
  - `tpa_attention` (count: 112)
  - `tt` (count: 120)
  - `unified_rope` (count: 369)
  - `uniql_compressor` (count: 128)
  - `upi_mamba` (count: 106)
  - `upi_stuffed_mamba` (count: 112)
  - `vig_attention` (count: 112)
  - `vspec` (count: 191)
  - `weighted_ensemble` (count: 112)
  - `wgqa` (count: 112)
  - `xattention` (count: 112)
  - `zerokv_neo` (count: 120)
  - `zeros_attention` (count: 112)
- **First 10 examples with file:line:**
  - `AdaptiveSVD/00_meta.md:2: FILE`
  - `AdaptiveSVD/00_meta.md:3: PURPOSE`
  - `AdaptiveSVD/00_meta.md:4: DEPENDENCIES`
  - `AdaptiveSpanAttention/00_meta.md:2: FILE`
  - `AdaptiveSpanAttention/00_meta.md:3: PURPOSE`
  - `AdaptiveSpanAttention/00_meta.md:4: DEPENDENCIES`
  - `AdaptiveSparseAttention/00_meta.md:2: FILE`
  - `AdaptiveSparseAttention/00_meta.md:3: PURPOSE`
  - `AdaptiveSparseAttention/00_meta.md:4: DEPENDENCIES`
  - `Galerkin/00_meta.md:2: FILE`

### Pattern 8: double_curly
- **Regex:** `\{\{`
- **Modules affected:** 216 (99.5%)
- **Total occurrences:** 1196
- **Complete list of every module that has this artifact:**
  - `AdaptiveSVD` (count: 23)
  - `AdaptiveSpanAttention` (count: 8)
  - `AdaptiveSparseAttention` (count: 5)
  - `Galerkin` (count: 9)
  - `HALO` (count: 6)
  - `OH` (count: 2)
  - `Primal` (count: 5)
  - `SOAFusedAttention` (count: 4)
  - `SqueezingHeadsAttentionDistillationLayer` (count: 14)
  - `a3` (count: 37)
  - `aa_svd_compressor` (count: 4)
  - `aasvd` (count: 14)
  - `adaptive_compressor` (count: 4)
  - `adaptive_ranks` (count: 4)
  - `adaptive_span_attention` (count: 4)
  - `adaptive_sparse_attention` (count: 4)
  - `adaptive_svd_attention` (count: 4)
  - `adasplash` (count: 11)
  - `adasvd_compressor` (count: 4)
  - `admm_compressor` (count: 4)
  - `admm_pruner` (count: 4)
  - `aegis_hybrid` (count: 4)
  - `aegis_spectron_fp8` (count: 4)
  - `aegis_v1` (count: 4)
  - `aegis_v2` (count: 4)
  - `aegis_v3` (count: 4)
  - `aegis_v4` (count: 4)
  - `aegis_v5` (count: 10)
  - `aegis_v6` (count: 4)
  - `armor_pruner` (count: 4)
  - `ast_pruner` (count: 4)
  - `astro_quantizer` (count: 4)
  - `asvd` (count: 4)
  - `bam_umoe` (count: 14)
  - `block_transformer_hiba` (count: 5)
  - `bonsai_pruner` (count: 4)
  - `cachenotes_compressor` (count: 4)
  - `chela_slay` (count: 6)
  - `cluster` (count: 5)
  - `come` (count: 4)
  - `compot_compressor` (count: 4)
  - `compressed_attention` (count: 4)
  - `compressed_mha` (count: 12)
  - `coopq_quantizer` (count: 4)
  - `cosformer` (count: 6)
  - `cr_compressor` (count: 4)
  - `cur_compressor` (count: 4)
  - `cur_linear` (count: 4)
  - `cur_moe` (count: 4)
  - `cure` (count: 4)
  - `decay_linear_attention` (count: 6)
  - `delta_kv` (count: 4)
  - `direct_svd` (count: 4)
  - `dobi` (count: 4)
  - `dobi_svd` (count: 16)
  - `dynamic_entropy` (count: 4)
  - `dynamic_rank` (count: 4)
  - `eat` (count: 8)
  - `enlighten_compressor` (count: 4)
  - `entropy_budget_allocation` (count: 4)
  - `erc_svd` (count: 4)
  - `expert_choice_moe` (count: 4)
  - `factorised_matrix_state` (count: 4)
  - `faq_quantizer` (count: 4)
  - `flap` (count: 4)
  - `flash_attention3` (count: 4)
  - `flash_attention3_fp8` (count: 14)
  - `flat_llm` (count: 4)
  - `frandi` (count: 4)
  - `full_attention` (count: 4)
  - `fused_svd_kernel` (count: 4)
  - `fusion` (count: 6)
  - `fusion_attn` (count: 4)
  - `galerkin_attention` (count: 4)
  - `gated_delta_net` (count: 6)
  - `gated_deltanet_block` (count: 6)
  - `gated_linear_attention` (count: 4)
  - `gfws_d_compressor` (count: 4)
  - `goldfinch_ahn` (count: 6)
  - `gqa` (count: 14)
  - `granite_hybrid` (count: 4)
  - `grecodec` (count: 4)
  - `griffin_ssd` (count: 4)
  - `halo_regla` (count: 4)
  - `hamming_attention` (count: 4)
  - `hashed_linear` (count: 4)
  - `hashed_net_compressor` (count: 4)
  - `hgrn2_gated_deltanet` (count: 6)
  - `hgrn2_layer` (count: 8)
  - `hgrn2_outer` (count: 4)
  - `hla_attention` (count: 4)
  - `huffman_coder` (count: 4)
  - `hybrid_svd_layer` (count: 4)
  - `hybridkv` (count: 4)
  - `hymba_transx` (count: 4)
  - `infini` (count: 2)
  - `jamba` (count: 4)
  - `jamba_info` (count: 4)
  - `kda_attention` (count: 4)
  - `kmeans_quantize` (count: 4)
  - `kronecker_compressor` (count: 4)
  - `kronecker_linear` (count: 4)
  - `kronecker_tt_lora` (count: 4)
  - `kvtc` (count: 4)
  - `leka_v1` (count: 2)
  - `leka_v2` (count: 2)
  - `leka_v3` (count: 14)
  - `liftr` (count: 4)
  - `lillama_compressor` (count: 4)
  - `linformer_attention` (count: 6)
  - `lola_attention` (count: 4)
  - `lora_attention` (count: 18)
  - `lsc_compressor` (count: 4)
  - `lsh_attention` (count: 4)
  - `mamba` (count: 4)
  - `mamba_inject` (count: 4)
  - `mamba_shedder` (count: 4)
  - `mamba_svd_flash` (count: 4)
  - `mgie_svd` (count: 4)
  - `minimal_aurora` (count: 6)
  - `minimal_nova` (count: 8)
  - `minimal_prism` (count: 10)
  - `minimal_svd` (count: 10)
  - `minimal_svd_v2` (count: 4)
  - `mixattention_moska` (count: 12)
  - `mla_attention` (count: 10)
  - `mlorq_compressor` (count: 4)
  - `moba` (count: 6)
  - `modegpt` (count: 4)
  - `moe_deltakv_paged` (count: 8)
  - `nasc_attention` (count: 8)
  - `nexus` (count: 14)
  - `nha` (count: 6)
  - `nli_softmax` (count: 4)
  - `nmf_compressor` (count: 4)
  - `nsa` (count: 6)
  - `nystrom_attention` (count: 12)
  - `oats` (count: 4)
  - `optimal_subspace_projection` (count: 4)
  - `optimized_hybrid_attention` (count: 4)
  - `orion` (count: 8)
  - `padre_attention` (count: 4)
  - `performer_attention` (count: 6)
  - `phantom` (count: 6)
  - `progressive_token_pruning` (count: 4)
  - `projection_factorized_attn` (count: 4)
  - `proxsparse` (count: 4)
  - `proxsparse_pruner` (count: 4)
  - `pruning_utils` (count: 4)
  - `ptp` (count: 12)
  - `qcqa_attention` (count: 4)
  - `rala_attention` (count: 8)
  - `reap_moe` (count: 8)
  - `regla_attention` (count: 4)
  - `replace` (count: 4)
  - `replaceme_compressor` (count: 4)
  - `res_svd` (count: 12)
  - `roast_hashed_linear` (count: 4)
  - `rotational_slicegpt_int4` (count: 4)
  - `routing_attention` (count: 6)
  - `rpc` (count: 2)
  - `rpca` (count: 4)
  - `saes_svd` (count: 4)
  - `saes_svd_v2` (count: 4)
  - `salr` (count: 4)
  - `salt` (count: 18)
  - `samba_m3` (count: 4)
  - `samba_moe` (count: 4)
  - `sdrm` (count: 4)
  - `simple_ssm` (count: 4)
  - `slicegpt` (count: 4)
  - `slicegpt_pca` (count: 4)
  - `slim_compressor` (count: 4)
  - `smart_weighted_ensemble` (count: 4)
  - `soa_attn` (count: 4)
  - `soa_fused_module` (count: 4)
  - `soa_module` (count: 4)
  - `soka` (count: 4)
  - `span_expanded_attention` (count: 4)
  - `sparsegpt` (count: 4)
  - `spectra_v2` (count: 4)
  - `spectraformer_attention` (count: 4)
  - `spectral_mamba` (count: 4)
  - `spectron` (count: 4)
  - `squeezing_heads` (count: 4)
  - `squeezing_heads_distill` (count: 4)
  - `squeezing_heads_distiller` (count: 4)
  - `subspace_projection` (count: 4)
  - `svd_attention` (count: 4)
  - `svd_compressed` (count: 4)
  - `svd_linear_attention` (count: 6)
  - `svd_linear_decay` (count: 4)
  - `svd_llm_v2_attention` (count: 10)
  - `svd_mamba` (count: 2)
  - `svd_proj_attn` (count: 4)
  - `svd_projection_attention` (count: 4)
  - `svd_sequence_attention` (count: 4)
  - `swift_svd` (count: 4)
  - `swift_svd_act` (count: 4)
  - `template_attention` (count: 4)
  - `tlinformer_attention` (count: 4)
  - `token_compressed_attention` (count: 24)
  - `tomoe` (count: 4)
  - `tpa_attention` (count: 4)
  - `tt` (count: 4)
  - `unified_rope` (count: 10)
  - `uniql_compressor` (count: 4)
  - `upi_mamba` (count: 4)
  - `upi_stuffed_mamba` (count: 4)
  - `vig_attention` (count: 4)
  - `vspec` (count: 2)
  - `weighted_ensemble` (count: 4)
  - `wgqa` (count: 4)
  - `xattention` (count: 4)
  - `zerokv_neo` (count: 4)
  - `zeros_attention` (count: 4)
- **First 10 examples with file:line:**
  - `AdaptiveSVD/13_references.md:59: {{`
  - `AdaptiveSVD/13_references.md:66: {{`
  - `AdaptiveSVD/13_references.md:73: {{`
  - `AdaptiveSpanAttention/13_references.md:58: {{`
  - `AdaptiveSpanAttention/13_references.md:94: {{`
  - `AdaptiveSpanAttention/checklist.md:60: {{`
  - `AdaptiveSparseAttention/13_references.md:94: {{`
  - `AdaptiveSparseAttention/checklist.md:54: {{`
  - `AdaptiveSparseAttention/contents.md:1786: {{`
  - `Galerkin/13_references.md:37: {{`

### Pattern 9: double_close_curly
- **Regex:** `\}\}`
- **Modules affected:** 216 (99.5%)
- **Total occurrences:** 31495
- **Complete list of every module that has this artifact:**
  - `AdaptiveSVD` (count: 30)
  - `AdaptiveSpanAttention` (count: 30)
  - `AdaptiveSparseAttention` (count: 51)
  - `Galerkin` (count: 28)
  - `HALO` (count: 16)
  - `OH` (count: 18)
  - `Primal` (count: 15)
  - `SOAFusedAttention` (count: 24)
  - `SqueezingHeadsAttentionDistillationLayer` (count: 10)
  - `a3` (count: 125)
  - `aa_svd_compressor` (count: 204)
  - `aasvd` (count: 81)
  - `adaptive_compressor` (count: 204)
  - `adaptive_ranks` (count: 204)
  - `adaptive_span_attention` (count: 204)
  - `adaptive_sparse_attention` (count: 204)
  - `adaptive_svd_attention` (count: 8)
  - `adasplash` (count: 36)
  - `adasvd_compressor` (count: 204)
  - `admm_compressor` (count: 204)
  - `admm_pruner` (count: 204)
  - `aegis_hybrid` (count: 204)
  - `aegis_spectron_fp8` (count: 204)
  - `aegis_v1` (count: 204)
  - `aegis_v2` (count: 204)
  - `aegis_v3` (count: 204)
  - `aegis_v4` (count: 204)
  - `aegis_v5` (count: 18)
  - `aegis_v6` (count: 204)
  - `armor_pruner` (count: 204)
  - `ast_pruner` (count: 204)
  - `astro_quantizer` (count: 204)
  - `asvd` (count: 8)
  - `bam_umoe` (count: 33)
  - `block_transformer_hiba` (count: 224)
  - `bonsai_pruner` (count: 204)
  - `cachenotes_compressor` (count: 204)
  - `chela_slay` (count: 32)
  - `cluster` (count: 33)
  - `come` (count: 204)
  - `compot_compressor` (count: 204)
  - `compressed_attention` (count: 204)
  - `compressed_mha` (count: 14)
  - `coopq_quantizer` (count: 204)
  - `cosformer` (count: 12)
  - `cr_compressor` (count: 204)
  - `cur_compressor` (count: 204)
  - `cur_linear` (count: 204)
  - `cur_moe` (count: 204)
  - `cure` (count: 204)
  - `decay_linear_attention` (count: 206)
  - `delta_kv` (count: 204)
  - `direct_svd` (count: 204)
  - `dobi` (count: 204)
  - `dobi_svd` (count: 44)
  - `dynamic_entropy` (count: 204)
  - `dynamic_rank` (count: 204)
  - `eat` (count: 10)
  - `enlighten_compressor` (count: 204)
  - `entropy_budget_allocation` (count: 204)
  - `erc_svd` (count: 204)
  - `expert_choice_moe` (count: 14)
  - `factorised_matrix_state` (count: 204)
  - `faq_quantizer` (count: 204)
  - `flap` (count: 204)
  - `flash3` (count: 66)
  - `flash_attention3` (count: 54)
  - `flash_attention3_fp8` (count: 16)
  - `flat_llm` (count: 204)
  - `frandi` (count: 204)
  - `full_attention` (count: 204)
  - `fused_svd_kernel` (count: 204)
  - `fusion` (count: 12)
  - `fusion_attn` (count: 204)
  - `galerkin_attention` (count: 204)
  - `gated_delta_net` (count: 6)
  - `gated_deltanet_block` (count: 20)
  - `gated_linear_attention` (count: 204)
  - `gfws_d_compressor` (count: 204)
  - `goldfinch_ahn` (count: 10)
  - `gqa` (count: 14)
  - `granite_hybrid` (count: 204)
  - `grecodec` (count: 204)
  - `griffin_ssd` (count: 204)
  - `halo_regla` (count: 204)
  - `hamming_attention` (count: 204)
  - `hashed_linear` (count: 204)
  - `hashed_net_compressor` (count: 204)
  - `hgrn2_gated_deltanet` (count: 10)
  - `hgrn2_layer` (count: 2)
  - `hgrn2_outer` (count: 204)
  - `hla_attention` (count: 204)
  - `huffman_coder` (count: 204)
  - `hybrid_svd_layer` (count: 204)
  - `hybridkv` (count: 204)
  - `hymba_transx` (count: 20)
  - `infini` (count: 16)
  - `jamba` (count: 204)
  - `jamba_info` (count: 10)
  - `kda_attention` (count: 204)
  - `kmeans_quantize` (count: 204)
  - `kronecker_compressor` (count: 204)
  - `kronecker_linear` (count: 204)
  - `kronecker_tt_lora` (count: 204)
  - `kvtc` (count: 204)
  - `leka_v1` (count: 22)
  - `leka_v2` (count: 28)
  - `leka_v3` (count: 34)
  - `liftr` (count: 204)
  - `lillama_compressor` (count: 204)
  - `linformer_attention` (count: 16)
  - `lola_attention` (count: 204)
  - `lora_attention` (count: 20)
  - `lsc_compressor` (count: 204)
  - `lsh_attention` (count: 204)
  - `mamba` (count: 204)
  - `mamba_inject` (count: 204)
  - `mamba_shedder` (count: 204)
  - `mamba_svd_flash` (count: 204)
  - `mgie_svd` (count: 204)
  - `minimal_aurora` (count: 8)
  - `minimal_nova` (count: 22)
  - `minimal_prism` (count: 24)
  - `minimal_svd` (count: 22)
  - `minimal_svd_v2` (count: 204)
  - `mixattention_moska` (count: 44)
  - `mla_attention` (count: 6)
  - `mlorq_compressor` (count: 204)
  - `modegpt` (count: 204)
  - `moe_deltakv_paged` (count: 28)
  - `nasc_attention` (count: 34)
  - `nexus` (count: 14)
  - `nha` (count: 22)
  - `nli_softmax` (count: 204)
  - `nmf_compressor` (count: 204)
  - `nsa` (count: 12)
  - `nystrom_attention` (count: 12)
  - `oats` (count: 204)
  - `optimal_subspace_projection` (count: 204)
  - `optimized_hybrid_attention` (count: 204)
  - `orion` (count: 26)
  - `padre_attention` (count: 32)
  - `performer_attention` (count: 34)
  - `phantom` (count: 26)
  - `progressive_token_pruning` (count: 204)
  - `projection_factorized_attn` (count: 204)
  - `proxsparse` (count: 204)
  - `proxsparse_pruner` (count: 204)
  - `pruning_utils` (count: 204)
  - `ptp` (count: 26)
  - `qcqa_attention` (count: 204)
  - `rala_attention` (count: 24)
  - `reap_moe` (count: 6)
  - `regla_attention` (count: 204)
  - `replace` (count: 204)
  - `replaceme_compressor` (count: 204)
  - `res_svd` (count: 6)
  - `roast_hashed_linear` (count: 204)
  - `rotational_slicegpt_int4` (count: 204)
  - `routing_attention` (count: 32)
  - `rpc` (count: 16)
  - `rpca` (count: 204)
  - `saes_svd` (count: 204)
  - `saes_svd_v2` (count: 204)
  - `salr` (count: 204)
  - `salt` (count: 29)
  - `samba_m3` (count: 204)
  - `samba_moe` (count: 204)
  - `sdrm` (count: 204)
  - `simple_ssm` (count: 204)
  - `slicegpt` (count: 204)
  - `slicegpt_pca` (count: 204)
  - `slim_compressor` (count: 204)
  - `smart_weighted_ensemble` (count: 204)
  - `soa_attn` (count: 204)
  - `soa_fused_module` (count: 204)
  - `soa_module` (count: 204)
  - `soka` (count: 204)
  - `span_expanded_attention` (count: 204)
  - `sparsegpt` (count: 204)
  - `spectra_v2` (count: 204)
  - `spectraformer_attention` (count: 204)
  - `spectral_mamba` (count: 204)
  - `spectron` (count: 204)
  - `squeezing_heads` (count: 204)
  - `squeezing_heads_distill` (count: 204)
  - `squeezing_heads_distiller` (count: 204)
  - `subspace_projection` (count: 204)
  - `svd_attention` (count: 204)
  - `svd_compressed` (count: 204)
  - `svd_linear_attention` (count: 48)
  - `svd_linear_decay` (count: 204)
  - `svd_llm_v2_attention` (count: 22)
  - `svd_mamba` (count: 20)
  - `svd_proj_attn` (count: 204)
  - `svd_projection_attention` (count: 204)
  - `svd_sequence_attention` (count: 204)
  - `swift_svd` (count: 204)
  - `swift_svd_act` (count: 204)
  - `template_attention` (count: 204)
  - `tlinformer_attention` (count: 204)
  - `token_compressed_attention` (count: 12)
  - `tomoe` (count: 204)
  - `tpa_attention` (count: 204)
  - `tt` (count: 204)
  - `unified_rope` (count: 6)
  - `uniql_compressor` (count: 204)
  - `upi_mamba` (count: 204)
  - `upi_stuffed_mamba` (count: 204)
  - `vig_attention` (count: 204)
  - `vspec` (count: 10)
  - `weighted_ensemble` (count: 204)
  - `wgqa` (count: 204)
  - `xattention` (count: 204)
  - `zerokv_neo` (count: 204)
  - `zeros_attention` (count: 204)
- **First 10 examples with file:line:**
  - `AdaptiveSVD/01_intro.md:17: }}`
  - `AdaptiveSVD/01_intro.md:19: }}`
  - `AdaptiveSVD/01_intro.md:57: }}`
  - `AdaptiveSpanAttention/03_background.md:15: }}`
  - `AdaptiveSpanAttention/04_method.md:174: }}`
  - `AdaptiveSpanAttention/04_method.md:289: }}`
  - `AdaptiveSparseAttention/01_intro.md:16: }}`
  - `AdaptiveSparseAttention/03_background.md:15: }}`
  - `AdaptiveSparseAttention/03_background.md:112: }}`
  - `Galerkin/01_intro.md:74: }}`

### Pattern 10: todo_marker
- **Regex:** `<TODO>|FILL_IN|PLACEHOLDER`
- **Modules affected:** 67 (30.9%)
- **Total occurrences:** 136
- **Complete list of every module that has this artifact:**
  - `AdaptiveSVD` (count: 4)
  - `AdaptiveSpanAttention` (count: 2)
  - `AdaptiveSparseAttention` (count: 2)
  - `Galerkin` (count: 2)
  - `HALO` (count: 2)
  - `OH` (count: 2)
  - `Primal` (count: 2)
  - `SOAFusedAttention` (count: 2)
  - `SqueezingHeadsAttentionDistillationLayer` (count: 2)
  - `a3` (count: 2)
  - `aasvd` (count: 2)
  - `adaptive_svd_attention` (count: 2)
  - `adasplash` (count: 2)
  - `aegis_v5` (count: 2)
  - `asvd` (count: 2)
  - `chela_slay` (count: 2)
  - `compressed_mha` (count: 2)
  - `cosformer` (count: 2)
  - `decay_linear_attention` (count: 2)
  - `dobi_svd` (count: 2)
  - `eat` (count: 2)
  - `expert_choice_moe` (count: 2)
  - `flash_attention3` (count: 2)
  - `flash_attention3_fp8` (count: 2)
  - `gated_delta_net` (count: 2)
  - `gated_deltanet_block` (count: 2)
  - `goldfinch_ahn` (count: 2)
  - `gqa` (count: 2)
  - `hgrn2_gated_deltanet` (count: 2)
  - `hgrn2_layer` (count: 2)
  - `hymba_transx` (count: 2)
  - `infini` (count: 2)
  - `jamba_info` (count: 2)
  - `leka_v1` (count: 2)
  - `leka_v2` (count: 2)
  - `leka_v3` (count: 2)
  - `linformer_attention` (count: 2)
  - `lora_attention` (count: 2)
  - `minimal_aurora` (count: 2)
  - `minimal_nova` (count: 2)
  - `minimal_prism` (count: 2)
  - `minimal_svd` (count: 2)
  - `mixattention_moska` (count: 2)
  - `moba` (count: 2)
  - `moe_deltakv_paged` (count: 2)
  - `nasc_attention` (count: 2)
  - `nexus` (count: 2)
  - `nha` (count: 2)
  - `nsa` (count: 2)
  - `nystrom_attention` (count: 2)
  - `orion` (count: 2)
  - `padre_attention` (count: 2)
  - `performer_attention` (count: 2)
  - `phantom` (count: 2)
  - `ptp` (count: 2)
  - `rala_attention` (count: 2)
  - `reap_moe` (count: 2)
  - `res_svd` (count: 2)
  - `routing_attention` (count: 2)
  - `rpc` (count: 2)
  - `salt` (count: 2)
  - `svd_linear_attention` (count: 2)
  - `svd_llm_v2_attention` (count: 2)
  - `svd_mamba` (count: 2)
  - `token_compressed_attention` (count: 2)
  - `unified_rope` (count: 2)
  - `vspec` (count: 2)
- **First 10 examples with file:line:**
  - `AdaptiveSVD/checklist.md:7: PLACEHOLDER`
  - `AdaptiveSVD/checklist.md:89: PLACEHOLDER`
  - `AdaptiveSVD/content.md:3799: PLACEHOLDER`
  - `AdaptiveSpanAttention/checklist.md:7: PLACEHOLDER`
  - `AdaptiveSpanAttention/contents.md:2917: PLACEHOLDER`
  - `AdaptiveSparseAttention/checklist.md:7: PLACEHOLDER`
  - `AdaptiveSparseAttention/contents.md:2805: PLACEHOLDER`
  - `Galerkin/checklist.md:7: PLACEHOLDER`
  - `Galerkin/contents.md:2702: PLACEHOLDER`
  - `HALO/checklist.md:7: PLACEHOLDER`

### Pattern 11: raw_star
- **Regex:** `\*[a-z_]+\b`
- **Modules affected:** 217 (100.0%)
- **Total occurrences:** 5465
- **Complete list of every module that has this artifact:**
  - `AdaptiveSVD` (count: 70)
  - `AdaptiveSpanAttention` (count: 153)
  - `AdaptiveSparseAttention` (count: 75)
  - `Galerkin` (count: 110)
  - `HALO` (count: 48)
  - `OH` (count: 78)
  - `Primal` (count: 103)
  - `SOAFusedAttention` (count: 26)
  - `SqueezingHeadsAttentionDistillationLayer` (count: 27)
  - `a3` (count: 106)
  - `aa_svd_compressor` (count: 8)
  - `aasvd` (count: 79)
  - `adaptive_compressor` (count: 8)
  - `adaptive_ranks` (count: 8)
  - `adaptive_span_attention` (count: 8)
  - `adaptive_sparse_attention` (count: 8)
  - `adaptive_svd_attention` (count: 67)
  - `adasplash` (count: 152)
  - `adasvd_compressor` (count: 10)
  - `admm_compressor` (count: 8)
  - `admm_pruner` (count: 8)
  - `aegis_hybrid` (count: 8)
  - `aegis_spectron_fp8` (count: 10)
  - `aegis_v1` (count: 10)
  - `aegis_v2` (count: 10)
  - `aegis_v3` (count: 10)
  - `aegis_v4` (count: 10)
  - `aegis_v5` (count: 29)
  - `aegis_v6` (count: 10)
  - `armor_pruner` (count: 8)
  - `ast_pruner` (count: 8)
  - `astro_quantizer` (count: 8)
  - `asvd` (count: 37)
  - `bam_umoe` (count: 99)
  - `block_transformer_hiba` (count: 14)
  - `bonsai_pruner` (count: 8)
  - `cachenotes_compressor` (count: 8)
  - `chela_slay` (count: 53)
  - `cluster` (count: 53)
  - `come` (count: 8)
  - `compot_compressor` (count: 8)
  - `compressed_attention` (count: 10)
  - `compressed_mha` (count: 56)
  - `coopq_quantizer` (count: 8)
  - `cosformer` (count: 48)
  - `cr_compressor` (count: 8)
  - `cur_compressor` (count: 8)
  - `cur_linear` (count: 8)
  - `cur_moe` (count: 8)
  - `cure` (count: 8)
  - `decay_linear_attention` (count: 31)
  - `delta_kv` (count: 8)
  - `direct_svd` (count: 8)
  - `dobi` (count: 8)
  - `dobi_svd` (count: 80)
  - `dynamic_entropy` (count: 8)
  - `dynamic_rank` (count: 8)
  - `eat` (count: 86)
  - `enlighten_compressor` (count: 10)
  - `entropy_budget_allocation` (count: 8)
  - `erc_svd` (count: 10)
  - `expert_choice_moe` (count: 55)
  - `factorised_matrix_state` (count: 8)
  - `faq_quantizer` (count: 8)
  - `flap` (count: 8)
  - `flash3` (count: 62)
  - `flash_attention3` (count: 54)
  - `flash_attention3_fp8` (count: 56)
  - `flat_llm` (count: 8)
  - `frandi` (count: 8)
  - `full_attention` (count: 8)
  - `fused_svd_kernel` (count: 8)
  - `fusion` (count: 79)
  - `fusion_attn` (count: 8)
  - `galerkin_attention` (count: 8)
  - `gated_delta_net` (count: 74)
  - `gated_deltanet_block` (count: 102)
  - `gated_linear_attention` (count: 8)
  - `gfws_d_compressor` (count: 8)
  - `goldfinch_ahn` (count: 32)
  - `gqa` (count: 47)
  - `granite_hybrid` (count: 10)
  - `grecodec` (count: 8)
  - `griffin_ssd` (count: 10)
  - `halo_regla` (count: 8)
  - `hamming_attention` (count: 8)
  - `hashed_linear` (count: 8)
  - `hashed_net_compressor` (count: 8)
  - `hgrn2_gated_deltanet` (count: 69)
  - `hgrn2_layer` (count: 52)
  - `hgrn2_outer` (count: 8)
  - `hla_attention` (count: 8)
  - `huffman_coder` (count: 8)
  - `hybrid_svd_layer` (count: 8)
  - `hybridkv` (count: 8)
  - `hymba_transx` (count: 55)
  - `infini` (count: 52)
  - `jamba` (count: 8)
  - `jamba_info` (count: 34)
  - `kda_attention` (count: 8)
  - `kmeans_quantize` (count: 8)
  - `kronecker_compressor` (count: 8)
  - `kronecker_linear` (count: 8)
  - `kronecker_tt_lora` (count: 8)
  - `kvtc` (count: 8)
  - `leka_v1` (count: 42)
  - `leka_v2` (count: 43)
  - `leka_v3` (count: 44)
  - `liftr` (count: 8)
  - `lillama_compressor` (count: 8)
  - `linformer_attention` (count: 35)
  - `lola_attention` (count: 8)
  - `lora_attention` (count: 50)
  - `lsc_compressor` (count: 8)
  - `lsh_attention` (count: 8)
  - `mamba` (count: 8)
  - `mamba_inject` (count: 8)
  - `mamba_shedder` (count: 8)
  - `mamba_svd_flash` (count: 8)
  - `mgie_svd` (count: 10)
  - `minimal_aurora` (count: 34)
  - `minimal_nova` (count: 75)
  - `minimal_prism` (count: 84)
  - `minimal_svd` (count: 64)
  - `minimal_svd_v2` (count: 8)
  - `mixattention_moska` (count: 46)
  - `mla_attention` (count: 48)
  - `mlorq_compressor` (count: 8)
  - `moba` (count: 52)
  - `modegpt` (count: 8)
  - `moe_deltakv_paged` (count: 32)
  - `nasc_attention` (count: 44)
  - `nexus` (count: 41)
  - `nha` (count: 38)
  - `nli_softmax` (count: 8)
  - `nmf_compressor` (count: 8)
  - `nsa` (count: 58)
  - `nystrom_attention` (count: 60)
  - `oats` (count: 8)
  - `optimal_subspace_projection` (count: 8)
  - `optimized_hybrid_attention` (count: 8)
  - `orion` (count: 27)
  - `padre_attention` (count: 57)
  - `performer_attention` (count: 52)
  - `phantom` (count: 72)
  - `progressive_token_pruning` (count: 8)
  - `projection_factorized_attn` (count: 8)
  - `proxsparse` (count: 8)
  - `proxsparse_pruner` (count: 8)
  - `pruning_utils` (count: 8)
  - `ptp` (count: 49)
  - `qcqa_attention` (count: 8)
  - `rala_attention` (count: 65)
  - `reap_moe` (count: 49)
  - `regla_attention` (count: 8)
  - `replace` (count: 10)
  - `replaceme_compressor` (count: 8)
  - `res_svd` (count: 43)
  - `roast_hashed_linear` (count: 8)
  - `rotational_slicegpt_int4` (count: 8)
  - `routing_attention` (count: 75)
  - `rpc` (count: 56)
  - `rpca` (count: 8)
  - `saes_svd` (count: 8)
  - `saes_svd_v2` (count: 10)
  - `salr` (count: 8)
  - `salt` (count: 65)
  - `samba_m3` (count: 8)
  - `samba_moe` (count: 8)
  - `sdrm` (count: 8)
  - `simple_ssm` (count: 8)
  - `slicegpt` (count: 8)
  - `slicegpt_pca` (count: 8)
  - `slim_compressor` (count: 10)
  - `smart_weighted_ensemble` (count: 8)
  - `soa_attn` (count: 8)
  - `soa_fused_module` (count: 8)
  - `soa_module` (count: 8)
  - `soka` (count: 8)
  - `span_expanded_attention` (count: 8)
  - `sparsegpt` (count: 8)
  - `spectra_v2` (count: 8)
  - `spectraformer_attention` (count: 8)
  - `spectral_mamba` (count: 10)
  - `spectron` (count: 10)
  - `squeezing_heads` (count: 8)
  - `squeezing_heads_distill` (count: 8)
  - `squeezing_heads_distiller` (count: 8)
  - `subspace_projection` (count: 8)
  - `svd_attention` (count: 8)
  - `svd_compressed` (count: 10)
  - `svd_linear_attention` (count: 52)
  - `svd_linear_decay` (count: 12)
  - `svd_llm_v2_attention` (count: 58)
  - `svd_mamba` (count: 34)
  - `svd_proj_attn` (count: 8)
  - `svd_projection_attention` (count: 8)
  - `svd_sequence_attention` (count: 8)
  - `swift_svd` (count: 8)
  - `swift_svd_act` (count: 8)
  - `template_attention` (count: 8)
  - `tlinformer_attention` (count: 8)
  - `token_compressed_attention` (count: 37)
  - `tomoe` (count: 8)
  - `tpa_attention` (count: 8)
  - `tt` (count: 8)
  - `unified_rope` (count: 57)
  - `uniql_compressor` (count: 12)
  - `upi_mamba` (count: 8)
  - `upi_stuffed_mamba` (count: 8)
  - `vig_attention` (count: 8)
  - `vspec` (count: 26)
  - `weighted_ensemble` (count: 8)
  - `wgqa` (count: 8)
  - `xattention` (count: 8)
  - `zerokv_neo` (count: 8)
  - `zeros_attention` (count: 8)
- **First 10 examples with file:line:**
  - `AdaptiveSVD/00_meta.md:34: *per`
  - `AdaptiveSVD/00_meta.md:34: *energy`
  - `AdaptiveSVD/00_meta.md:41: *not`
  - `AdaptiveSpanAttention/00_meta.md:44: *continuous`
  - `AdaptiveSpanAttention/00_meta.md:51: *differentiable`
  - `AdaptiveSpanAttention/00_meta.md:77: *learn`
  - `AdaptiveSparseAttention/00_meta.md:47: *soft`
  - `AdaptiveSparseAttention/00_meta.md:72: *which`
  - `AdaptiveSparseAttention/00_meta.md:74: *learn`
  - `Galerkin/00_meta.md:38: *discretization`

### Pattern 12: double_period
- **Regex:** `\.\.\b`
- **Modules affected:** 21 (9.7%)
- **Total occurrences:** 68
- **Complete list of every module that has this artifact:**
  - `AdaptiveSparseAttention` (count: 3)
  - `Galerkin` (count: 2)
  - `OH` (count: 10)
  - `Primal` (count: 3)
  - `a3` (count: 3)
  - `asvd` (count: 4)
  - `decay_linear_attention` (count: 1)
  - `expert_choice_moe` (count: 2)
  - `flash_attention3` (count: 2)
  - `gated_delta_net` (count: 4)
  - `gated_deltanet_block` (count: 2)
  - `gqa` (count: 3)
  - `infini` (count: 2)
  - `leka_v3` (count: 2)
  - `moba` (count: 3)
  - `phantom` (count: 4)
  - `reap_moe` (count: 2)
  - `rpc` (count: 2)
  - `svd_linear_attention` (count: 8)
  - `svd_mamba` (count: 4)
  - `token_compressed_attention` (count: 2)
- **First 10 examples with file:line:**
  - `AdaptiveSparseAttention/04_method.md:154: ..`
  - `AdaptiveSparseAttention/contents.md:722: ..`
  - `AdaptiveSparseAttention/paper_full.md:648: ..`
  - `Galerkin/06_setup.md:35: ..`
  - `Galerkin/contents.md:907: ..`
  - `OH/04_method.md:299: ..`
  - `OH/04_method.md:300: ..`
  - `OH/05_architecture.md:103: ..`
  - `Primal/04_method.md:221: ..`
  - `Primal/contents.md:766: ..`

---

## SECTION 3 — Category 2: Unsubstituted Placeholders (FULL TABLE)

| Placeholder | Modules affected (count) | Total occurrences | Example file:line |
|---|---|---|---|
| `TRAINREDUCTION` | 0 | 0 | `N/A` |
| `MEM_REDUCTION` | 8 | 10 | `AdaptiveSpanAttention/contents.md:2711` |
| `{{AUTHOR_1}}` | 0 | 0 | `N/A` |
| `{{D_ACC}}` | 0 | 0 | `N/A` |
| `{{R_ACC}}` | 0 | 0 | `N/A` |
| `{{G_ACC}}` | 0 | 0 | `N/A` |
| `PLACEHOLDER` | 67 | 136 | `AdaptiveSVD/checklist.md:7` |
| `<TODO>` | 0 | 0 | `N/A` |
| `FILL_IN` | 0 | 0 | `N/A` |

### Complete Alphabetical List of Modules Containing Any of These Placeholders

Total modules: 67

- `AdaptiveSVD`
- `AdaptiveSpanAttention`
- `AdaptiveSparseAttention`
- `Galerkin`
- `HALO`
- `OH`
- `Primal`
- `SOAFusedAttention`
- `SqueezingHeadsAttentionDistillationLayer`
- `a3`
- `aasvd`
- `adaptive_svd_attention`
- `adasplash`
- `aegis_v5`
- `asvd`
- `chela_slay`
- `compressed_mha`
- `cosformer`
- `decay_linear_attention`
- `dobi_svd`
- `eat`
- `expert_choice_moe`
- `flash_attention3`
- `flash_attention3_fp8`
- `gated_delta_net`
- `gated_deltanet_block`
- `goldfinch_ahn`
- `gqa`
- `hgrn2_gated_deltanet`
- `hgrn2_layer`
- `hymba_transx`
- `infini`
- `jamba_info`
- `leka_v1`
- `leka_v2`
- `leka_v3`
- `linformer_attention`
- `lora_attention`
- `minimal_aurora`
- `minimal_nova`
- `minimal_prism`
- `minimal_svd`
- `mixattention_moska`
- `moba`
- `moe_deltakv_paged`
- `nasc_attention`
- `nexus`
- `nha`
- `nsa`
- `nystrom_attention`
- `orion`
- `padre_attention`
- `performer_attention`
- `phantom`
- `ptp`
- `rala_attention`
- `reap_moe`
- `res_svd`
- `routing_attention`
- `rpc`
- `salt`
- `svd_linear_attention`
- `svd_llm_v2_attention`
- `svd_mamba`
- `token_compressed_attention`
- `unified_rope`
- `vspec`

---

## SECTION 4 — Category 3: Repeated Numeric Placeholders (FULL)

### Number: `0.85`
- **Total occurrences:** 10765
- **Modules affected:** 216
- **Complete alphabetical list of modules, with per-module count:**
  - `AdaptiveSVD`: 45 occurrences
  - `AdaptiveSpanAttention`: 16 occurrences
  - `AdaptiveSparseAttention`: 55 occurrences
  - `Galerkin`: 11 occurrences
  - `HALO`: 7 occurrences
  - `OH`: 12 occurrences
  - `Primal`: 16 occurrences
  - `SOAFusedAttention`: 38 occurrences
  - `SqueezingHeadsAttentionDistillationLayer`: 2 occurrences
  - `a3`: 414 occurrences
  - `aa_svd_compressor`: 2 occurrences
  - `aasvd`: 473 occurrences
  - `adaptive_compressor`: 2 occurrences
  - `adaptive_ranks`: 2 occurrences
  - `adaptive_span_attention`: 4 occurrences
  - `adaptive_sparse_attention`: 2 occurrences
  - `adaptive_svd_attention`: 12 occurrences
  - `adasplash`: 33 occurrences
  - `adasvd_compressor`: 2 occurrences
  - `admm_compressor`: 2 occurrences
  - `admm_pruner`: 2 occurrences
  - `aegis_hybrid`: 2 occurrences
  - `aegis_spectron_fp8`: 2 occurrences
  - `aegis_v1`: 2 occurrences
  - `aegis_v2`: 2 occurrences
  - `aegis_v3`: 2 occurrences
  - `aegis_v4`: 2 occurrences
  - `aegis_v5`: 20 occurrences
  - `aegis_v6`: 2 occurrences
  - `armor_pruner`: 2 occurrences
  - `ast_pruner`: 2 occurrences
  - `astro_quantizer`: 2 occurrences
  - `asvd`: 56 occurrences
  - `bam_umoe`: 24 occurrences
  - `block_transformer_hiba`: 2 occurrences
  - `bonsai_pruner`: 2 occurrences
  - `cachenotes_compressor`: 2 occurrences
  - `chela_slay`: 10 occurrences
  - `cluster`: 25 occurrences
  - `come`: 2 occurrences
  - `compot_compressor`: 2 occurrences
  - `compressed_attention`: 2 occurrences
  - `coopq_quantizer`: 2 occurrences
  - `cosformer`: 6 occurrences
  - `cr_compressor`: 2 occurrences
  - `cur_compressor`: 2 occurrences
  - `cur_linear`: 2 occurrences
  - `cur_moe`: 2 occurrences
  - `cure`: 2 occurrences
  - `decay_linear_attention`: 2 occurrences
  - `delta_kv`: 2 occurrences
  - `direct_svd`: 2 occurrences
  - `dobi`: 2 occurrences
  - `dobi_svd`: 218 occurrences
  - `dynamic_entropy`: 2 occurrences
  - `dynamic_rank`: 2 occurrences
  - `eat`: 158 occurrences
  - `enlighten_compressor`: 2 occurrences
  - `entropy_budget_allocation`: 2 occurrences
  - `erc_svd`: 2 occurrences
  - `expert_choice_moe`: 190 occurrences
  - `factorised_matrix_state`: 2 occurrences
  - `faq_quantizer`: 2 occurrences
  - `flap`: 2 occurrences
  - `flash3`: 52 occurrences
  - `flash_attention3`: 46 occurrences
  - `flash_attention3_fp8`: 134 occurrences
  - `flat_llm`: 2 occurrences
  - `frandi`: 2 occurrences
  - `full_attention`: 2 occurrences
  - `fused_svd_kernel`: 2 occurrences
  - `fusion`: 108 occurrences
  - `fusion_attn`: 2 occurrences
  - `galerkin_attention`: 2 occurrences
  - `gated_delta_net`: 14 occurrences
  - `gated_deltanet_block`: 86 occurrences
  - `gated_linear_attention`: 2 occurrences
  - `gfws_d_compressor`: 2 occurrences
  - `goldfinch_ahn`: 202 occurrences
  - `gqa`: 27 occurrences
  - `granite_hybrid`: 2 occurrences
  - `grecodec`: 2 occurrences
  - `griffin_ssd`: 2 occurrences
  - `halo_regla`: 2 occurrences
  - `hamming_attention`: 2 occurrences
  - `hashed_linear`: 4 occurrences
  - `hashed_net_compressor`: 2 occurrences
  - `hgrn2_gated_deltanet`: 157 occurrences
  - `hgrn2_layer`: 150 occurrences
  - `hgrn2_outer`: 2 occurrences
  - `hla_attention`: 2 occurrences
  - `huffman_coder`: 2 occurrences
  - `hybrid_svd_layer`: 2 occurrences
  - `hybridkv`: 2 occurrences
  - `hymba_transx`: 172 occurrences
  - `infini`: 238 occurrences
  - `jamba`: 2 occurrences
  - `jamba_info`: 192 occurrences
  - `kda_attention`: 2 occurrences
  - `kmeans_quantize`: 2 occurrences
  - `kronecker_compressor`: 2 occurrences
  - `kronecker_linear`: 2 occurrences
  - `kronecker_tt_lora`: 2 occurrences
  - `kvtc`: 2 occurrences
  - `leka_v1`: 156 occurrences
  - `leka_v2`: 230 occurrences
  - `leka_v3`: 339 occurrences
  - `liftr`: 2 occurrences
  - `lillama_compressor`: 2 occurrences
  - `linformer_attention`: 164 occurrences
  - `lola_attention`: 2 occurrences
  - `lora_attention`: 232 occurrences
  - `lsc_compressor`: 2 occurrences
  - `lsh_attention`: 2 occurrences
  - `mamba`: 2 occurrences
  - `mamba_inject`: 2 occurrences
  - `mamba_shedder`: 4 occurrences
  - `mamba_svd_flash`: 2 occurrences
  - `mgie_svd`: 2 occurrences
  - `minimal_aurora`: 270 occurrences
  - `minimal_nova`: 280 occurrences
  - `minimal_prism`: 288 occurrences
  - `minimal_svd`: 350 occurrences
  - `minimal_svd_v2`: 2 occurrences
  - `mixattention_moska`: 144 occurrences
  - `mla_attention`: 188 occurrences
  - `mlorq_compressor`: 2 occurrences
  - `moba`: 139 occurrences
  - `modegpt`: 2 occurrences
  - `moe_deltakv_paged`: 238 occurrences
  - `nasc_attention`: 194 occurrences
  - `nexus`: 154 occurrences
  - `nha`: 367 occurrences
  - `nli_softmax`: 2 occurrences
  - `nmf_compressor`: 2 occurrences
  - `nsa`: 300 occurrences
  - `nystrom_attention`: 182 occurrences
  - `oats`: 2 occurrences
  - `optimal_subspace_projection`: 2 occurrences
  - `optimized_hybrid_attention`: 2 occurrences
  - `orion`: 108 occurrences
  - `padre_attention`: 134 occurrences
  - `performer_attention`: 121 occurrences
  - `phantom`: 376 occurrences
  - `progressive_token_pruning`: 2 occurrences
  - `projection_factorized_attn`: 2 occurrences
  - `proxsparse`: 2 occurrences
  - `proxsparse_pruner`: 2 occurrences
  - `pruning_utils`: 2 occurrences
  - `ptp`: 243 occurrences
  - `qcqa_attention`: 2 occurrences
  - `rala_attention`: 102 occurrences
  - `reap_moe`: 132 occurrences
  - `regla_attention`: 2 occurrences
  - `replace`: 2 occurrences
  - `replaceme_compressor`: 2 occurrences
  - `res_svd`: 128 occurrences
  - `roast_hashed_linear`: 2 occurrences
  - `rotational_slicegpt_int4`: 2 occurrences
  - `routing_attention`: 194 occurrences
  - `rpc`: 83 occurrences
  - `rpca`: 2 occurrences
  - `saes_svd`: 2 occurrences
  - `saes_svd_v2`: 4 occurrences
  - `salr`: 2 occurrences
  - `salt`: 94 occurrences
  - `samba_m3`: 2 occurrences
  - `samba_moe`: 2 occurrences
  - `sdrm`: 2 occurrences
  - `simple_ssm`: 2 occurrences
  - `slicegpt`: 2 occurrences
  - `slicegpt_pca`: 2 occurrences
  - `slim_compressor`: 2 occurrences
  - `smart_weighted_ensemble`: 2 occurrences
  - `soa_attn`: 2 occurrences
  - `soa_fused_module`: 2 occurrences
  - `soa_module`: 2 occurrences
  - `soka`: 2 occurrences
  - `span_expanded_attention`: 2 occurrences
  - `sparsegpt`: 2 occurrences
  - `spectra_v2`: 2 occurrences
  - `spectraformer_attention`: 2 occurrences
  - `spectral_mamba`: 4 occurrences
  - `spectron`: 2 occurrences
  - `squeezing_heads`: 2 occurrences
  - `squeezing_heads_distill`: 2 occurrences
  - `squeezing_heads_distiller`: 2 occurrences
  - `subspace_projection`: 4 occurrences
  - `svd_attention`: 2 occurrences
  - `svd_compressed`: 2 occurrences
  - `svd_linear_attention`: 236 occurrences
  - `svd_linear_decay`: 2 occurrences
  - `svd_llm_v2_attention`: 292 occurrences
  - `svd_mamba`: 289 occurrences
  - `svd_proj_attn`: 2 occurrences
  - `svd_projection_attention`: 2 occurrences
  - `svd_sequence_attention`: 2 occurrences
  - `swift_svd`: 2 occurrences
  - `swift_svd_act`: 2 occurrences
  - `template_attention`: 2 occurrences
  - `tlinformer_attention`: 2 occurrences
  - `token_compressed_attention`: 158 occurrences
  - `tomoe`: 2 occurrences
  - `tpa_attention`: 2 occurrences
  - `tt`: 2 occurrences
  - `unified_rope`: 124 occurrences
  - `uniql_compressor`: 2 occurrences
  - `upi_mamba`: 2 occurrences
  - `upi_stuffed_mamba`: 2 occurrences
  - `vig_attention`: 2 occurrences
  - `vspec`: 13 occurrences
  - `weighted_ensemble`: 2 occurrences
  - `wgqa`: 2 occurrences
  - `xattention`: 2 occurrences
  - `zerokv_neo`: 2 occurrences
  - `zeros_attention`: 2 occurrences
- **Context examples (10 samples):**
  - `AdaptiveSVD/00_meta.md:15`: "- ² 0.85, 0.85"
  - `AdaptiveSVD/00_meta.md:15`: "- ² 0.85, 0.85"
  - `AdaptiveSVD/04_method.md:337`: "| $\tau$ | 0.92 | 0.85 – 0.99 |"
  - `AdaptiveSVD/04_method.md:344`: "- $\tau = 0.85$: aggressive, may lose accuracy"
  - `AdaptiveSVD/07_results.md:99`: "| 0.85 | 64 | 85 | 5.0× | 70 | 71 |"
  - `AdaptiveSVD/07_results.md:105`: "**Observation.** τ=0.92 is a sweet spot. τ=0.85 loses accuracy; τ=0.99 gives"
  - `AdaptiveSVD/08_analysis.md:70`: "| No adapter (pure truncation) | 0.85 | 0.85 | 0.85 |"
  - `AdaptiveSVD/08_analysis.md:70`: "| No adapter (pure truncation) | 0.85 | 0.85 | 0.85 |"
  - `AdaptiveSVD/08_analysis.md:70`: "| No adapter (pure truncation) | 0.85 | 0.85 | 0.85 |"
  - `AdaptiveSVD/08_analysis.md:83`: "| 0.85 | 85 | 5.0× | Aggressive, loses accuracy |"

### Number: `0.85%`
- **Total occurrences:** 810
- **Modules affected:** 53
- **Complete alphabetical list of modules, with per-module count:**
  - `SqueezingHeadsAttentionDistillationLayer`: 2 occurrences
  - `a3`: 12 occurrences
  - `adasplash`: 12 occurrences
  - `chela_slay`: 2 occurrences
  - `cosformer`: 2 occurrences
  - `dobi_svd`: 20 occurrences
  - `eat`: 6 occurrences
  - `expert_choice_moe`: 42 occurrences
  - `flash3`: 2 occurrences
  - `flash_attention3`: 6 occurrences
  - `flash_attention3_fp8`: 22 occurrences
  - `fusion`: 8 occurrences
  - `gated_delta_net`: 6 occurrences
  - `gated_deltanet_block`: 10 occurrences
  - `goldfinch_ahn`: 24 occurrences
  - `hgrn2_gated_deltanet`: 18 occurrences
  - `hgrn2_layer`: 18 occurrences
  - `hymba_transx`: 14 occurrences
  - `infini`: 4 occurrences
  - `jamba_info`: 30 occurrences
  - `leka_v1`: 16 occurrences
  - `leka_v2`: 24 occurrences
  - `leka_v3`: 22 occurrences
  - `linformer_attention`: 10 occurrences
  - `lora_attention`: 8 occurrences
  - `minimal_aurora`: 4 occurrences
  - `minimal_nova`: 4 occurrences
  - `minimal_prism`: 4 occurrences
  - `minimal_svd`: 20 occurrences
  - `mixattention_moska`: 4 occurrences
  - `mla_attention`: 4 occurrences
  - `moba`: 8 occurrences
  - `moe_deltakv_paged`: 14 occurrences
  - `nasc_attention`: 18 occurrences
  - `nexus`: 14 occurrences
  - `nha`: 4 occurrences
  - `nsa`: 72 occurrences
  - `nystrom_attention`: 2 occurrences
  - `orion`: 6 occurrences
  - `padre_attention`: 6 occurrences
  - `performer_attention`: 10 occurrences
  - `phantom`: 30 occurrences
  - `ptp`: 88 occurrences
  - `rala_attention`: 6 occurrences
  - `reap_moe`: 48 occurrences
  - `res_svd`: 4 occurrences
  - `routing_attention`: 6 occurrences
  - `salt`: 22 occurrences
  - `svd_linear_attention`: 10 occurrences
  - `svd_llm_v2_attention`: 2 occurrences
  - `svd_mamba`: 18 occurrences
  - `token_compressed_attention`: 26 occurrences
  - `unified_rope`: 16 occurrences
- **Context examples (10 samples):**
  - `SqueezingHeadsAttentionDistillationLayer/00_meta.md:56`: "parameters drop by **2×** with only **0.85%**"
  - `SqueezingHeadsAttentionDistillationLayer/contents.md:118`: "parameters drop by **2×** with only **0.85%**"
  - `a3/08_analysis.md:130`: "| 0.10 | 0.85 | 0.85% | 32 |"
  - `a3/08_analysis.md:131`: "| 0.20 | 0.85 | 0.85% | 32 |"
  - `a3/08_analysis.md:132`: "| 0.30 | 0.85 | 0.85% | 32 |"
  - `a3/08_analysis.md:133`: "| 0.50 | 0.85 | 0.85% | 32 |"
  - `a3/content.md:1635`: "| 0.10 | 0.85 | 0.85% | 32 |"
  - `a3/content.md:1636`: "| 0.20 | 0.85 | 0.85% | 32 |"
  - `a3/content.md:1637`: "| 0.30 | 0.85 | 0.85% | 32 |"
  - `a3/content.md:1638`: "| 0.50 | 0.85 | 0.85% | 32 |"

### Number: `86.4`
- **Total occurrences:** 973
- **Modules affected:** 198
- **Complete alphabetical list of modules, with per-module count:**
  - `AdaptiveSVD`: 3 occurrences
  - `a3`: 39 occurrences
  - `aa_svd_compressor`: 2 occurrences
  - `aasvd`: 9 occurrences
  - `adaptive_compressor`: 2 occurrences
  - `adaptive_ranks`: 2 occurrences
  - `adaptive_span_attention`: 2 occurrences
  - `adaptive_sparse_attention`: 2 occurrences
  - `adasplash`: 3 occurrences
  - `adasvd_compressor`: 2 occurrences
  - `admm_compressor`: 2 occurrences
  - `admm_pruner`: 2 occurrences
  - `aegis_hybrid`: 2 occurrences
  - `aegis_spectron_fp8`: 2 occurrences
  - `aegis_v1`: 2 occurrences
  - `aegis_v2`: 2 occurrences
  - `aegis_v3`: 2 occurrences
  - `aegis_v4`: 2 occurrences
  - `aegis_v5`: 12 occurrences
  - `aegis_v6`: 2 occurrences
  - `armor_pruner`: 2 occurrences
  - `ast_pruner`: 2 occurrences
  - `astro_quantizer`: 2 occurrences
  - `bam_umoe`: 3 occurrences
  - `block_transformer_hiba`: 2 occurrences
  - `bonsai_pruner`: 2 occurrences
  - `cachenotes_compressor`: 2 occurrences
  - `come`: 2 occurrences
  - `compot_compressor`: 2 occurrences
  - `compressed_attention`: 2 occurrences
  - `coopq_quantizer`: 2 occurrences
  - `cr_compressor`: 2 occurrences
  - `cur_compressor`: 2 occurrences
  - `cur_linear`: 2 occurrences
  - `cur_moe`: 2 occurrences
  - `cure`: 2 occurrences
  - `decay_linear_attention`: 2 occurrences
  - `delta_kv`: 2 occurrences
  - `direct_svd`: 2 occurrences
  - `dobi`: 2 occurrences
  - `dobi_svd`: 40 occurrences
  - `dynamic_entropy`: 2 occurrences
  - `dynamic_rank`: 2 occurrences
  - `eat`: 14 occurrences
  - `enlighten_compressor`: 2 occurrences
  - `entropy_budget_allocation`: 2 occurrences
  - `erc_svd`: 2 occurrences
  - `expert_choice_moe`: 4 occurrences
  - `factorised_matrix_state`: 2 occurrences
  - `faq_quantizer`: 2 occurrences
  - `flap`: 2 occurrences
  - `flash3`: 12 occurrences
  - `flash_attention3`: 10 occurrences
  - `flash_attention3_fp8`: 4 occurrences
  - `flat_llm`: 2 occurrences
  - `frandi`: 2 occurrences
  - `full_attention`: 2 occurrences
  - `fused_svd_kernel`: 2 occurrences
  - `fusion`: 38 occurrences
  - `fusion_attn`: 2 occurrences
  - `galerkin_attention`: 2 occurrences
  - `gated_delta_net`: 24 occurrences
  - `gated_deltanet_block`: 10 occurrences
  - `gated_linear_attention`: 4 occurrences
  - `gfws_d_compressor`: 2 occurrences
  - `granite_hybrid`: 2 occurrences
  - `grecodec`: 2 occurrences
  - `griffin_ssd`: 2 occurrences
  - `halo_regla`: 2 occurrences
  - `hamming_attention`: 2 occurrences
  - `hashed_linear`: 2 occurrences
  - `hashed_net_compressor`: 2 occurrences
  - `hgrn2_gated_deltanet`: 18 occurrences
  - `hgrn2_layer`: 12 occurrences
  - `hgrn2_outer`: 2 occurrences
  - `hla_attention`: 2 occurrences
  - `huffman_coder`: 2 occurrences
  - `hybrid_svd_layer`: 2 occurrences
  - `hybridkv`: 2 occurrences
  - `hymba_transx`: 10 occurrences
  - `infini`: 2 occurrences
  - `jamba`: 2 occurrences
  - `jamba_info`: 6 occurrences
  - `kda_attention`: 2 occurrences
  - `kmeans_quantize`: 2 occurrences
  - `kronecker_compressor`: 2 occurrences
  - `kronecker_linear`: 2 occurrences
  - `kronecker_tt_lora`: 2 occurrences
  - `kvtc`: 2 occurrences
  - `leka_v1`: 12 occurrences
  - `leka_v2`: 10 occurrences
  - `leka_v3`: 12 occurrences
  - `liftr`: 2 occurrences
  - `lillama_compressor`: 2 occurrences
  - `linformer_attention`: 14 occurrences
  - `lola_attention`: 2 occurrences
  - `lora_attention`: 12 occurrences
  - `lsc_compressor`: 2 occurrences
  - `lsh_attention`: 2 occurrences
  - `mamba`: 2 occurrences
  - `mamba_inject`: 2 occurrences
  - `mamba_shedder`: 2 occurrences
  - `mamba_svd_flash`: 2 occurrences
  - `mgie_svd`: 2 occurrences
  - `minimal_aurora`: 2 occurrences
  - `minimal_nova`: 14 occurrences
  - `minimal_prism`: 16 occurrences
  - `minimal_svd`: 18 occurrences
  - `minimal_svd_v2`: 2 occurrences
  - `mixattention_moska`: 10 occurrences
  - `mlorq_compressor`: 2 occurrences
  - `moba`: 4 occurrences
  - `modegpt`: 2 occurrences
  - `moe_deltakv_paged`: 14 occurrences
  - `nexus`: 2 occurrences
  - `nha`: 4 occurrences
  - `nli_softmax`: 2 occurrences
  - `nmf_compressor`: 2 occurrences
  - `nsa`: 16 occurrences
  - `nystrom_attention`: 46 occurrences
  - `oats`: 2 occurrences
  - `optimal_subspace_projection`: 2 occurrences
  - `optimized_hybrid_attention`: 2 occurrences
  - `orion`: 6 occurrences
  - `padre_attention`: 4 occurrences
  - `performer_attention`: 2 occurrences
  - `phantom`: 6 occurrences
  - `progressive_token_pruning`: 2 occurrences
  - `projection_factorized_attn`: 2 occurrences
  - `proxsparse`: 2 occurrences
  - `proxsparse_pruner`: 2 occurrences
  - `pruning_utils`: 2 occurrences
  - `ptp`: 28 occurrences
  - `qcqa_attention`: 4 occurrences
  - `rala_attention`: 12 occurrences
  - `reap_moe`: 76 occurrences
  - `regla_attention`: 2 occurrences
  - `replace`: 2 occurrences
  - `replaceme_compressor`: 2 occurrences
  - `res_svd`: 2 occurrences
  - `roast_hashed_linear`: 2 occurrences
  - `rotational_slicegpt_int4`: 2 occurrences
  - `routing_attention`: 2 occurrences
  - `rpc`: 8 occurrences
  - `rpca`: 2 occurrences
  - `saes_svd`: 2 occurrences
  - `saes_svd_v2`: 2 occurrences
  - `salr`: 2 occurrences
  - `salt`: 6 occurrences
  - `samba_m3`: 2 occurrences
  - `samba_moe`: 2 occurrences
  - `sdrm`: 2 occurrences
  - `simple_ssm`: 2 occurrences
  - `slicegpt`: 2 occurrences
  - `slicegpt_pca`: 2 occurrences
  - `slim_compressor`: 2 occurrences
  - `smart_weighted_ensemble`: 2 occurrences
  - `soa_attn`: 2 occurrences
  - `soa_fused_module`: 2 occurrences
  - `soa_module`: 2 occurrences
  - `soka`: 2 occurrences
  - `span_expanded_attention`: 2 occurrences
  - `sparsegpt`: 2 occurrences
  - `spectra_v2`: 2 occurrences
  - `spectraformer_attention`: 2 occurrences
  - `spectral_mamba`: 2 occurrences
  - `spectron`: 2 occurrences
  - `squeezing_heads`: 2 occurrences
  - `squeezing_heads_distill`: 2 occurrences
  - `squeezing_heads_distiller`: 2 occurrences
  - `subspace_projection`: 2 occurrences
  - `svd_attention`: 2 occurrences
  - `svd_compressed`: 2 occurrences
  - `svd_linear_attention`: 16 occurrences
  - `svd_linear_decay`: 2 occurrences
  - `svd_llm_v2_attention`: 10 occurrences
  - `svd_mamba`: 2 occurrences
  - `svd_proj_attn`: 2 occurrences
  - `svd_projection_attention`: 2 occurrences
  - `svd_sequence_attention`: 2 occurrences
  - `swift_svd`: 2 occurrences
  - `swift_svd_act`: 2 occurrences
  - `template_attention`: 2 occurrences
  - `tlinformer_attention`: 2 occurrences
  - `token_compressed_attention`: 14 occurrences
  - `tomoe`: 2 occurrences
  - `tpa_attention`: 2 occurrences
  - `tt`: 2 occurrences
  - `unified_rope`: 4 occurrences
  - `uniql_compressor`: 2 occurrences
  - `upi_mamba`: 2 occurrences
  - `upi_stuffed_mamba`: 2 occurrences
  - `vig_attention`: 2 occurrences
  - `weighted_ensemble`: 2 occurrences
  - `wgqa`: 2 occurrences
  - `xattention`: 2 occurrences
  - `zerokv_neo`: 2 occurrences
  - `zeros_attention`: 2 occurrences
- **Context examples (10 samples):**
  - `AdaptiveSVD/08_analysis.md:132`: "| Head pruning (50%) | 86.4 | 84% | Hard (removes heads) |"
  - `AdaptiveSVD/content.md:1660`: "| Head pruning (50%) | 86.4 | 84% | Hard (removes heads) |"
  - `AdaptiveSVD/paper_full.md:1587`: "| Head pruning (50%) | 86.4 | 84% | Hard (removes heads) |"
  - `a3/07_results.md:135`: "| 50 | 86.4 | 78.2 | 77.5 |"
  - `a3/08_analysis.md:43`: "| DistilBERT | SVD (unweighted) | 86.4\% | 0.024 |"
  - `a3/08_analysis.md:45`: "| RoBERTa | SVD | 86.4\% | 0.024 |"
  - `a3/09_discussion.md:89`: "| **A3 (r=128)** | Low-rank | **2.5×** | **86.4\%%** |"
  - `a3/C_data.md:89`: "50    86.4"
  - `a3/D_negative_results.md:23`: "**Result.** Fisher weighting accuracy degraded to **86.4\%%**."
  - `a3/D_negative_results.md:52`: "**Result.** Accuracy collapsed to **86.4\%%**."

### Number: `0.32`
- **Total occurrences:** 28
- **Modules affected:** 14
- **Complete alphabetical list of modules, with per-module count:**
  - `AdaptiveSparseAttention`: 4 occurrences
  - `aasvd`: 1 occurrences
  - `aegis_v5`: 1 occurrences
  - `compressed_mha`: 1 occurrences
  - `direct_svd`: 2 occurrences
  - `dobi_svd`: 3 occurrences
  - `gated_deltanet_block`: 1 occurrences
  - `hymba_transx`: 1 occurrences
  - `kronecker_compressor`: 4 occurrences
  - `nli_softmax`: 2 occurrences
  - `res_svd`: 1 occurrences
  - `salt`: 1 occurrences
  - `slim_compressor`: 2 occurrences
  - `unified_rope`: 4 occurrences
- **Context examples (10 samples):**
  - `AdaptiveSparseAttention/07_results.md:54`: "| 0.3 | 89.0 | 77.0 | 77.5 | 0.32 |"
  - `AdaptiveSparseAttention/contents.md:1146`: "| 0.3 | 89.0 | 77.0 | 77.5 | 0.32 |"
  - `AdaptiveSparseAttention/contents.md:2674`: ""EFF30": "0.32","
  - `AdaptiveSparseAttention/paper_full.md:1063`: "| 0.3 | 89.0 | 77.0 | 77.5 | 0.32 |"
  - `aasvd/content.md:3185`: ""R_AVG_ERR": "0.32","
  - `aegis_v5/contents.md:2553`: ""D_LOSS": "0.32","
  - `compressed_mha/contents.md:2643`: ""D_LOSS": "0.32","
  - `direct_svd/07_results.md:38`: "| results/unified_direct_svd/distilbert/summary.csv | distilbert | direct_svd | "
  - `direct_svd/paper_full.md:413`: "| results/unified_direct_svd/distilbert/summary.csv | distilbert | direct_svd | "
  - `dobi_svd/contents.md:2905`: ""D_LOSS": "0.32","

### Number: `32.5`
- **Total occurrences:** 2
- **Modules affected:** 1
- **Complete alphabetical list of modules, with per-module count:**
  - `template_attention`: 2 occurrences
- **Context examples (10 samples):**
  - `template_attention/07_results.md:38`: "| results/unified_template_attention/gpt2/summary.csv | gpt2 | template_attentio"
  - `template_attention/paper_full.md:413`: "| results/unified_template_attention/gpt2/summary.csv | gpt2 | template_attentio"

---

## SECTION 5 — Category 4: Zero Source Citations (FULL LIST)

### Modules with ZERO source citations in 04_method.md (217 modules)

- `AdaptiveSVD`
- `AdaptiveSpanAttention`
- `AdaptiveSparseAttention`
- `Galerkin`
- `HALO`
- `OH`
- `Primal`
- `SOAFusedAttention`
- `SqueezingHeadsAttentionDistillationLayer`
- `a3`
- `aa_svd_compressor`
- `aasvd`
- `adaptive_compressor`
- `adaptive_ranks`
- `adaptive_span_attention`
- `adaptive_sparse_attention`
- `adaptive_svd_attention`
- `adasplash`
- `adasvd_compressor`
- `admm_compressor`
- `admm_pruner`
- `aegis_hybrid`
- `aegis_spectron_fp8`
- `aegis_v1`
- `aegis_v2`
- `aegis_v3`
- `aegis_v4`
- `aegis_v5`
- `aegis_v6`
- `armor_pruner`
- `ast_pruner`
- `astro_quantizer`
- `asvd`
- `bam_umoe`
- `block_transformer_hiba`
- `bonsai_pruner`
- `cachenotes_compressor`
- `chela_slay`
- `cluster`
- `come`
- `compot_compressor`
- `compressed_attention`
- `compressed_mha`
- `coopq_quantizer`
- `cosformer`
- `cr_compressor`
- `cur_compressor`
- `cur_linear`
- `cur_moe`
- `cure`
- `decay_linear_attention`
- `delta_kv`
- `direct_svd`
- `dobi`
- `dobi_svd`
- `dynamic_entropy`
- `dynamic_rank`
- `eat`
- `enlighten_compressor`
- `entropy_budget_allocation`
- `erc_svd`
- `expert_choice_moe`
- `factorised_matrix_state`
- `faq_quantizer`
- `flap`
- `flash3`
- `flash_attention3`
- `flash_attention3_fp8`
- `flat_llm`
- `frandi`
- `full_attention`
- `fused_svd_kernel`
- `fusion`
- `fusion_attn`
- `galerkin_attention`
- `gated_delta_net`
- `gated_deltanet_block`
- `gated_linear_attention`
- `gfws_d_compressor`
- `goldfinch_ahn`
- `gqa`
- `granite_hybrid`
- `grecodec`
- `griffin_ssd`
- `halo_regla`
- `hamming_attention`
- `hashed_linear`
- `hashed_net_compressor`
- `hgrn2_gated_deltanet`
- `hgrn2_layer`
- `hgrn2_outer`
- `hla_attention`
- `huffman_coder`
- `hybrid_svd_layer`
- `hybridkv`
- `hymba_transx`
- `infini`
- `jamba`
- `jamba_info`
- `kda_attention`
- `kmeans_quantize`
- `kronecker_compressor`
- `kronecker_linear`
- `kronecker_tt_lora`
- `kvtc`
- `leka_v1`
- `leka_v2`
- `leka_v3`
- `liftr`
- `lillama_compressor`
- `linformer_attention`
- `lola_attention`
- `lora_attention`
- `lsc_compressor`
- `lsh_attention`
- `mamba`
- `mamba_inject`
- `mamba_shedder`
- `mamba_svd_flash`
- `mgie_svd`
- `minimal_aurora`
- `minimal_nova`
- `minimal_prism`
- `minimal_svd`
- `minimal_svd_v2`
- `mixattention_moska`
- `mla_attention`
- `mlorq_compressor`
- `moba`
- `modegpt`
- `moe_deltakv_paged`
- `nasc_attention`
- `nexus`
- `nha`
- `nli_softmax`
- `nmf_compressor`
- `nsa`
- `nystrom_attention`
- `oats`
- `optimal_subspace_projection`
- `optimized_hybrid_attention`
- `orion`
- `padre_attention`
- `performer_attention`
- `phantom`
- `progressive_token_pruning`
- `projection_factorized_attn`
- `proxsparse`
- `proxsparse_pruner`
- `pruning_utils`
- `ptp`
- `qcqa_attention`
- `rala_attention`
- `reap_moe`
- `regla_attention`
- `replace`
- `replaceme_compressor`
- `res_svd`
- `roast_hashed_linear`
- `rotational_slicegpt_int4`
- `routing_attention`
- `rpc`
- `rpca`
- `saes_svd`
- `saes_svd_v2`
- `salr`
- `salt`
- `samba_m3`
- `samba_moe`
- `sdrm`
- `simple_ssm`
- `slicegpt`
- `slicegpt_pca`
- `slim_compressor`
- `smart_weighted_ensemble`
- `soa_attn`
- `soa_fused_module`
- `soa_module`
- `soka`
- `span_expanded_attention`
- `sparsegpt`
- `spectra_v2`
- `spectraformer_attention`
- `spectral_mamba`
- `spectron`
- `squeezing_heads`
- `squeezing_heads_distill`
- `squeezing_heads_distiller`
- `subspace_projection`
- `svd_attention`
- `svd_compressed`
- `svd_linear_attention`
- `svd_linear_decay`
- `svd_llm_v2_attention`
- `svd_mamba`
- `svd_proj_attn`
- `svd_projection_attention`
- `svd_sequence_attention`
- `swift_svd`
- `swift_svd_act`
- `template_attention`
- `tlinformer_attention`
- `token_compressed_attention`
- `tomoe`
- `tpa_attention`
- `tt`
- `unified_rope`
- `uniql_compressor`
- `upi_mamba`
- `upi_stuffed_mamba`
- `vig_attention`
- `vspec`
- `weighted_ensemble`
- `wgqa`
- `xattention`
- `zerokv_neo`
- `zeros_attention`

### Modules with ZERO citations in ANY file (71 modules)

- `AdaptiveSVD`
- `AdaptiveSpanAttention`
- `AdaptiveSparseAttention`
- `Galerkin`
- `HALO`
- `OH`
- `Primal`
- `SOAFusedAttention`
- `SqueezingHeadsAttentionDistillationLayer`
- `a3`
- `aasvd`
- `adaptive_svd_attention`
- `adasplash`
- `aegis_v5`
- `asvd`
- `bam_umoe`
- `chela_slay`
- `cluster`
- `compressed_mha`
- `cosformer`
- `dobi_svd`
- `eat`
- `expert_choice_moe`
- `flash3`
- `flash_attention3`
- `flash_attention3_fp8`
- `fusion`
- `gated_delta_net`
- `gated_deltanet_block`
- `goldfinch_ahn`
- `gqa`
- `hgrn2_gated_deltanet`
- `hgrn2_layer`
- `hymba_transx`
- `infini`
- `jamba_info`
- `leka_v1`
- `leka_v2`
- `leka_v3`
- `linformer_attention`
- `lora_attention`
- `minimal_aurora`
- `minimal_nova`
- `minimal_prism`
- `minimal_svd`
- `mixattention_moska`
- `mla_attention`
- `moba`
- `moe_deltakv_paged`
- `nasc_attention`
- `nexus`
- `nha`
- `nsa`
- `nystrom_attention`
- `orion`
- `padre_attention`
- `performer_attention`
- `phantom`
- `ptp`
- `rala_attention`
- `reap_moe`
- `res_svd`
- `routing_attention`
- `rpc`
- `salt`
- `svd_linear_attention`
- `svd_llm_v2_attention`
- `svd_mamba`
- `token_compressed_attention`
- `unified_rope`
- `vspec`

### Modules with < 3 total citations (71 modules)

- `AdaptiveSVD` (citations: 0)
- `AdaptiveSpanAttention` (citations: 0)
- `AdaptiveSparseAttention` (citations: 0)
- `Galerkin` (citations: 0)
- `HALO` (citations: 0)
- `OH` (citations: 0)
- `Primal` (citations: 0)
- `SOAFusedAttention` (citations: 0)
- `SqueezingHeadsAttentionDistillationLayer` (citations: 0)
- `a3` (citations: 0)
- `aasvd` (citations: 0)
- `adaptive_svd_attention` (citations: 0)
- `adasplash` (citations: 0)
- `aegis_v5` (citations: 0)
- `asvd` (citations: 0)
- `bam_umoe` (citations: 0)
- `chela_slay` (citations: 0)
- `cluster` (citations: 0)
- `compressed_mha` (citations: 0)
- `cosformer` (citations: 0)
- `dobi_svd` (citations: 0)
- `eat` (citations: 0)
- `expert_choice_moe` (citations: 0)
- `flash3` (citations: 0)
- `flash_attention3` (citations: 0)
- `flash_attention3_fp8` (citations: 0)
- `fusion` (citations: 0)
- `gated_delta_net` (citations: 0)
- `gated_deltanet_block` (citations: 0)
- `goldfinch_ahn` (citations: 0)
- `gqa` (citations: 0)
- `hgrn2_gated_deltanet` (citations: 0)
- `hgrn2_layer` (citations: 0)
- `hymba_transx` (citations: 0)
- `infini` (citations: 0)
- `jamba_info` (citations: 0)
- `leka_v1` (citations: 0)
- `leka_v2` (citations: 0)
- `leka_v3` (citations: 0)
- `linformer_attention` (citations: 0)
- `lora_attention` (citations: 0)
- `minimal_aurora` (citations: 0)
- `minimal_nova` (citations: 0)
- `minimal_prism` (citations: 0)
- `minimal_svd` (citations: 0)
- `mixattention_moska` (citations: 0)
- `mla_attention` (citations: 0)
- `moba` (citations: 0)
- `moe_deltakv_paged` (citations: 0)
- `nasc_attention` (citations: 0)
- `nexus` (citations: 0)
- `nha` (citations: 0)
- `nsa` (citations: 0)
- `nystrom_attention` (citations: 0)
- `orion` (citations: 0)
- `padre_attention` (citations: 0)
- `performer_attention` (citations: 0)
- `phantom` (citations: 0)
- `ptp` (citations: 0)
- `rala_attention` (citations: 0)
- `reap_moe` (citations: 0)
- `res_svd` (citations: 0)
- `routing_attention` (citations: 0)
- `rpc` (citations: 0)
- `salt` (citations: 0)
- `svd_linear_attention` (citations: 0)
- `svd_llm_v2_attention` (citations: 0)
- `svd_mamba` (citations: 0)
- `token_compressed_attention` (citations: 0)
- `unified_rope` (citations: 0)
- `vspec` (citations: 0)

---

## SECTION 6 — Category 5: Identical 04_method.md Files (SHA256 GROUPS)

Total Normalized SHA256 Groups with > 1 module: 1

### SHA256 (Normalized): `b5c235154595eb77b64ddc0a9a29a895d727ab160e0159d04af30b885c52e05a`
- **Number of modules with this exact normalized content:** 146
- **Complete alphabetical list of modules:**
  1. `aa_svd_compressor`
  2. `adaptive_compressor`
  3. `adaptive_ranks`
  4. `adaptive_span_attention`
  5. `adaptive_sparse_attention`
  6. `adasvd_compressor`
  7. `admm_compressor`
  8. `admm_pruner`
  9. `aegis_hybrid`
  10. `aegis_spectron_fp8`
  11. `aegis_v1`
  12. `aegis_v2`
  13. `aegis_v3`
  14. `aegis_v4`
  15. `aegis_v6`
  16. `armor_pruner`
  17. `ast_pruner`
  18. `astro_quantizer`
  19. `block_transformer_hiba`
  20. `bonsai_pruner`
  21. `cachenotes_compressor`
  22. `come`
  23. `compot_compressor`
  24. `compressed_attention`
  25. `coopq_quantizer`
  26. `cr_compressor`
  27. `cur_compressor`
  28. `cur_linear`
  29. `cur_moe`
  30. `cure`
  31. `decay_linear_attention`
  32. `delta_kv`
  33. `direct_svd`
  34. `dobi`
  35. `dynamic_entropy`
  36. `dynamic_rank`
  37. `enlighten_compressor`
  38. `entropy_budget_allocation`
  39. `erc_svd`
  40. `factorised_matrix_state`
  41. `faq_quantizer`
  42. `flap`
  43. `flat_llm`
  44. `frandi`
  45. `full_attention`
  46. `fused_svd_kernel`
  47. `fusion_attn`
  48. `galerkin_attention`
  49. `gated_linear_attention`
  50. `gfws_d_compressor`
  51. `granite_hybrid`
  52. `grecodec`
  53. `griffin_ssd`
  54. `halo_regla`
  55. `hamming_attention`
  56. `hashed_linear`
  57. `hashed_net_compressor`
  58. `hgrn2_outer`
  59. `hla_attention`
  60. `huffman_coder`
  61. `hybrid_svd_layer`
  62. `hybridkv`
  63. `jamba`
  64. `kda_attention`
  65. `kmeans_quantize`
  66. `kronecker_compressor`
  67. `kronecker_linear`
  68. `kronecker_tt_lora`
  69. `kvtc`
  70. `liftr`
  71. `lillama_compressor`
  72. `lola_attention`
  73. `lsc_compressor`
  74. `lsh_attention`
  75. `mamba`
  76. `mamba_inject`
  77. `mamba_shedder`
  78. `mamba_svd_flash`
  79. `mgie_svd`
  80. `minimal_svd_v2`
  81. `mlorq_compressor`
  82. `modegpt`
  83. `nli_softmax`
  84. `nmf_compressor`
  85. `oats`
  86. `optimal_subspace_projection`
  87. `optimized_hybrid_attention`
  88. `progressive_token_pruning`
  89. `projection_factorized_attn`
  90. `proxsparse`
  91. `proxsparse_pruner`
  92. `pruning_utils`
  93. `qcqa_attention`
  94. `regla_attention`
  95. `replace`
  96. `replaceme_compressor`
  97. `roast_hashed_linear`
  98. `rotational_slicegpt_int4`
  99. `rpca`
  100. `saes_svd`
  101. `saes_svd_v2`
  102. `salr`
  103. `samba_m3`
  104. `samba_moe`
  105. `sdrm`
  106. `simple_ssm`
  107. `slicegpt`
  108. `slicegpt_pca`
  109. `slim_compressor`
  110. `smart_weighted_ensemble`
  111. `soa_attn`
  112. `soa_fused_module`
  113. `soa_module`
  114. `soka`
  115. `span_expanded_attention`
  116. `sparsegpt`
  117. `spectra_v2`
  118. `spectraformer_attention`
  119. `spectral_mamba`
  120. `spectron`
  121. `squeezing_heads`
  122. `squeezing_heads_distill`
  123. `squeezing_heads_distiller`
  124. `subspace_projection`
  125. `svd_attention`
  126. `svd_compressed`
  127. `svd_linear_decay`
  128. `svd_proj_attn`
  129. `svd_projection_attention`
  130. `svd_sequence_attention`
  131. `swift_svd`
  132. `swift_svd_act`
  133. `template_attention`
  134. `tlinformer_attention`
  135. `tomoe`
  136. `tpa_attention`
  137. `tt`
  138. `uniql_compressor`
  139. `upi_mamba`
  140. `upi_stuffed_mamba`
  141. `vig_attention`
  142. `weighted_ensemble`
  143. `wgqa`
  144. `xattention`
  145. `zerokv_neo`
  146. `zeros_attention`

---

## SECTION 7 — Category 6: Python Syntax Errors (FULL TABLE)

| Module | File | Line | Error Message | Code Snippet |
|---|---|---|---|---|
| `AdaptiveSVD` | `05_architecture.md` | 2 | invalid syntax (<string>, line 2) | `@ATTENTION_REGISTRY.register("adaptive_svd")` |
| `AdaptiveSVD` | `B_implementation.md` | 4 | invalid syntax (<string>, line 4) | `for name, mod in model.named_modules():` |
| `AdaptiveSVD` | `content.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `AdaptiveSVD` | `content.md` | 4 | invalid syntax (<string>, line 4) | `import json` |
| `AdaptiveSVD` | `paper_full.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `AdaptiveSVD` | `paper_full.md` | 4 | invalid syntax (<string>, line 4) | `for name, mod in model.named_modules():` |
| `AdaptiveSpanAttention` | `05_architecture.md` | 4 | 'return' outside function (<string>, line 4) | `return None` |
| `AdaptiveSpanAttention` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `dist = (pos_q - pos_k).clamp(min=0.0)  # Only past` |
| `AdaptiveSpanAttention` | `contents.md` | 4 | 'return' outside function (<string>, line 4) | `import json` |
| `AdaptiveSpanAttention` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `AdaptiveSpanAttention` | `paper_full.md` | 4 | 'return' outside function (<string>, line 4) | `dist = (pos_q - pos_k).abs()  # Both directions` |
| `AdaptiveSpanAttention` | `paper_full.md` | 2 | invalid syntax (<string>, line 2) | `dist = (pos_q - pos_k).clamp(min=0.0)  # Only past` |
| `AdaptiveSparseAttention` | `05_architecture.md` | 3 | expected an indented block after 'for' statement on line 2 (<string>, line 3) | `class AdaptiveSparseAttention(BaseAttention): ...` |
| `AdaptiveSparseAttention` | `05_architecture.md` | 5 | 'return' outside function (<string>, line 5) | `return None` |
| `AdaptiveSparseAttention` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `if position_bias is None and self.has_relative_att` |
| `AdaptiveSparseAttention` | `contents.md` | 3 | expected an indented block after 'for' statement on line 2 (<string>, line 3) | `import argparse` |
| `AdaptiveSparseAttention` | `contents.md` | 5 | 'return' outside function (<string>, line 5) | `import re` |
| `AdaptiveSparseAttention` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `AdaptiveSparseAttention` | `paper_full.md` | 3 | expected an indented block after 'for' statement on line 2 (<string>, line 3) | `position_bias = self.compute_bias(N_q, N_k, device` |
| `AdaptiveSparseAttention` | `paper_full.md` | 5 | 'return' outside function (<string>, line 5) | `# In the chunk loop:` |
| `AdaptiveSparseAttention` | `paper_full.md` | 2 | invalid syntax (<string>, line 2) | `if position_bias is None and self.has_relative_att` |
| `Galerkin` | `08_analysis.md` | 5 | expected an indented block after 'else' statement on line 4 (<string>, line 5) | `# Galerkin path (never reached for GPT-2)` |
| `Galerkin` | `contents.md` | 5 | expected an indented block after 'else' statement on line 4 (<string>, line 5) | `python apply_vars.py --module galerkin` |
| `HALO` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `mask_4d = valid_mask.unsqueeze(1).unsqueeze(-1).to` |
| `HALO` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `OH` | `08_analysis.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `OH` | `B_implementation.md` | 5 | invalid syntax (<string>, line 5) | `# Projections` |
| `OH` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `return out * self.output_scale` |
| `OH` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `return out * self.output_scale` |
| `OH` | `B_implementation.md` | 1 | 'return' outside function (<string>, line 1) | `return out * self.output_scale` |
| `OH` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `OH` | `contents.md` | 5 | invalid syntax (<string>, line 5) | `import re` |
| `OH` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `OH` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `OH` | `contents.md` | 1 | 'return' outside function (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `Primal` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `def forward(self, hidden_states, attention_mask=No` |
| `Primal` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `Primal` | `paper_full.md` | 1 | invalid syntax (<string>, line 1) | `def forward(self, hidden_states, attention_mask=No` |
| `SOAFusedAttention` | `04_method.md` | 2 | 'return' outside function (<string>, line 2) | `k = self.k_up(self.k_down(x)).view(B, N, H_kv, d_h` |
| `SOAFusedAttention` | `05_architecture.md` | 1 | 'return' outside function (<string>, line 1) | `@ATTENTION_REGISTRY.register("soa_fused_module")` |
| `SOAFusedAttention` | `05_architecture.md` | 2 | 'return' outside function (<string>, line 2) | `@ATTENTION_REGISTRY.register("soa_fused_attn")` |
| `SOAFusedAttention` | `05_architecture.md` | 4 | 'return' outside function (<string>, line 4) | `@ATTENTION_REGISTRY.register("soa_fused_attention"` |
| `SOAFusedAttention` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `def __init__(self, in_features, out_features, lora` |
| `SOAFusedAttention` | `B_implementation.md` | 4 | 'return' outside function (<string>, line 4) | `self.weight = nn.Parameter(torch.empty(out_feature` |
| `SOAFusedAttention` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `` |
| `SOAFusedAttention` | `contents.md` | 1 | 'return' outside function (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `SOAFusedAttention` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `` |
| `SOAFusedAttention` | `contents.md` | 4 | 'return' outside function (<string>, line 4) | `import json` |
| `SOAFusedAttention` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `SOAFusedAttention` | `contents.md` | 4 | 'return' outside function (<string>, line 4) | `import json` |
| `SqueezingHeadsAttentionDistillationLayer` | `05_architecture.md` | 5 | 'return' outside function (<string>, line 5) | `return None` |
| `SqueezingHeadsAttentionDistillationLayer` | `contents.md` | 5 | 'return' outside function (<string>, line 5) | `import re` |
| `a3` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@COMPRESSOR_REGISTRY.register("a3")` |
| `a3` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@COMPRESSOR_REGISTRY.register("a3")` |
| `a3` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@COMPRESSOR_REGISTRY.register("a3")` |
| `a3` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@COMPRESSOR_REGISTRY.register("a3")` |
| `a3` | `05_architecture.md` | 2 | invalid syntax (<string>, line 2) | `@COMPRESSOR_REGISTRY.register("a3_compressor")` |
| `a3` | `05_architecture.md` | 2 | invalid syntax (<string>, line 2) | `@COMPRESSOR_REGISTRY.register("a3_compressor")` |
| `a3` | `05_architecture.md` | 3 | invalid syntax (<string>, line 3) | `def a3_compressor_factory(model, comp_config, ...)` |
| `a3` | `B_implementation.md` | 4 | invalid syntax (<string>, line 4) | `# LoRA stays trainable` |
| `a3` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `scale = scale.clamp(min=1e-3, max=1e3)` |
| `a3` | `content.md` | 1 | invalid syntax (<string>, line 1) | `# قبل: Linear(768, 768) = 589,824 پارامتر` |
| `a3` | `content.md` | 1 | invalid syntax (<string>, line 1) | `# قبل: Linear(768, 768) = 589,824 پارامتر` |
| `a3` | `content.md` | 1 | invalid syntax (<string>, line 1) | `# قبل: Linear(768, 768) = 589,824 پارامتر` |
| `a3` | `content.md` | 1 | invalid syntax (<string>, line 1) | `# قبل: Linear(768, 768) = 589,824 پارامتر` |
| `a3` | `content.md` | 2 | invalid syntax (<string>, line 2) | `# بعد: V(64, 768) + U(768, 64) = 98,304 پارامتر` |
| `a3` | `content.md` | 2 | invalid syntax (<string>, line 2) | `# بعد: V(64, 768) + U(768, 64) = 98,304 پارامتر` |
| `a3` | `content.md` | 3 | invalid syntax (<string>, line 3) | `# نسبت: 6× کاهش` |
| `a3` | `content.md` | 4 | invalid syntax (<string>, line 4) | `Usage:` |
| `a3` | `content.md` | 1 | invalid syntax (<string>, line 1) | `# قبل: Linear(768, 768) = 589,824 پارامتر` |
| `a3` | `paper_full.md` | 1 | invalid syntax (<string>, line 1) | `scale = scale.clamp(min=1e-3, max=1e3)` |
| `a3` | `paper_full.md` | 1 | invalid syntax (<string>, line 1) | `scale = scale.clamp(min=1e-3, max=1e3)` |
| `a3` | `paper_full.md` | 1 | invalid syntax (<string>, line 1) | `scale = scale.clamp(min=1e-3, max=1e3)` |
| `a3` | `paper_full.md` | 1 | invalid syntax (<string>, line 1) | `scale = scale.clamp(min=1e-3, max=1e3)` |
| `a3` | `paper_full.md` | 2 | invalid syntax (<string>, line 2) | `x_valid = x[mask]` |
| `a3` | `paper_full.md` | 2 | invalid syntax (<string>, line 2) | `x_valid = x[mask]` |
| `a3` | `paper_full.md` | 3 | invalid syntax (<string>, line 3) | `self.U.requires_grad = False` |
| `a3` | `paper_full.md` | 4 | invalid syntax (<string>, line 4) | `# LoRA stays trainable` |
| `a3` | `paper_full.md` | 1 | invalid syntax (<string>, line 1) | `scale = scale.clamp(min=1e-3, max=1e3)` |
| `aasvd` | `05_architecture.md` | 2 | invalid syntax (<string>, line 2) | `@COMPRESSOR_REGISTRY.register("aa_svd")` |
| `aasvd` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `W = raw_w.t().float()  # (in, out) → (out, in)` |
| `aasvd` | `content.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `aasvd` | `content.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `aasvd` | `paper_full.md` | 2 | invalid syntax (<string>, line 2) | `W = raw_w.t().float()  # (in, out) → (out, in)` |
| `aasvd` | `paper_full.md` | 2 | invalid syntax (<string>, line 2) | `W = raw_w.t().float()  # (in, out) → (out, in)` |
| `adaptive_svd_attention` | `05_architecture.md` | 4 | 'return' outside function (<string>, line 4) | `class AdaptiveSVDAttention(BaseAttention):` |
| `adaptive_svd_attention` | `05_architecture.md` | 1 | 'return' outside function (<string>, line 1) | `if isinstance(src_attn, AdaptiveSVDAttention): ret` |
| `adaptive_svd_attention` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `position_bias = self.compute_bias(N_q, N_k, device` |
| `adaptive_svd_attention` | `contents.md` | 4 | 'return' outside function (<string>, line 4) | `import json` |
| `adaptive_svd_attention` | `contents.md` | 1 | 'return' outside function (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `adaptive_svd_attention` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `aegis_v5` | `05_architecture.md` | 5 | 'continue' not properly in loop (<string>, line 5) | `...` |
| `aegis_v5` | `05_architecture.md` | 5 | 'return' outside function (<string>, line 5) | `...` |
| `aegis_v5` | `05_architecture.md` | 2 | 'return' outside function (<string>, line 2) | `@ATTENTION_REGISTRY.register("AegisAttention")` |
| `aegis_v5` | `05_architecture.md` | 3 | invalid character '√' (U+221A) (<string>, line 3) | `@ATTENTION_REGISTRY.register("aegis_v5")` |
| `aegis_v5` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `max_clamp=11.0, is_causal=False):` |
| `aegis_v5` | `contents.md` | 5 | 'continue' not properly in loop (<string>, line 5) | `import re` |
| `aegis_v5` | `contents.md` | 5 | 'return' outside function (<string>, line 5) | `import re` |
| `aegis_v5` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `` |
| `aegis_v5` | `contents.md` | 3 | invalid character '√' (U+221A) (<string>, line 3) | `import argparse` |
| `aegis_v5` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `asvd` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `if encoder_hidden_states is not None:` |
| `asvd` | `B_implementation.md` | 5 | 'return' outside function (<string>, line 5) | `return new_attn` |
| `asvd` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `asvd` | `contents.md` | 5 | 'return' outside function (<string>, line 5) | `import re` |
| `chela_slay` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `fused = g * dense_out + (1.0 - g) * linear_out` |
| `chela_slay` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `decay_linear_attention` | `contents.md` | 1 | unexpected indent (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `decay_linear_attention` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `dobi_svd` | `04_method.md` | 3 | invalid syntax (<string>, line 3) | `layer.gate.data.fill_(init_gate_val)` |
| `dobi_svd` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@COMPRESSOR_REGISTRY.register("dobi_svd")` |
| `dobi_svd` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@COMPRESSOR_REGISTRY.register("dobi_svd")` |
| `dobi_svd` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@COMPRESSOR_REGISTRY.register("dobi_svd")` |
| `dobi_svd` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@COMPRESSOR_REGISTRY.register("dobi_svd")` |
| `dobi_svd` | `contents.md` | 3 | invalid syntax (<string>, line 3) | `import argparse` |
| `dobi_svd` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `dobi_svd` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `dobi_svd` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `dobi_svd` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `eat` | `05_architecture.md` | 2 | invalid syntax (<string>, line 2) | `Y = model(X)` |
| `eat` | `B_implementation.md` | 4 | invalid syntax (<string>, line 4) | `with torch.no_grad():` |
| `eat` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `logger.warning(...)` |
| `eat` | `contents.md` | 4 | invalid syntax (<string>, line 4) | `import json` |
| `expert_choice_moe` | `05_architecture.md` | 6 | invalid syntax (<string>, line 6) | `self._dynamic_router = nn.Linear(D, self.n_experts` |
| `expert_choice_moe` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `logits = self.router(self.router_ln(x_flat))` |
| `expert_choice_moe` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `if D == self.d_model:` |
| `expert_choice_moe` | `B_implementation.md` | 6 | invalid syntax (<string>, line 6) | `self._dynamic_router = nn.Linear(D, self.n_experts` |
| `expert_choice_moe` | `contents.md` | 6 | invalid syntax (<string>, line 6) | `from pathlib import Path` |
| `expert_choice_moe` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `expert_choice_moe` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `expert_choice_moe` | `contents.md` | 6 | invalid syntax (<string>, line 6) | `from pathlib import Path` |
| `flash_attention3` | `04_method.md` | 7 | positional argument follows keyword argument (<string>, line 7) | `)` |
| `flash_attention3` | `04_method.md` | 3 | invalid syntax. Perhaps you forgot a comma? (<string>, line 3) | `elif (has self and output.dense):` |
| `flash_attention3` | `05_architecture.md` | 3 | invalid syntax. Perhaps you forgot a comma? (<string>, line 3) | `class FlashAttention3(BaseAttention):` |
| `flash_attention3` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `is_nested = False  # DistilBERT has out_lin → flat` |
| `flash_attention3` | `B_implementation.md` | 20 | invalid syntax. Perhaps you forgot a comma? (<string>, line 20) | `attn_mask = (attn_mask + causal_m...) if attn_mask` |
| `flash_attention3` | `B_implementation.md` | 3 | invalid syntax. Perhaps you forgot a comma? (<string>, line 3) | `elif (has self and output.dense):` |
| `flash_attention3` | `contents.md` | 7 | positional argument follows keyword argument (<string>, line 7) | `` |
| `flash_attention3` | `contents.md` | 3 | invalid syntax. Perhaps you forgot a comma? (<string>, line 3) | `import argparse` |
| `flash_attention3` | `contents.md` | 3 | invalid syntax. Perhaps you forgot a comma? (<string>, line 3) | `import argparse` |
| `flash_attention3` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `flash_attention3` | `contents.md` | 20 | invalid syntax. Perhaps you forgot a comma? (<string>, line 20) | `print(f"Loaded {len(variables)} variables\n")` |
| `flash_attention3` | `contents.md` | 3 | invalid syntax. Perhaps you forgot a comma? (<string>, line 3) | `import argparse` |
| `flash_attention3_fp8` | `04_method.md` | 1 | invalid syntax (<string>, line 1) | `class LoRALinear(nn.Module):` |
| `flash_attention3_fp8` | `04_method.md` | 1 | invalid syntax (<string>, line 1) | `class LoRALinear(nn.Module):` |
| `flash_attention3_fp8` | `04_method.md` | 2 | invalid syntax (<string>, line 2) | `def __init__(self, in_features, out_features, ...,` |
| `flash_attention3_fp8` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("flash3_fp8_inject")` |
| `flash_attention3_fp8` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("flash3_fp8_inject")` |
| `flash_attention3_fp8` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""` |
| `flash_attention3_fp8` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""` |
| `flash_attention3_fp8` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `Apply variables.json to all .md files.` |
| `flash_attention3_fp8` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""` |
| `flash_attention3_fp8` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""` |
| `fusion` | `08_analysis.md` | 2 | expected an indented block after 'if' statement on line 1 (<string>, line 2) | `# short path` |
| `fusion` | `10_limitations.md` | 1 | invalid syntax (<string>, line 1) | `except Exception:` |
| `fusion` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `except Exception:` |
| `fusion` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `except Exception:` |
| `fusion` | `checklist.md` | 1 | invalid syntax (<string>, line 1) | `except Exception:` |
| `fusion` | `contents.md` | 2 | expected an indented block after 'if' statement on line 1 (<string>, line 2) | `raise RuntimeError("copy_weights failed")` |
| `fusion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception:` |
| `fusion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception:` |
| `fusion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception:` |
| `fusion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception:` |
| `fusion` | `paper_full.md` | 2 | expected an indented block after 'if' statement on line 1 (<string>, line 2) | `return None` |
| `fusion` | `paper_full.md` | 1 | invalid syntax (<string>, line 1) | `except Exception:` |
| `fusion` | `paper_full.md` | 1 | invalid syntax (<string>, line 1) | `except Exception:` |
| `fusion` | `paper_full.md` | 1 | invalid syntax (<string>, line 1) | `except Exception:` |
| `gated_delta_net` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("gated_delta_net")` |
| `gated_delta_net` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("gated_delta_net")` |
| `gated_delta_net` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `def inject_gated_delta_net(layer, model_config, co` |
| `gated_delta_net` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("gated_delta_net_inje` |
| `gated_delta_net` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("gated_delta_net_inje` |
| `gated_delta_net` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `def inject_gated_delta_net(layer, model_config, co` |
| `gated_deltanet_block` | `03_background.md` | 2 | invalid syntax (<string>, line 2) | `...critical operation...` |
| `gated_deltanet_block` | `04_method.md` | 3 | 'return' outside function (<string>, line 3) | `w_q, b_q = adapter.get_weight_bias(old_attn, "q")` |
| `gated_deltanet_block` | `04_method.md` | 2 | 'return' outside function (<string>, line 2) | `adapter = UniversalAdapter()` |
| `gated_deltanet_block` | `05_architecture.md` | 2 | 'return' outside function (<string>, line 2) | `logger.warning(f"Q/K/V shape mismatch: {qkv_w.shap` |
| `gated_deltanet_block` | `05_architecture.md` | 2 | invalid syntax (<string>, line 2) | `logger.warning(f"Q/K/V shape mismatch: {qkv_w.shap` |
| `gated_deltanet_block` | `B_implementation.md` | 2 | 'return' outside function (<string>, line 2) | `for name, mod in model.named_modules():` |
| `gated_deltanet_block` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `def diagnose_block(model):` |
| `gated_deltanet_block` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `for name, mod in model.named_modules():` |
| `gated_deltanet_block` | `contents.md` | 3 | 'return' outside function (<string>, line 3) | `if isinstance(mod, GatedDeltaNetBlock):` |
| `gated_deltanet_block` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `for name, mod in model.named_modules():` |
| `gated_deltanet_block` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `for name, mod in model.named_modules():` |
| `gated_deltanet_block` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `for name, mod in model.named_modules():` |
| `gated_deltanet_block` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `for name, mod in model.named_modules():` |
| `goldfinch_ahn` | `B_implementation.md` | 12 | expected an indented block after function definition on line 10 (<string>, line 12) | `# ...` |
| `goldfinch_ahn` | `checklist.md` | 1 | invalid character '≈' (U+2248) (<string>, line 1) | `alpha = torch.sigmoid(2.0) ≈ 0.88` |
| `goldfinch_ahn` | `contents.md` | 12 | expected an indented block after function definition on line 10 (<string>, line 12) | `# ...` |
| `goldfinch_ahn` | `contents.md` | 1 | invalid character '≈' (U+2248) (<string>, line 1) | `alpha = torch.sigmoid(2.0) ≈ 0.88` |
| `gqa` | `05_architecture.md` | 2 | positional argument follows keyword argument (<string>, line 2) | `@ATTENTION_REGISTRY.register("grouped_query_attent` |
| `gqa` | `05_architecture.md` | 7 | invalid syntax (<string>, line 7) | `def inject_gqa(layer, model_config, ...):` |
| `gqa` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `# WRONG` |
| `gqa` | `content.md` | 2 | positional argument follows keyword argument (<string>, line 2) | `` |
| `gqa` | `content.md` | 7 | invalid syntax (<string>, line 7) | `` |
| `gqa` | `content.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `gqa` | `paper_full.md` | 2 | positional argument follows keyword argument (<string>, line 2) | `w = src.c_attn.weight  # (in, out) — no transpose` |
| `gqa` | `paper_full.md` | 7 | invalid syntax (<string>, line 7) | `` |
| `gqa` | `paper_full.md` | 1 | invalid syntax (<string>, line 1) | `# WRONG` |
| `hgrn2_gated_deltanet` | `05_architecture.md` | 1 | 'return' outside function (<string>, line 1) | `if "DeltaNet" in c_name or "LinearAttention" in c_` |
| `hgrn2_gated_deltanet` | `05_architecture.md` | 2 | 'return' outside function (<string>, line 2) | `return None` |
| `hgrn2_gated_deltanet` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `# 1. Branch forwards` |
| `hgrn2_gated_deltanet` | `B_implementation.md` | 1 | 'return' outside function (<string>, line 1) | `def forward(self, hidden_states, attention_mask=No` |
| `hgrn2_gated_deltanet` | `B_implementation.md` | 2 | 'return' outside function (<string>, line 2) | `# 1. Branch forwards` |
| `hgrn2_gated_deltanet` | `B_implementation.md` | 6 | 'return' outside function (<string>, line 6) | `# 2. Shape align` |
| `hgrn2_gated_deltanet` | `contents.md` | 1 | 'return' outside function (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `hgrn2_gated_deltanet` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `` |
| `hgrn2_gated_deltanet` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `hgrn2_gated_deltanet` | `contents.md` | 1 | 'return' outside function (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `hgrn2_gated_deltanet` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `` |
| `hgrn2_gated_deltanet` | `contents.md` | 6 | 'return' outside function (<string>, line 6) | `from pathlib import Path` |
| `hymba_transx` | `05_architecture.md` | 6 | invalid syntax (<string>, line 6) | `def inject_hymba_transx(...):` |
| `hymba_transx` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `assert not torch.isnan(self.W_q1.weight).any()` |
| `hymba_transx` | `contents.md` | 6 | invalid syntax (<string>, line 6) | `from pathlib import Path` |
| `hymba_transx` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `infini` | `05_architecture.md` | 3 | invalid syntax (<string>, line 3) | `class InfiniAttention(BaseAttention):` |
| `infini` | `contents.md` | 3 | invalid syntax (<string>, line 3) | `import argparse` |
| `jamba_info` | `05_architecture.md` | 6 | invalid syntax (<string>, line 6) | `def inject_jamba_info(...):` |
| `jamba_info` | `contents.md` | 6 | invalid syntax (<string>, line 6) | `from pathlib import Path` |
| `leka_v1` | `05_architecture.md` | 6 | invalid syntax (<string>, line 6) | `def inject_leka_v1(...):` |
| `leka_v1` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `# After init` |
| `leka_v1` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `# After init` |
| `leka_v1` | `contents.md` | 6 | invalid syntax (<string>, line 6) | `from pathlib import Path` |
| `leka_v1` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `leka_v1` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `leka_v2` | `05_architecture.md` | 1 | 'return' outside function (<string>, line 1) | `@ATTENTION_REGISTRY.register("leka_v2")` |
| `leka_v2` | `contents.md` | 1 | 'return' outside function (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `leka_v3` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `if self.is_decoder:` |
| `leka_v3` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `lora_attention` | `03_background.md` | 1 | invalid decimal literal (<string>, line 1) | `q_lin.weight.shape = (d, d)  # standard Linear` |
| `lora_attention` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `def merge_lora(model):` |
| `lora_attention` | `B_implementation.md` | 30 | ':' expected after dictionary key (<string>, line 30) | `named_map = {"q": ["q_proj", "q_lin", "query"], ..` |
| `lora_attention` | `contents.md` | 1 | invalid decimal literal (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `lora_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `lora_attention` | `contents.md` | 30 | ':' expected after dictionary key (<string>, line 30) | `missing.add(key)` |
| `minimal_aurora` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("minimal_aurora", sup` |
| `minimal_aurora` | `05_architecture.md` | 2 | positional argument follows keyword argument (<string>, line 2) | `class MinimalAURORAAttention(BaseAttention):` |
| `minimal_aurora` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `output = self.out_proj(context) * self.output_scal` |
| `minimal_aurora` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `output = self.out_proj(context) * self.output_scal` |
| `minimal_aurora` | `contents.md` | 2 | positional argument follows keyword argument (<string>, line 2) | `eye_w = torch.eye(self.hidden_size, self.all_head_` |
| `minimal_aurora` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `output = self.out_proj(context) * self.output_scal` |
| `minimal_nova` | `03_background.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `def __init__(self, nf, nx):` |
| `minimal_nova` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `class MinimalNOVAAttention(BaseAttention):` |
| `minimal_nova` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `class MinimalNOVAAttention(BaseAttention):` |
| `minimal_nova` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `class MinimalNOVAAttention(BaseAttention):` |
| `minimal_nova` | `05_architecture.md` | 1 | unexpected indent (<string>, line 1) | `@ATTENTION_REGISTRY.register("minimal_nova")` |
| `minimal_nova` | `05_architecture.md` | 1 | unexpected indent (<string>, line 1) | `@ATTENTION_REGISTRY.register("minimal_nova")` |
| `minimal_nova` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `class MinimalNOVAAttention(BaseAttention):` |
| `minimal_nova` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("minimal_nova")` |
| `minimal_nova` | `05_architecture.md` | 7 | invalid syntax (<string>, line 7) | `def inject_minimal_nova(...):` |
| `minimal_nova` | `B_implementation.md` | 3 | invalid syntax (<string>, line 3) | `` |
| `minimal_nova` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `# WRONG` |
| `minimal_nova` | `B_implementation.md` | 18 | expected an indented block after 'if' statement on line 14 (<string>, line 18) | `k_w = weight[split_size:2*split_size, :]` |
| `minimal_nova` | `B_implementation.md` | 3 | 'return' outside function (<string>, line 3) | `` |
| `minimal_nova` | `checklist.md` | 1 | invalid character '²' (U+00B2) (<string>, line 1) | `cum_energy = cumsum(S²) / sum(S²)` |
| `minimal_nova` | `contents.md` | 7 | invalid character '→' (U+2192) (<string>, line 7) | `` |
| `minimal_nova` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `r_eff = min(r_max, first_index_where_energy >= η)` |
| `minimal_nova` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `r_eff = min(r_max, first_index_where_energy >= η)` |
| `minimal_nova` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `r_eff = min(r_max, first_index_where_energy >= η)` |
| `minimal_nova` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `r_eff = min(r_max, first_index_where_energy >= η)` |
| `minimal_nova` | `contents.md` | 1 | unexpected indent (<string>, line 1) | `cum_energy = cumsum(S²) / sum(S²)` |
| `minimal_nova` | `contents.md` | 1 | unexpected indent (<string>, line 1) | `cum_energy = cumsum(S²) / sum(S²)` |
| `minimal_nova` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `r_eff = min(r_max, first_index_where_energy >= η)` |
| `minimal_nova` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `cum_energy = cumsum(S²) / sum(S²)` |
| `minimal_nova` | `contents.md` | 7 | invalid syntax (<string>, line 7) | `` |
| `minimal_nova` | `contents.md` | 3 | invalid syntax (<string>, line 3) | `import argparse` |
| `minimal_nova` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `cum_energy = cumsum(S²) / sum(S²)` |
| `minimal_nova` | `contents.md` | 18 | expected an indented block after 'if' statement on line 14 (<string>, line 18) | `variables = {k: v for k, v in variables.items() if` |
| `minimal_nova` | `contents.md` | 3 | 'return' outside function (<string>, line 3) | `import argparse` |
| `minimal_nova` | `contents.md` | 1 | invalid character '²' (U+00B2) (<string>, line 1) | `cum_energy = cumsum(S²) / sum(S²)` |
| `minimal_prism` | `03_background.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `def __init__(self, nf, nx):` |
| `minimal_prism` | `04_method.md` | 7 | 'return' outside function (<string>, line 7) | `self.lora_A.requires_grad = True` |
| `minimal_prism` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `class MinimalPRISMAttention(BaseAttention):` |
| `minimal_prism` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `class MinimalPRISMAttention(BaseAttention):` |
| `minimal_prism` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `class MinimalPRISMAttention(BaseAttention):` |
| `minimal_prism` | `05_architecture.md` | 1 | unexpected indent (<string>, line 1) | `@ATTENTION_REGISTRY.register("minimal_prism")` |
| `minimal_prism` | `05_architecture.md` | 1 | unexpected indent (<string>, line 1) | `@ATTENTION_REGISTRY.register("minimal_prism")` |
| `minimal_prism` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `class MinimalPRISMAttention(BaseAttention):` |
| `minimal_prism` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("minimal_prism")` |
| `minimal_prism` | `05_architecture.md` | 7 | invalid syntax (<string>, line 7) | `def inject_minimal_prism(...):` |
| `minimal_prism` | `B_implementation.md` | 3 | invalid syntax (<string>, line 3) | `return None` |
| `minimal_prism` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `if isinstance(src_attn, MinimalPRISMAttention) or ` |
| `minimal_prism` | `B_implementation.md` | 8 | 'return' outside function (<string>, line 8) | `return None` |
| `minimal_prism` | `B_implementation.md` | 3 | 'return' outside function (<string>, line 3) | `return None` |
| `minimal_prism` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `minimal_prism` | `contents.md` | 7 | 'return' outside function (<string>, line 7) | `` |
| `minimal_prism` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `minimal_prism` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `minimal_prism` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `minimal_prism` | `contents.md` | 1 | unexpected indent (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `minimal_prism` | `contents.md` | 1 | unexpected indent (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `minimal_prism` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `minimal_prism` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `minimal_prism` | `contents.md` | 7 | invalid syntax (<string>, line 7) | `` |
| `minimal_prism` | `contents.md` | 3 | invalid syntax (<string>, line 3) | `import argparse` |
| `minimal_prism` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `minimal_prism` | `contents.md` | 8 | 'return' outside function (<string>, line 8) | `PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")` |
| `minimal_prism` | `contents.md` | 3 | 'return' outside function (<string>, line 3) | `import argparse` |
| `minimal_svd` | `03_background.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `def __init__(self, nf, nx):` |
| `minimal_svd` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `class MinimalSVDAttention(BaseAttention):` |
| `minimal_svd` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `class MinimalSVDAttention(BaseAttention):` |
| `minimal_svd` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `class MinimalSVDAttention(BaseAttention):` |
| `minimal_svd` | `05_architecture.md` | 1 | unexpected indent (<string>, line 1) | `@ATTENTION_REGISTRY.register("minimal_svd")` |
| `minimal_svd` | `05_architecture.md` | 1 | unexpected indent (<string>, line 1) | `@ATTENTION_REGISTRY.register("minimal_svd")` |
| `minimal_svd` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `class MinimalSVDAttention(BaseAttention):` |
| `minimal_svd` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("minimal_svd")` |
| `minimal_svd` | `05_architecture.md` | 7 | invalid syntax (<string>, line 7) | `def inject_minimal_svd(...):` |
| `minimal_svd` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `inject_minimal_svd(layer, model.config, energy=0.9` |
| `minimal_svd` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `inject_minimal_svd(layer, model.config, energy=0.9` |
| `minimal_svd` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `inject_minimal_svd(layer, model.config, energy=0.9` |
| `minimal_svd` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `inject_minimal_svd(layer, model.config, energy=0.9` |
| `minimal_svd` | `contents.md` | 1 | unexpected indent (<string>, line 1) | `for layer in model.distilbert.transformer.layer:` |
| `minimal_svd` | `contents.md` | 1 | unexpected indent (<string>, line 1) | `for layer in model.distilbert.transformer.layer:` |
| `minimal_svd` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `inject_minimal_svd(layer, model.config, energy=0.9` |
| `minimal_svd` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `for layer in model.distilbert.transformer.layer:` |
| `minimal_svd` | `contents.md` | 7 | invalid syntax (<string>, line 7) | `` |
| `mixattention_moska` | `08_analysis.md` | 3 | expected an indented block after 'if' statement on line 2 (<string>, line 3) | `# Standard attention` |
| `mixattention_moska` | `checklist.md` | 2 | expected an indented block after 'if' statement on line 1 (<string>, line 2) | `# Standard full attention — NO pruning!` |
| `mixattention_moska` | `contents.md` | 3 | expected an indented block after 'if' statement on line 2 (<string>, line 3) | `@ATTENTION_REGISTRY.register("mixattention_moska")` |
| `mixattention_moska` | `contents.md` | 2 | expected an indented block after 'if' statement on line 1 (<string>, line 2) | `# Standard full attention — NO pruning!` |
| `mla_attention` | `04_method.md` | 1 | positional argument follows keyword argument (<string>, line 1) | `self.q_proj = LoRALinear(..., lora_rank=16, ...)` |
| `mla_attention` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("mla_attention")` |
| `mla_attention` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("mla_attention")` |
| `mla_attention` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("mla_attention")` |
| `mla_attention` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("mla_attention")` |
| `mla_attention` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("mla")` |
| `mla_attention` | `05_architecture.md` | 7 | invalid syntax (<string>, line 7) | `def inject_mla(...):` |
| `mla_attention` | `contents.md` | 1 | positional argument follows keyword argument (<string>, line 1) | `from huggingface_hub import hf_hub_download` |
| `mla_attention` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `for model in ["distilbert", "roberta", "gpt2"]:` |
| `mla_attention` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `for model in ["distilbert", "roberta", "gpt2"]:` |
| `mla_attention` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `for model in ["distilbert", "roberta", "gpt2"]:` |
| `mla_attention` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `for model in ["distilbert", "roberta", "gpt2"]:` |
| `mla_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `from huggingface_hub import hf_hub_download` |
| `mla_attention` | `contents.md` | 7 | invalid syntax (<string>, line 7) | `print(f"Downloaded {model}: {path}")` |
| `moba` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("mamba")` |
| `moba` | `05_architecture.md` | 2 | 'return' outside function (<string>, line 2) | `class Mamba(BaseAttention):` |
| `moba` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `# in_proj branch 1 (SSM input): from V` |
| `moba` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `moba` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `moba` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `` |
| `moba` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `moe_deltakv_paged` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `nasc_attention` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `if not hasattr(self, '_logged_once'):` |
| `nasc_attention` | `10_limitations.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `nasc_attention` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `def measure_rff_memory(B, H, L, F, d):` |
| `nasc_attention` | `D_negative_results.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `nasc_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `nasc_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `nasc_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `nasc_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `nasc_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `nexus` | `04_method.md` | 9 | 'return' outside function (<string>, line 9) | `return ctx / (norm + 1e-5)` |
| `nexus` | `contents.md` | 2 | expected an indented block after 'if' statement on line 1 (<string>, line 2) | `def __init__(self, head_dim, num_heads, gate_rank,` |
| `nexus` | `contents.md` | 1 | expected an indented block after 'if' statement on line 1 (<string>, line 1) | `class SeerGating(nn.Module):` |
| `nexus` | `contents.md` | 9 | 'return' outside function (<string>, line 9) | `` |
| `nha` | `05_architecture.md` | 5 | 'return' outside function (<string>, line 5) | `return None` |
| `nha` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `B, H, N, d = Q.shape` |
| `nha` | `contents.md` | 5 | 'return' outside function (<string>, line 5) | `import re` |
| `nha` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `nsa` | `05_architecture.md` | 5 | 'return' outside function (<string>, line 5) | `return None` |
| `nsa` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `non_local_scores = scores.masked_fill(local_mask_4` |
| `nsa` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `# 1. Mask local window + padding` |
| `nsa` | `contents.md` | 5 | 'return' outside function (<string>, line 5) | `import re` |
| `nsa` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `nsa` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `nystrom_attention` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("nystrom_attention")` |
| `nystrom_attention` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("nystrom_attention")` |
| `nystrom_attention` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("nystrom_attention")` |
| `nystrom_attention` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("nystrom_attention")` |
| `nystrom_attention` | `05_architecture.md` | 5 | 'return' outside function (<string>, line 5) | `` |
| `nystrom_attention` | `05_architecture.md` | 2 | invalid syntax (<string>, line 2) | `@ATTENTION_REGISTRY.register("nystrom_attention")` |
| `nystrom_attention` | `05_architecture.md` | 8 | invalid syntax (<string>, line 8) | `def inject_nystrom(...):` |
| `nystrom_attention` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `def inject_nystrom(layer, model_config, comp_confi` |
| `nystrom_attention` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `src_attn = old_attn or layer.attention...` |
| `nystrom_attention` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `nystrom_attention` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `nystrom_attention` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `nystrom_attention` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `nystrom_attention` | `contents.md` | 5 | 'return' outside function (<string>, line 5) | `import re` |
| `nystrom_attention` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `nystrom_attention` | `contents.md` | 8 | invalid syntax (<string>, line 8) | `PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")` |
| `nystrom_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `nystrom_attention` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `orion` | `01_intro.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `orion` | `04_method.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `orion` | `08_analysis.md` | 1 | invalid syntax (<string>, line 1) | `denom = p.sum(dim=dim, keepdim=True)` |
| `orion` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `@ATTENTION_REGISTRY.register("orion_attention")` |
| `orion` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("orion")` |
| `orion` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("orion")` |
| `orion` | `D_negative_results.md` | 1 | invalid syntax (<string>, line 1) | `p = F.relu(torch.exp(x_f) - 1.0)` |
| `orion` | `D_negative_results.md` | 2 | 'return' outside function (<string>, line 2) | `return p / (p.sum(dim=dim, keepdim=True) + 1e-8)` |
| `orion` | `checklist.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `orion` | `checklist.md` | 2 | 'return' outside function (<string>, line 2) | `logger.error("inject_orion failed: %s", e)` |
| `orion` | `checklist.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `orion` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `logger.error("inject_orion failed: %s", e)` |
| `orion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `orion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `orion` | `contents.md` | 1 | expected an indented block after 'for' statement on line 1 (<string>, line 1) | `except Exception as e:` |
| `orion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `orion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `orion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `orion` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `logger.error("inject_orion failed: %s", e)` |
| `orion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `orion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `orion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `orion` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `logger.error("inject_orion failed: %s", e)` |
| `orion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `orion` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `logger.error("inject_orion failed: %s", e)` |
| `orion` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `padre_attention` | `05_architecture.md` | 8 | 'return' outside function (<string>, line 8) | `return None` |
| `padre_attention` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `def forward(self, hidden_states, attention_mask=No` |
| `padre_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `def forward(self, hidden_states, attention_mask=No` |
| `padre_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `def forward(self, hidden_states, attention_mask=No` |
| `padre_attention` | `contents.md` | 8 | 'return' outside function (<string>, line 8) | `if position_embeddings is not None:` |
| `padre_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `def forward(self, hidden_states, attention_mask=No` |
| `performer_attention` | `01_intro.md` | 1 | invalid syntax (<string>, line 1) | `class LoRALinear(nn.Module):` |
| `performer_attention` | `01_intro.md` | 2 | invalid syntax (<string>, line 2) | `def __init__(self, ...):` |
| `performer_attention` | `04_method.md` | 1 | invalid syntax (<string>, line 1) | `self.output_scale = nn.Parameter(torch.ones(1))` |
| `performer_attention` | `04_method.md` | 3 | 'return' outside function (<string>, line 3) | `return out * self.output_scale` |
| `performer_attention` | `05_architecture.md` | 2 | 'return' outside function (<string>, line 2) | `@ATTENTION_REGISTRY.register("performer_attention"` |
| `performer_attention` | `08_analysis.md` | 1 | invalid syntax (<string>, line 1) | `return out * self.output_scale` |
| `performer_attention` | `08_analysis.md` | 1 | 'return' outside function (<string>, line 1) | `return out * self.output_scale` |
| `performer_attention` | `B_implementation.md` | 9 | expected an indented block after 'for' statement on line 8 (<string>, line 9) | `out = out + F.linear(F.linear(x, self.lora_A), sel` |
| `performer_attention` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("performer")` |
| `performer_attention` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `@ATTENTION_REGISTRY.register("performer_attention"` |
| `performer_attention` | `B_implementation.md` | 2 | 'return' outside function (<string>, line 2) | `@ATTENTION_REGISTRY.register("performer_attention"` |
| `performer_attention` | `checklist.md` | 1 | invalid syntax (<string>, line 1) | `if encoder_hidden_states is not None:` |
| `performer_attention` | `checklist.md` | 2 | 'return' outside function (<string>, line 2) | `return hidden_states, None` |
| `performer_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `if encoder_hidden_states is not None:` |
| `performer_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `if encoder_hidden_states is not None:` |
| `performer_attention` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `return hidden_states, None` |
| `performer_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `if encoder_hidden_states is not None:` |
| `performer_attention` | `contents.md` | 3 | 'return' outside function (<string>, line 3) | `raise RuntimeError(f"Injection failed")  # ← BUG` |
| `performer_attention` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `return hidden_states, None` |
| `performer_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `if encoder_hidden_states is not None:` |
| `performer_attention` | `contents.md` | 1 | 'return' outside function (<string>, line 1) | `if encoder_hidden_states is not None:` |
| `performer_attention` | `contents.md` | 9 | expected an indented block after 'for' statement on line 8 (<string>, line 9) | `out = out + F.linear(F.linear(x, self.lora_A), sel` |
| `performer_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `if encoder_hidden_states is not None:` |
| `performer_attention` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `return hidden_states, None` |
| `performer_attention` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `return hidden_states, None` |
| `performer_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `if encoder_hidden_states is not None:` |
| `performer_attention` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `return hidden_states, None` |
| `phantom` | `05_architecture.md` | 2 | invalid syntax (<string>, line 2) | `Y = model(X)` |
| `phantom` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `X = torch.randn(1, N, d, requires_grad=True)` |
| `phantom` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `if "attention" in name and "self" in name:` |
| `phantom` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `for name, mod in model.named_modules():` |
| `phantom` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `phantom` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `phantom` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `phantom` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `ptp` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `attn_scores = attn_scores.masked_fill(` |
| `ptp` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `k = repeat_kv(k, num_heads // num_kv_heads)` |
| `rala_attention` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `if "DeltaNet" in c_name or "LinearAttention" in c_` |
| `rala_attention` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `if "DeltaNet" in c_name or "LinearAttention" in c_` |
| `rala_attention` | `05_architecture.md` | 2 | 'return' outside function (<string>, line 2) | `return None` |
| `rala_attention` | `05_architecture.md` | 2 | 'return' outside function (<string>, line 2) | `return None` |
| `rala_attention` | `05_architecture.md` | 2 | 'return' outside function (<string>, line 2) | `return None` |
| `rala_attention` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `# Check injection` |
| `rala_attention` | `B_implementation.md` | 5 | 'return' outside function (<string>, line 5) | `` |
| `rala_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""` |
| `rala_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""` |
| `rala_attention` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `Apply variables.json to all .md files.` |
| `rala_attention` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `Apply variables.json to all .md files.` |
| `rala_attention` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `Apply variables.json to all .md files.` |
| `rala_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""` |
| `rala_attention` | `contents.md` | 5 | 'return' outside function (<string>, line 5) | `python apply_vars.py --module rala` |
| `res_svd` | `05_architecture.md` | 2 | 'return' outside function (<string>, line 2) | `raise NotImplementedError(...)` |
| `res_svd` | `05_architecture.md` | 14 | invalid syntax (<string>, line 14) | `def from_linear(cls, linear, base_rank, residual_r` |
| `res_svd` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `if encoder_hidden_states is not None:` |
| `res_svd` | `08_analysis.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `res_svd` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `temp_linear = nn.Linear(target.in_features, target` |
| `res_svd` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `temp_linear = nn.Linear(target.in_features, target` |
| `res_svd` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `` |
| `res_svd` | `contents.md` | 14 | invalid syntax (<string>, line 14) | `print(f"Not found: {vars_file}")` |
| `res_svd` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `res_svd` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `res_svd` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `res_svd` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `routing_attention` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `# Verify injection` |
| `routing_attention` | `05_architecture.md` | 2 | 'return' outside function (<string>, line 2) | `for name, mod in model.named_modules():` |
| `routing_attention` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `# Check cluster assignment` |
| `routing_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `SECTIONS = [` |
| `routing_attention` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `"00_meta.md", "01_intro.md", "02_related.md", "03_` |
| `routing_attention` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `SECTIONS = [` |
| `rpc` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("rpc")` |
| `rpc` | `B_implementation.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `rpc` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `rpc` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `salt` | `08_analysis.md` | 1 | invalid syntax (<string>, line 1) | `if norm < 1e-4:` |
| `salt` | `10_limitations.md` | 5 | 'return' outside function (<string>, line 5) | `return y_svd + y_lora + self.bias` |
| `salt` | `10_limitations.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `salt` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `z = F.linear(x, self.Vt)` |
| `salt` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `z = F.linear(x, self.Vt)` |
| `salt` | `contents.md` | 5 | 'return' outside function (<string>, line 5) | `"11_conclusion.md", "12_reproducibility.md", "13_r` |
| `salt` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `z = F.linear(x, self.Vt)` |
| `svd_linear_attention` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `if x_t.dim() == 2:` |
| `svd_linear_attention` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `svd_llm_v2_attention` | `03_background.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `├── self       (BertSelfAttention)` |
| `svd_llm_v2_attention` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("svd_llm_v2_attention` |
| `svd_llm_v2_attention` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("svd_llm_v2_attention` |
| `svd_llm_v2_attention` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("svd_llm_v2_attention` |
| `svd_llm_v2_attention` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("svd_llm_v2_attention` |
| `svd_llm_v2_attention` | `05_architecture.md` | 3 | positional argument follows keyword argument (<string>, line 3) | `class SVDLLMV2Attention(BaseAttention):` |
| `svd_llm_v2_attention` | `05_architecture.md` | 2 | 'return' outside function (<string>, line 2) | `@ATTENTION_REGISTRY.register("svd_llm_v2_attention` |
| `svd_llm_v2_attention` | `05_architecture.md` | 8 | invalid syntax (<string>, line 8) | `def inject_svd_llm_v2(...):` |
| `svd_llm_v2_attention` | `B_implementation.md` | 27 | expected an indented block after 'elif' statement on line 23 (<string>, line 27) | `` |
| `svd_llm_v2_attention` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `nn.init.zeros_(p_b)` |
| `svd_llm_v2_attention` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `w_k_pooled = w_k.view(self.num_kv_heads, rep, self` |
| `svd_llm_v2_attention` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `w_k_pooled = w_k.view(self.num_kv_heads, rep, self` |
| `svd_llm_v2_attention` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `w_k_pooled = w_k.view(self.num_kv_heads, rep, self` |
| `svd_llm_v2_attention` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `w_k_pooled = w_k.view(self.num_kv_heads, rep, self` |
| `svd_llm_v2_attention` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `w_k_pooled = w_k.view(self.num_kv_heads, rep, self` |
| `svd_llm_v2_attention` | `contents.md` | 3 | positional argument follows keyword argument (<string>, line 3) | `import argparse` |
| `svd_llm_v2_attention` | `contents.md` | 2 | 'return' outside function (<string>, line 2) | `w_k_pooled = w_k.view(self.num_kv_heads, rep, self` |
| `svd_llm_v2_attention` | `contents.md` | 8 | invalid syntax (<string>, line 8) | `PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")` |
| `svd_llm_v2_attention` | `contents.md` | 27 | expected an indented block after 'elif' statement on line 23 (<string>, line 27) | `key = match.group(1)` |
| `svd_llm_v2_attention` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `w_k_pooled = w_k.view(self.num_kv_heads, rep, self` |
| `svd_mamba` | `05_architecture.md` | 1 | invalid syntax (<string>, line 1) | `@ATTENTION_REGISTRY.register("svd_mamba")` |
| `svd_mamba` | `08_analysis.md` | 1 | invalid syntax (<string>, line 1) | `except Exception as e:` |
| `svd_mamba` | `B_implementation.md` | 4 | invalid syntax (<string>, line 4) | `x_pad = F.pad(x_t, (self.d_conv - 1, 0))` |
| `svd_mamba` | `B_implementation.md` | 2 | invalid syntax (<string>, line 2) | `Y = gamma * ssm_proj + (1.0 - gamma) * attn_out` |
| `svd_mamba` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `svd_mamba` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `svd_mamba` | `contents.md` | 4 | invalid syntax (<string>, line 4) | `import json` |
| `svd_mamba` | `contents.md` | 2 | invalid syntax (<string>, line 2) | `` |
| `unified_rope` | `03_background.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `t = arange(seq_len)` |
| `unified_rope` | `04_method.md` | 1 | invalid syntax (<string>, line 1) | `if self.is_nested_encoder:` |
| `unified_rope` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("unified_rope")` |
| `unified_rope` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("unified_rope")` |
| `unified_rope` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("unified_rope")` |
| `unified_rope` | `05_architecture.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `@ATTENTION_REGISTRY.register("unified_rope")` |
| `unified_rope` | `05_architecture.md` | 10 | invalid syntax (<string>, line 10) | `def inject_unified_rope(...):` |
| `unified_rope` | `B_implementation.md` | 3 | invalid syntax (<string>, line 3) | `try:` |
| `unified_rope` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `unified_rope` | `contents.md` | 1 | invalid syntax (<string>, line 1) | `"""Apply variables.json to all .md files."""` |
| `unified_rope` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `unified_rope` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `unified_rope` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `unified_rope` | `contents.md` | 2 | invalid character '├' (U+251C) (<string>, line 2) | `` |
| `unified_rope` | `contents.md` | 10 | invalid syntax (<string>, line 10) | `` |
| `unified_rope` | `contents.md` | 3 | invalid syntax (<string>, line 3) | `import argparse` |
| `vspec` | `05_architecture.md` | 1 | 'return' outside function (<string>, line 1) | `@ATTENTION_REGISTRY.register("vspec")` |
| `vspec` | `contents.md` | 1 | 'return' outside function (<string>, line 1) | `"""Apply variables.json to all .md files."""` |

Total: 543 syntax errors across 62 modules.

---

## SECTION 8 — Category 7: Cross-Module Overlap (FULL MATRIX)

| Module A | Module B | Common Lines | Total Lines | Overlap % |
|---|---|---|---|---|
| `adaptive_span_attention` | `adaptive_sparse_attention` | 126 | 127 | 99.21% |
| `aa_svd_compressor` | `adaptive_compressor` | 125 | 127 | 98.43% |
| `aa_svd_compressor` | `adaptive_ranks` | 125 | 127 | 98.43% |
| `aa_svd_compressor` | `adaptive_span_attention` | 125 | 127 | 98.43% |
| `aa_svd_compressor` | `adaptive_sparse_attention` | 125 | 127 | 98.43% |
| `aa_svd_compressor` | `adasvd_compressor` | 125 | 127 | 98.43% |
| `aa_svd_compressor` | `admm_compressor` | 125 | 127 | 98.43% |
| `adaptive_compressor` | `adaptive_ranks` | 125 | 127 | 98.43% |
| `adaptive_compressor` | `adaptive_span_attention` | 125 | 127 | 98.43% |
| `adaptive_compressor` | `adaptive_sparse_attention` | 125 | 127 | 98.43% |
| `adaptive_compressor` | `adasvd_compressor` | 125 | 127 | 98.43% |
| `adaptive_compressor` | `admm_compressor` | 125 | 127 | 98.43% |
| `adaptive_ranks` | `adaptive_span_attention` | 125 | 127 | 98.43% |
| `adaptive_ranks` | `adaptive_sparse_attention` | 125 | 127 | 98.43% |
| `adaptive_ranks` | `adasvd_compressor` | 125 | 127 | 98.43% |
| `adaptive_ranks` | `admm_compressor` | 125 | 127 | 98.43% |
| `adaptive_span_attention` | `adasvd_compressor` | 125 | 127 | 98.43% |
| `adaptive_span_attention` | `admm_compressor` | 125 | 127 | 98.43% |
| `adaptive_sparse_attention` | `adasvd_compressor` | 125 | 127 | 98.43% |
| `adaptive_sparse_attention` | `admm_compressor` | 125 | 127 | 98.43% |
| `adasvd_compressor` | `admm_compressor` | 125 | 127 | 98.43% |
| `AdaptiveSpanAttention` | `AdaptiveSparseAttention` | 34 | 181 | 18.78% |
| `AdaptiveSpanAttention` | `Primal` | 36 | 197 | 18.27% |
| `AdaptiveSpanAttention` | `HALO` | 29 | 186 | 15.59% |
| `AdaptiveSparseAttention` | `HALO` | 29 | 186 | 15.59% |
| `AdaptiveSparseAttention` | `SqueezingHeadsAttentionDistillationLayer` | 23 | 154 | 14.94% |
| `HALO` | `SqueezingHeadsAttentionDistillationLayer` | 26 | 186 | 13.98% |
| `HALO` | `OH` | 27 | 197 | 13.71% |
| `SOAFusedAttention` | `adaptive_svd_attention` | 21 | 159 | 13.21% |
| `AdaptiveSparseAttention` | `Primal` | 26 | 197 | 13.20% |
| `HALO` | `Primal` | 26 | 197 | 13.20% |
| `AdaptiveSVD` | `adasplash` | 34 | 261 | 13.03% |
| `AdaptiveSVD` | `AdaptiveSpanAttention` | 27 | 221 | 12.22% |
| `AdaptiveSVD` | `a3` | 27 | 221 | 12.22% |
| `AdaptiveSpanAttention` | `OH` | 24 | 197 | 12.18% |
| `AdaptiveSpanAttention` | `SqueezingHeadsAttentionDistillationLayer` | 22 | 181 | 12.15% |
| `AdaptiveSVD` | `Primal` | 26 | 221 | 11.76% |
| `AdaptiveSparseAttention` | `OH` | 23 | 197 | 11.68% |
| `OH` | `Primal` | 23 | 197 | 11.68% |
| `a3` | `aasvd` | 29 | 251 | 11.55% |
| `Primal` | `SqueezingHeadsAttentionDistillationLayer` | 22 | 197 | 11.17% |
| `AdaptiveSparseAttention` | `Galerkin` | 17 | 154 | 11.04% |
| `Galerkin` | `SqueezingHeadsAttentionDistillationLayer` | 16 | 146 | 10.96% |
| `AdaptiveSVD` | `OH` | 24 | 221 | 10.86% |
| `AdaptiveSparseAttention` | `adaptive_svd_attention` | 17 | 159 | 10.69% |
| `AdaptiveSVD` | `HALO` | 23 | 221 | 10.41% |
| `OH` | `SqueezingHeadsAttentionDistillationLayer` | 20 | 197 | 10.15% |
| `OH` | `adaptive_svd_attention` | 20 | 197 | 10.15% |
| `AdaptiveSpanAttention` | `a3` | 22 | 217 | 10.14% |
| `Primal` | `a3` | 22 | 217 | 10.14% |

- **Average overlap across sampled pairs:** 15.56%
- **Maximum overlap:** 99.21% (pair: `adaptive_span_attention` vs `adaptive_sparse_attention`)
- **Number of sampled pairs with > 80% overlap:** 21

---

## SECTION 9 — Category 8: Word Count Distribution (FULL HISTOGRAM)

- **Minimum:** 749 words (module: `minimal_aurora`)
- **Maximum:** 1722 words (module: `adasplash`)
- **Mean:** 1168.99
- **Median:** 1221.00
- **Standard deviation:** 139.36

### Histogram of Word Counts in 100-word Bins

| Range | Count | Percentage |
|---|---|---|
| 0-99 | 0 | 0.0% |
| 100-199 | 0 | 0.0% |
| 200-299 | 0 | 0.0% |
| 300-399 | 0 | 0.0% |
| 400-499 | 0 | 0.0% |
| 500-599 | 0 | 0.0% |
| 600-699 | 0 | 0.0% |
| 700-799 | 3 | 1.4% |
| 800-899 | 16 | 7.4% |
| 900-999 | 15 | 6.9% |
| 1000-1099 | 9 | 4.1% |
| 1100-1199 | 14 | 6.5% |
| 1200-1299 | 149 | 68.7% |
| 1300-1399 | 5 | 2.3% |
| 1400-1499 | 4 | 1.8% |
| 1500-1599 | 1 | 0.5% |
| 1600-1699 | 0 | 0.0% |
| 1700-1799 | 1 | 0.5% |

### Top 20 Most Common Exact Word Counts

| Word Count | Number of Modules |
|---|---|
| 1221 | 146 |
| 1146 | 2 |
| 992 | 2 |
| 990 | 2 |
| 1447 | 1 |
| 1103 | 1 |
| 1003 | 1 |
| 819 | 1 |
| 1129 | 1 |
| 1156 | 1 |
| 1282 | 1 |
| 895 | 1 |
| 853 | 1 |
| 1429 | 1 |
| 1551 | 1 |
| 987 | 1 |
| 1722 | 1 |
| 1087 | 1 |
| 1377 | 1 |
| 981 | 1 |

---

## SECTION 10 — Category 9: Section Duplication Within Papers (FULL)

| Module | Section A | Section B | Common Lines | % of Shorter Section |
|---|---|---|---|---|
| `unified_rope` | `04_method.md` | `B_implementation.md` | 42 | 29.8% |
| `nystrom_attention` | `05_architecture.md` | `B_implementation.md` | 38 | 27.7% |
| `nha` | `07_results.md` | `C_data.md` | 34 | 38.6% |
| `HALO` | `07_results.md` | `C_data.md` | 33 | 39.3% |
| `asvd` | `05_architecture.md` | `B_implementation.md` | 33 | 34.7% |
| `eat` | `05_architecture.md` | `B_implementation.md` | 33 | 25.2% |
| `minimal_nova` | `07_results.md` | `C_data.md` | 33 | 30.6% |
| `phantom` | `05_architecture.md` | `B_implementation.md` | 33 | 24.6% |
| `minimal_svd` | `07_results.md` | `C_data.md` | 32 | 26.9% |
| `asvd` | `04_method.md` | `B_implementation.md` | 30 | 31.6% |
| `minimal_prism` | `07_results.md` | `C_data.md` | 30 | 28.3% |
| `adasplash` | `05_architecture.md` | `B_implementation.md` | 29 | 20.6% |
| `asvd` | `04_method.md` | `05_architecture.md` | 28 | 18.9% |
| `asvd` | `07_results.md` | `C_data.md` | 28 | 27.2% |
| `ptp` | `07_results.md` | `C_data.md` | 28 | 32.2% |
| `flash_attention3` | `04_method.md` | `B_implementation.md` | 27 | 26.5% |
| `nsa` | `07_results.md` | `C_data.md` | 27 | 32.9% |
| `flash3` | `05_architecture.md` | `B_implementation.md` | 25 | 29.1% |
| `performer_attention` | `04_method.md` | `B_implementation.md` | 25 | 25.5% |
| `svd_llm_v2_attention` | `07_results.md` | `C_data.md` | 25 | 21.9% |
| `compressed_mha` | `05_architecture.md` | `B_implementation.md` | 24 | 26.1% |
| `reap_moe` | `05_architecture.md` | `B_implementation.md` | 24 | 25.3% |
| `svd_linear_attention` | `07_results.md` | `C_data.md` | 24 | 27.6% |
| `flash_attention3_fp8` | `04_method.md` | `B_implementation.md` | 23 | 19.7% |
| `nystrom_attention` | `08_analysis.md` | `C_data.md` | 23 | 24.0% |
| `gated_delta_net` | `05_architecture.md` | `B_implementation.md` | 22 | 16.5% |
| `lora_attention` | `04_method.md` | `B_implementation.md` | 22 | 25.9% |
| `salt` | `05_architecture.md` | `B_implementation.md` | 22 | 25.0% |
| `svd_mamba` | `07_results.md` | `C_data.md` | 22 | 25.0% |
| `minimal_aurora` | `05_architecture.md` | `B_implementation.md` | 21 | 33.9% |
| `unified_rope` | `07_results.md` | `C_data.md` | 21 | 24.1% |

Total internal duplicate pairs (> 20 lines): 31

---

## SECTION 11 — Category 10: Manual Inspection (Expanded to 20 Modules)

### Module 1: `halo_regla`
- **Expected math:** Specific formulation for `halo_regla`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 2: `upi_stuffed_mamba`
- **Expected math:** Specific formulation for `upi_stuffed_mamba`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 3: `compot_compressor`
- **Expected math:** Specific formulation for `compot_compressor`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 4: `nexus`
- **Expected math:** Specific formulation for `nexus`
- **Word count in 04_method.md:** 813
- **Source citations in 04_method.md:** 0
- **Verdict:** ⚠️ Partial / Stub
- **Actual equations found (first 14 displayed):**
  *(No display equations)*

### Module 5: `flap`
- **Expected math:** Specific formulation for `flap`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 6: `mamba`
- **Expected math:** Specific formulation for `mamba`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 7: `lora_attention`
- **Expected math:** Specific formulation for `lora_attention`
- **Word count in 04_method.md:** 996
- **Source citations in 04_method.md:** 0
- **Verdict:** ⚠️ Partial / Stub
- **Actual equations found (first 14 displayed):**
  1. $$y = \underbrace{W_\text{frozen} x}_{\text{frozen, copied from pretrained}} +      \underbrace{\frac{\alpha}{r_L} B A x}_{\text{trainable LoRA}} \tag{4.1}$$
  2. $$\text{Trainable}_\text{LoRA} = 16 \cdot (2 \cdot 768 + 2 \cdot 768) = 49,152 \tag{4.2}$$
  3. $$W_\text{merged} = W_\text{frozen} + \frac{\alpha}{r_L} B A \tag{4.3}$$
  4. $$y = W_\text{merged} x + b \tag{4.4}$$

### Module 8: `slicegpt`
- **Expected math:** Specific formulation for `slicegpt`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 9: `proxsparse_pruner`
- **Expected math:** Specific formulation for `proxsparse_pruner`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 10: `cur_linear`
- **Expected math:** Specific formulation for `cur_linear`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 11: `galerkin_attention`
- **Expected math:** Specific formulation for `galerkin_attention`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 12: `kmeans_quantize`
- **Expected math:** Specific formulation for `kmeans_quantize`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 13: `kronecker_compressor`
- **Expected math:** Specific formulation for `kronecker_compressor`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 14: `roast_hashed_linear`
- **Expected math:** Specific formulation for `roast_hashed_linear`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 15: `squeezing_heads`
- **Expected math:** Specific formulation for `squeezing_heads`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 16: `swift_svd`
- **Expected math:** Specific formulation for `swift_svd`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 17: `uniql_compressor`
- **Expected math:** Specific formulation for `uniql_compressor`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 18: `xattention`
- **Expected math:** Specific formulation for `xattention`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 19: `zerokv_neo`
- **Expected math:** Specific formulation for `zerokv_neo`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Module 20: `zeros_attention`
- **Expected math:** Specific formulation for `zeros_attention`
- **Word count in 04_method.md:** 1221
- **Source citations in 04_method.md:** 0
- **Verdict:** ❌ TEMPLATE CLONE
- **Actual equations found (first 14 displayed):**
  1. $$Z = X W^\top + b$$
  2. $$\hat{W} = U_r V_r$$
  3. $$\tilde{Z} = (X V_r^\top) U_r^\top + b$$
  4. $$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$
  5. $$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$
  6. $$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$
  7. $$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$
  8. $$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$
  9. $$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$
  10. $$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$
  11. $$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$
  12. $$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$
  13. $$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$
  14. $$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

### Full Content of 04_method.md for 3 Sample Modules

#### Complete Verbatim Content: `paper/PerModulePaper/halo_regla/04_method.md`

```markdown
# 4. Methodological Formulation and Execution Pipeline

## 4.1 Algorithmic Overview
The `halo_regla` engine executes a multi-stage transformation that replaces dense parameter projections with optimized computation kernels. The procedure comprises five formal phases:
1. **Model Inspection and Layer Discovery**
2. **Spectral or Saliency Decomposition**
3. **Hardware-Aligned Constraint Projection**
4. **Adaptive LoRA Residual Initialization**
5. **Numerical Stability and Fallback Verification**

```
+-------------------------------------------------------------------+
|               Pretrained Transformer Projection W                 |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 1: Numerical Inspection & Conditioning Analysis             |
|   - Compute ||W||_F, condition number kappa(W), verify NaN/Inf    |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 2: Factorization / Sparsification Transformation            |
|   - SVD: W -> U Sigma V^T   OR   Proximal: W -> S_lambda(W)       |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 3: Energy Thresholding & Constraint Projection              |
|   - Truncate rank r via E(r) >= tau   OR   Apply 2:4 Mask         |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 4: Residual Adapter Hooking                                 |
|   - Initialize LoRA adapters A, B with zero initial residual      |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 5: Verification Guardrail Check                             |
|   - If ||W - W_hat||_F / ||W||_F > max_err -> Fallback to Dense   |
+-------------------------------------------------------------------+
```

## 4.2 Mathematical Derivations and Equations

### Equation (1): Input Projection Mapping
Given hidden activation tensor $X \in \mathbb{R}^{\mathcal{B} \times N \times d_{\text{in}}}$, the linear projection generates target states:
$$Z = X W^\top + b$$

### Equation (2): Low-Rank Bilinear Factorization
For low-rank compressed variants, $W$ is decomposed into down-projection $V_r \in \mathbb{R}^{r \times d_{\text{in}}}$ and up-projection $U_r \in \mathbb{R}^{d_{\text{out}} \times r}$:
$$\hat{W} = U_r V_r$$

### Equation (3): Compressed Linear Forward Step
Evaluating the decomposed projection transforms standard $\mathcal{O}(d_{\text{in}} d_{\text{out}})$ operations into two chained operations:
$$\tilde{Z} = (X V_r^\top) U_r^\top + b$$

### Equation (4): Cumulative Spectral Energy Function
Rank truncation is dynamically governed by the normalized cumulative singular energy $\mathcal{E}(r)$:
$$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$

### Equation (5): Energy-Thresholded Optimal Rank Selection
Given energy threshold $\tau \in (0, 1)$ (default $\tau = 0.92$), the rank $r^*$ is the minimum integer satisfying:
$$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$

### Equation (6): Soft-Thresholding Operator
For proximal sparsity engines, the $\ell_1$ proximal shrinkage operator $\mathcal{S}_\lambda(\cdot)$ is applied element-wise:
$$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$

### Equation (7): Hardware-Aligned 2:4 Projection
For semi-structured acceleration on NVIDIA tensor cores, every group of four consecutive row elements is projected to retain exactly two non-zero coefficients:
$$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$

### Equation (8): Iterative Proximal Gradient Descent Step
The parameter matrix is updated across iterative calibration steps:
$$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$

### Equation (9): Low-Rank Residual LoRA Adaptation
To recover representation fidelity post-compression, low-rank adapters $B \in \mathbb{R}^{d_{\text{out}} \times r_{\text{lora}}}$ and $A \in \mathbb{R}^{r_{\text{lora}} \times d_{\text{in}}}$ augment the frozen base:
$$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$

### Equation (10): LoRA Adapter Initialization Condition
To guarantee identical inference output at step $t=0$, adapter parameters are initialized with:
$$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$

### Equation (11): Relative Reconstruction Error Metric
Prior to layer substitution, the framework validates numerical fidelity via relative Frobenius error:
$$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$

### Equation (12): Safety Fallback Verification Criterion
If the reconstruction discrepancy breaches the user-configured guardrail threshold $\delta_{\text{max}}$:
$$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$

### Equation (13): Attention Kernel Normalization (for Attention Modules)
For linear recurrent attention pathways, causal feature aggregation satisfies:
$$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$

### Equation (14): Multi-Head Projection Splitting
For multi-head attention blocks, projection weights are partitioned across $H$ heads:
$$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

## 4.3 Detailed Algorithmic Execution Trace

The complete structured execution of `LoRALinear` is carried out in four phases:

```python
# Phase 1: Conditioning and norm evaluation
W_f = layer.weight.data.float()
frob_norm = torch.linalg.norm(W_f)
assert frob_norm >= 1e-4, 'Degenerate weight norm'

# Phase 2: Core mathematical decomposition
if is_svd_factorization:
    U, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
    cum_energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
    r = torch.searchsorted(cum_energy, tau).item() + 1
    r = max(min_rank, min(r, max_rank))
    # Symmetric square-root distribution
    U_r = U[:, :r] * torch.sqrt(S[:r])
    V_r = torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]
elif is_proximal_sparsity:
    W_curr = W_f.clone()
    for step in range(num_steps):
        # Soft-thresholding
        W_soft = torch.sign(W_curr) * torch.clamp(torch.abs(W_curr) - lambda_prox, min=0.0)
        # Structural 2:4 or block projection
        W_proj = apply_structural_mask(W_soft, sparsity_target)
        W_curr = W_curr - eta * (W_curr - W_proj)
    W_r = W_proj

# Phase 3: Adapter binding & zero-initialization
lora_A = nn.Parameter(torch.randn(lora_r, d_in) / math.sqrt(lora_r))
lora_B = nn.Parameter(torch.zeros(d_out, lora_r))

# Phase 4: Fallback verification
recon_err = torch.linalg.norm(W_f - W_reconstructed) / frob_norm
if recon_err > max_error_ceiling or torch.isnan(recon_err):
    logger.warning('Triggering safety fallback to uncompressed layer')
    layer = uncompressed_layer_backup
```

## 4.4 Complexity and FLOPs Analysis

Evaluating a dense linear projection of dimension $d_{\text{in}} \times d_{\text{out}}$ requires $2 d_{\text{in}} d_{\text{out}}$ FLOPs per sequence token. For low-rank compressed factorization at rank $r$, the sequential forward evaluation requires:
$$\text{FLOPs}_{\text{factorized}} = 2 N d_{\text{in}} r + 2 N r d_{\text{out}} = 2 N r (d_{\text{in}} + d_{\text{out}})$$

The theoretical FLOP reduction ratio $\rho_{\text{FLOPs}}$ is therefore:
$$\rho_{\text{FLOPs}} = \frac{2 N d_{\text{in}} d_{\text{out}}}{2 N r (d_{\text{in}} + d_{\text{out}})} = \frac{d_{\text{in}} d_{\text{out}}}{r (d_{\text{in}} + d_{\text{out}})}$$

For symmetric hidden projections with $d_{\text{in}} = d_{\text{out}} = d$:
$$\rho_{\text{FLOPs}} = \frac{d^2}{2 r d} = \frac{d}{2 r}$$

When $d = 768$ and $r = 128$, the mathematical speedup reaches $3.0\times$.

## 4.5 Theoretical Guarantees and Proofs

**Theorem 1 (Energy Truncation Bound).** *Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ have singular value decomposition $W = U \Sigma V^\top$ with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k > 0$, where $k = \min(d_{\text{in}}, d_{\text{out}})$. If rank $r$ is selected such that:*
$$\mathcal{E}(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^k \sigma_j^2} \ge \tau, \quad \tau \in (0, 1)$$
*then the relative Frobenius reconstruction error satisfies:*
$$\frac{\|W - W_r\|_F}{\|W\|_F} \le \sqrt{1 - \tau}$$

*Proof.* By the Eckart-Young-Mirsky theorem, the optimal rank-$r$ approximation in the Frobenius norm satisfies $\|W - W_r\|_F^2 = \sum_{i=r+1}^k \sigma_i^2$. Total Frobenius energy satisfies $\|W\|_F^2 = \sum_{j=1}^k \sigma_j^2$. Expanding the energy ratio:
$$\frac{\|W - W_r\|_F^2}{\|W\|_F^2} = \frac{\sum_{i=r+1}^k \sigma_i^2}{\sum_{j=1}^k \sigma_j^2} = 1 - \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^k \sigma_j^2} = 1 - \mathcal{E}(r)$$
Since $\mathcal{E}(r) \ge \tau$, we immediately obtain:
$$\frac{\|W - W_r\|_F^2}{\|W\|_F^2} \le 1 - \tau \implies \frac{\|W - W_r\|_F}{\|W\|_F} \le \sqrt{1 - \tau}$$
This completes the proof. $\square$

**Theorem 2 (Initial Inference Invariance under Zero Residual).** *Let the effective weight matrix be parameterised as $W_{\text{eff}} = U_r V_r + \frac{\alpha}{r_L} B A$. If adapter up-projection $B$ is initialized to zero ($B = 0$), then $W_{\text{eff}}$ evaluated at initialization step $t=0$ identically equals the truncated base projection $W_r$.*

*Proof.* At initialization $t=0$, $B = 0 \in \mathbb{R}^{d_{\text{out}} \times r_L}$. For any arbitrary matrix $A \in \mathbb{R}^{r_L \times d_{\text{in}}}$:
$$B A = 0 \cdot A = 0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$
Consequently:
$$W_{\text{eff}} = U_r V_r + \frac{\alpha}{r_L} (0) = U_r V_r = W_r$$
Hence, output activations satisfy:
$$Y = X W_{\text{eff}}^\top + b = X W_r^\top + b$$
The initial forward pass preserves the exact mathematical characteristics of the truncated base projection without noise introduced by uncalibrated random adapter weights. $\square$

```

#### Complete Verbatim Content: `paper/PerModulePaper/upi_stuffed_mamba/04_method.md`

```markdown
# 4. Methodological Formulation and Execution Pipeline

## 4.1 Algorithmic Overview
The `upi_stuffed_mamba` engine executes a multi-stage transformation that replaces dense parameter projections with optimized computation kernels. The procedure comprises five formal phases:
1. **Model Inspection and Layer Discovery**
2. **Spectral or Saliency Decomposition**
3. **Hardware-Aligned Constraint Projection**
4. **Adaptive LoRA Residual Initialization**
5. **Numerical Stability and Fallback Verification**

```
+-------------------------------------------------------------------+
|               Pretrained Transformer Projection W                 |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 1: Numerical Inspection & Conditioning Analysis             |
|   - Compute ||W||_F, condition number kappa(W), verify NaN/Inf    |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 2: Factorization / Sparsification Transformation            |
|   - SVD: W -> U Sigma V^T   OR   Proximal: W -> S_lambda(W)       |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 3: Energy Thresholding & Constraint Projection              |
|   - Truncate rank r via E(r) >= tau   OR   Apply 2:4 Mask         |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 4: Residual Adapter Hooking                                 |
|   - Initialize LoRA adapters A, B with zero initial residual      |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 5: Verification Guardrail Check                             |
|   - If ||W - W_hat||_F / ||W||_F > max_err -> Fallback to Dense   |
+-------------------------------------------------------------------+
```

## 4.2 Mathematical Derivations and Equations

### Equation (1): Input Projection Mapping
Given hidden activation tensor $X \in \mathbb{R}^{\mathcal{B} \times N \times d_{\text{in}}}$, the linear projection generates target states:
$$Z = X W^\top + b$$

### Equation (2): Low-Rank Bilinear Factorization
For low-rank compressed variants, $W$ is decomposed into down-projection $V_r \in \mathbb{R}^{r \times d_{\text{in}}}$ and up-projection $U_r \in \mathbb{R}^{d_{\text{out}} \times r}$:
$$\hat{W} = U_r V_r$$

### Equation (3): Compressed Linear Forward Step
Evaluating the decomposed projection transforms standard $\mathcal{O}(d_{\text{in}} d_{\text{out}})$ operations into two chained operations:
$$\tilde{Z} = (X V_r^\top) U_r^\top + b$$

### Equation (4): Cumulative Spectral Energy Function
Rank truncation is dynamically governed by the normalized cumulative singular energy $\mathcal{E}(r)$:
$$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$

### Equation (5): Energy-Thresholded Optimal Rank Selection
Given energy threshold $\tau \in (0, 1)$ (default $\tau = 0.92$), the rank $r^*$ is the minimum integer satisfying:
$$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$

### Equation (6): Soft-Thresholding Operator
For proximal sparsity engines, the $\ell_1$ proximal shrinkage operator $\mathcal{S}_\lambda(\cdot)$ is applied element-wise:
$$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$

### Equation (7): Hardware-Aligned 2:4 Projection
For semi-structured acceleration on NVIDIA tensor cores, every group of four consecutive row elements is projected to retain exactly two non-zero coefficients:
$$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$

### Equation (8): Iterative Proximal Gradient Descent Step
The parameter matrix is updated across iterative calibration steps:
$$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$

### Equation (9): Low-Rank Residual LoRA Adaptation
To recover representation fidelity post-compression, low-rank adapters $B \in \mathbb{R}^{d_{\text{out}} \times r_{\text{lora}}}$ and $A \in \mathbb{R}^{r_{\text{lora}} \times d_{\text{in}}}$ augment the frozen base:
$$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$

### Equation (10): LoRA Adapter Initialization Condition
To guarantee identical inference output at step $t=0$, adapter parameters are initialized with:
$$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$

### Equation (11): Relative Reconstruction Error Metric
Prior to layer substitution, the framework validates numerical fidelity via relative Frobenius error:
$$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$

### Equation (12): Safety Fallback Verification Criterion
If the reconstruction discrepancy breaches the user-configured guardrail threshold $\delta_{\text{max}}$:
$$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$

### Equation (13): Attention Kernel Normalization (for Attention Modules)
For linear recurrent attention pathways, causal feature aggregation satisfies:
$$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$

### Equation (14): Multi-Head Projection Splitting
For multi-head attention blocks, projection weights are partitioned across $H$ heads:
$$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

## 4.3 Detailed Algorithmic Execution Trace

The complete structured execution of `LoRALinear` is carried out in four phases:

```python
# Phase 1: Conditioning and norm evaluation
W_f = layer.weight.data.float()
frob_norm = torch.linalg.norm(W_f)
assert frob_norm >= 1e-4, 'Degenerate weight norm'

# Phase 2: Core mathematical decomposition
if is_svd_factorization:
    U, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
    cum_energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
    r = torch.searchsorted(cum_energy, tau).item() + 1
    r = max(min_rank, min(r, max_rank))
    # Symmetric square-root distribution
    U_r = U[:, :r] * torch.sqrt(S[:r])
    V_r = torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]
elif is_proximal_sparsity:
    W_curr = W_f.clone()
    for step in range(num_steps):
        # Soft-thresholding
        W_soft = torch.sign(W_curr) * torch.clamp(torch.abs(W_curr) - lambda_prox, min=0.0)
        # Structural 2:4 or block projection
        W_proj = apply_structural_mask(W_soft, sparsity_target)
        W_curr = W_curr - eta * (W_curr - W_proj)
    W_r = W_proj

# Phase 3: Adapter binding & zero-initialization
lora_A = nn.Parameter(torch.randn(lora_r, d_in) / math.sqrt(lora_r))
lora_B = nn.Parameter(torch.zeros(d_out, lora_r))

# Phase 4: Fallback verification
recon_err = torch.linalg.norm(W_f - W_reconstructed) / frob_norm
if recon_err > max_error_ceiling or torch.isnan(recon_err):
    logger.warning('Triggering safety fallback to uncompressed layer')
    layer = uncompressed_layer_backup
```

## 4.4 Complexity and FLOPs Analysis

Evaluating a dense linear projection of dimension $d_{\text{in}} \times d_{\text{out}}$ requires $2 d_{\text{in}} d_{\text{out}}$ FLOPs per sequence token. For low-rank compressed factorization at rank $r$, the sequential forward evaluation requires:
$$\text{FLOPs}_{\text{factorized}} = 2 N d_{\text{in}} r + 2 N r d_{\text{out}} = 2 N r (d_{\text{in}} + d_{\text{out}})$$

The theoretical FLOP reduction ratio $\rho_{\text{FLOPs}}$ is therefore:
$$\rho_{\text{FLOPs}} = \frac{2 N d_{\text{in}} d_{\text{out}}}{2 N r (d_{\text{in}} + d_{\text{out}})} = \frac{d_{\text{in}} d_{\text{out}}}{r (d_{\text{in}} + d_{\text{out}})}$$

For symmetric hidden projections with $d_{\text{in}} = d_{\text{out}} = d$:
$$\rho_{\text{FLOPs}} = \frac{d^2}{2 r d} = \frac{d}{2 r}$$

When $d = 768$ and $r = 128$, the mathematical speedup reaches $3.0\times$.

## 4.5 Theoretical Guarantees and Proofs

**Theorem 1 (Energy Truncation Bound).** *Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ have singular value decomposition $W = U \Sigma V^\top$ with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k > 0$, where $k = \min(d_{\text{in}}, d_{\text{out}})$. If rank $r$ is selected such that:*
$$\mathcal{E}(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^k \sigma_j^2} \ge \tau, \quad \tau \in (0, 1)$$
*then the relative Frobenius reconstruction error satisfies:*
$$\frac{\|W - W_r\|_F}{\|W\|_F} \le \sqrt{1 - \tau}$$

*Proof.* By the Eckart-Young-Mirsky theorem, the optimal rank-$r$ approximation in the Frobenius norm satisfies $\|W - W_r\|_F^2 = \sum_{i=r+1}^k \sigma_i^2$. Total Frobenius energy satisfies $\|W\|_F^2 = \sum_{j=1}^k \sigma_j^2$. Expanding the energy ratio:
$$\frac{\|W - W_r\|_F^2}{\|W\|_F^2} = \frac{\sum_{i=r+1}^k \sigma_i^2}{\sum_{j=1}^k \sigma_j^2} = 1 - \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^k \sigma_j^2} = 1 - \mathcal{E}(r)$$
Since $\mathcal{E}(r) \ge \tau$, we immediately obtain:
$$\frac{\|W - W_r\|_F^2}{\|W\|_F^2} \le 1 - \tau \implies \frac{\|W - W_r\|_F}{\|W\|_F} \le \sqrt{1 - \tau}$$
This completes the proof. $\square$

**Theorem 2 (Initial Inference Invariance under Zero Residual).** *Let the effective weight matrix be parameterised as $W_{\text{eff}} = U_r V_r + \frac{\alpha}{r_L} B A$. If adapter up-projection $B$ is initialized to zero ($B = 0$), then $W_{\text{eff}}$ evaluated at initialization step $t=0$ identically equals the truncated base projection $W_r$.*

*Proof.* At initialization $t=0$, $B = 0 \in \mathbb{R}^{d_{\text{out}} \times r_L}$. For any arbitrary matrix $A \in \mathbb{R}^{r_L \times d_{\text{in}}}$:
$$B A = 0 \cdot A = 0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$
Consequently:
$$W_{\text{eff}} = U_r V_r + \frac{\alpha}{r_L} (0) = U_r V_r = W_r$$
Hence, output activations satisfy:
$$Y = X W_{\text{eff}}^\top + b = X W_r^\top + b$$
The initial forward pass preserves the exact mathematical characteristics of the truncated base projection without noise introduced by uncalibrated random adapter weights. $\square$

```

#### Complete Verbatim Content: `paper/PerModulePaper/compot_compressor/04_method.md`

```markdown
# 4. Methodological Formulation and Execution Pipeline

## 4.1 Algorithmic Overview
The `compot_compressor` engine executes a multi-stage transformation that replaces dense parameter projections with optimized computation kernels. The procedure comprises five formal phases:
1. **Model Inspection and Layer Discovery**
2. **Spectral or Saliency Decomposition**
3. **Hardware-Aligned Constraint Projection**
4. **Adaptive LoRA Residual Initialization**
5. **Numerical Stability and Fallback Verification**

```
+-------------------------------------------------------------------+
|               Pretrained Transformer Projection W                 |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 1: Numerical Inspection & Conditioning Analysis             |
|   - Compute ||W||_F, condition number kappa(W), verify NaN/Inf    |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 2: Factorization / Sparsification Transformation            |
|   - SVD: W -> U Sigma V^T   OR   Proximal: W -> S_lambda(W)       |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 3: Energy Thresholding & Constraint Projection              |
|   - Truncate rank r via E(r) >= tau   OR   Apply 2:4 Mask         |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 4: Residual Adapter Hooking                                 |
|   - Initialize LoRA adapters A, B with zero initial residual      |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Phase 5: Verification Guardrail Check                             |
|   - If ||W - W_hat||_F / ||W||_F > max_err -> Fallback to Dense   |
+-------------------------------------------------------------------+
```

## 4.2 Mathematical Derivations and Equations

### Equation (1): Input Projection Mapping
Given hidden activation tensor $X \in \mathbb{R}^{\mathcal{B} \times N \times d_{\text{in}}}$, the linear projection generates target states:
$$Z = X W^\top + b$$

### Equation (2): Low-Rank Bilinear Factorization
For low-rank compressed variants, $W$ is decomposed into down-projection $V_r \in \mathbb{R}^{r \times d_{\text{in}}}$ and up-projection $U_r \in \mathbb{R}^{d_{\text{out}} \times r}$:
$$\hat{W} = U_r V_r$$

### Equation (3): Compressed Linear Forward Step
Evaluating the decomposed projection transforms standard $\mathcal{O}(d_{\text{in}} d_{\text{out}})$ operations into two chained operations:
$$\tilde{Z} = (X V_r^\top) U_r^\top + b$$

### Equation (4): Cumulative Spectral Energy Function
Rank truncation is dynamically governed by the normalized cumulative singular energy $\mathcal{E}(r)$:
$$\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2}$$

### Equation (5): Energy-Thresholded Optimal Rank Selection
Given energy threshold $\tau \in (0, 1)$ (default $\tau = 0.92$), the rank $r^*$ is the minimum integer satisfying:
$$r^* = \arg\min_{r} \left\{ r \in \mathbb{N} : \mathcal{E}(r) \ge \tau \right\}$$

### Equation (6): Soft-Thresholding Operator
For proximal sparsity engines, the $\ell_1$ proximal shrinkage operator $\mathcal{S}_\lambda(\cdot)$ is applied element-wise:
$$\mathcal{S}_\lambda(W) = \operatorname{sign}(W) \odot \max\left( |W| - \lambda, 0 \right)$$

### Equation (7): Hardware-Aligned 2:4 Projection
For semi-structured acceleration on NVIDIA tensor cores, every group of four consecutive row elements is projected to retain exactly two non-zero coefficients:
$$\Pi_{2:4}(W_{i, 4k:4k+4}) = W_{i, 4k:4k+4} \odot M_{i, 4k:4k+4}, \quad \sum_{j=0}^{3} M_{i, 4k+j} = 2$$

### Equation (8): Iterative Proximal Gradient Descent Step
The parameter matrix is updated across iterative calibration steps:
$$W^{(t+1)} = \Pi_{\mathcal{S}}\left( W^{(t)} - \eta \nabla f(W^{(t)}) \right)$$

### Equation (9): Low-Rank Residual LoRA Adaptation
To recover representation fidelity post-compression, low-rank adapters $B \in \mathbb{R}^{d_{\text{out}} \times r_{\text{lora}}}$ and $A \in \mathbb{R}^{r_{\text{lora}} \times d_{\text{in}}}$ augment the frozen base:
$$Y = X \hat{W}^\top + \frac{\alpha}{r_{\text{lora}}} (X A^\top) B^\top + b$$

### Equation (10): LoRA Adapter Initialization Condition
To guarantee identical inference output at step $t=0$, adapter parameters are initialized with:
$$A \sim \mathcal{N}\left(0, \frac{1}{r_{\text{lora}}}\right), \quad B = 0 \implies B A = 0$$

### Equation (11): Relative Reconstruction Error Metric
Prior to layer substitution, the framework validates numerical fidelity via relative Frobenius error:
$$\delta_{\text{rel}}(W, \hat{W}) = \frac{\|W - \hat{W}\|_F}{\|W\|_F + \epsilon}$$

### Equation (12): Safety Fallback Verification Criterion
If the reconstruction discrepancy breaches the user-configured guardrail threshold $\delta_{\text{max}}$:
$$\operatorname{Module}(W) = \begin{cases} \hat{W} + \frac{\alpha}{r} B A & \text{if } \delta_{\text{rel}} \le \delta_{\text{max}} \text{ and } \operatorname{isfinite}(\hat{W}) \\ W & \text{otherwise (safety fallback triggered)} \end{cases}$$

### Equation (13): Attention Kernel Normalization (for Attention Modules)
For linear recurrent attention pathways, causal feature aggregation satisfies:
$$O_t = \frac{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) (\phi(K_s)^\top V_s)}{\sum_{s=1}^{t} \exp(-\gamma (t-s)) \phi(Q_t) \phi(K_s)^\top + \epsilon}$$

### Equation (14): Multi-Head Projection Splitting
For multi-head attention blocks, projection weights are partitioned across $H$ heads:
$$W_Q = [W_Q^{(1)}; W_Q^{(2)}; \dots; W_Q^{(H)}], \quad W_Q^{(h)} \in \mathbb{R}^{d \times d_{\text{in}}}$$

## 4.3 Detailed Algorithmic Execution Trace

The complete structured execution of `COMPOTLinear` is carried out in four phases:

```python
# Phase 1: Conditioning and norm evaluation
W_f = layer.weight.data.float()
frob_norm = torch.linalg.norm(W_f)
assert frob_norm >= 1e-4, 'Degenerate weight norm'

# Phase 2: Core mathematical decomposition
if is_svd_factorization:
    U, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
    cum_energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
    r = torch.searchsorted(cum_energy, tau).item() + 1
    r = max(min_rank, min(r, max_rank))
    # Symmetric square-root distribution
    U_r = U[:, :r] * torch.sqrt(S[:r])
    V_r = torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]
elif is_proximal_sparsity:
    W_curr = W_f.clone()
    for step in range(num_steps):
        # Soft-thresholding
        W_soft = torch.sign(W_curr) * torch.clamp(torch.abs(W_curr) - lambda_prox, min=0.0)
        # Structural 2:4 or block projection
        W_proj = apply_structural_mask(W_soft, sparsity_target)
        W_curr = W_curr - eta * (W_curr - W_proj)
    W_r = W_proj

# Phase 3: Adapter binding & zero-initialization
lora_A = nn.Parameter(torch.randn(lora_r, d_in) / math.sqrt(lora_r))
lora_B = nn.Parameter(torch.zeros(d_out, lora_r))

# Phase 4: Fallback verification
recon_err = torch.linalg.norm(W_f - W_reconstructed) / frob_norm
if recon_err > max_error_ceiling or torch.isnan(recon_err):
    logger.warning('Triggering safety fallback to uncompressed layer')
    layer = uncompressed_layer_backup
```

## 4.4 Complexity and FLOPs Analysis

Evaluating a dense linear projection of dimension $d_{\text{in}} \times d_{\text{out}}$ requires $2 d_{\text{in}} d_{\text{out}}$ FLOPs per sequence token. For low-rank compressed factorization at rank $r$, the sequential forward evaluation requires:
$$\text{FLOPs}_{\text{factorized}} = 2 N d_{\text{in}} r + 2 N r d_{\text{out}} = 2 N r (d_{\text{in}} + d_{\text{out}})$$

The theoretical FLOP reduction ratio $\rho_{\text{FLOPs}}$ is therefore:
$$\rho_{\text{FLOPs}} = \frac{2 N d_{\text{in}} d_{\text{out}}}{2 N r (d_{\text{in}} + d_{\text{out}})} = \frac{d_{\text{in}} d_{\text{out}}}{r (d_{\text{in}} + d_{\text{out}})}$$

For symmetric hidden projections with $d_{\text{in}} = d_{\text{out}} = d$:
$$\rho_{\text{FLOPs}} = \frac{d^2}{2 r d} = \frac{d}{2 r}$$

When $d = 768$ and $r = 128$, the mathematical speedup reaches $3.0\times$.

## 4.5 Theoretical Guarantees and Proofs

**Theorem 1 (Energy Truncation Bound).** *Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ have singular value decomposition $W = U \Sigma V^\top$ with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k > 0$, where $k = \min(d_{\text{in}}, d_{\text{out}})$. If rank $r$ is selected such that:*
$$\mathcal{E}(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^k \sigma_j^2} \ge \tau, \quad \tau \in (0, 1)$$
*then the relative Frobenius reconstruction error satisfies:*
$$\frac{\|W - W_r\|_F}{\|W\|_F} \le \sqrt{1 - \tau}$$

*Proof.* By the Eckart-Young-Mirsky theorem, the optimal rank-$r$ approximation in the Frobenius norm satisfies $\|W - W_r\|_F^2 = \sum_{i=r+1}^k \sigma_i^2$. Total Frobenius energy satisfies $\|W\|_F^2 = \sum_{j=1}^k \sigma_j^2$. Expanding the energy ratio:
$$\frac{\|W - W_r\|_F^2}{\|W\|_F^2} = \frac{\sum_{i=r+1}^k \sigma_i^2}{\sum_{j=1}^k \sigma_j^2} = 1 - \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^k \sigma_j^2} = 1 - \mathcal{E}(r)$$
Since $\mathcal{E}(r) \ge \tau$, we immediately obtain:
$$\frac{\|W - W_r\|_F^2}{\|W\|_F^2} \le 1 - \tau \implies \frac{\|W - W_r\|_F}{\|W\|_F} \le \sqrt{1 - \tau}$$
This completes the proof. $\square$

**Theorem 2 (Initial Inference Invariance under Zero Residual).** *Let the effective weight matrix be parameterised as $W_{\text{eff}} = U_r V_r + \frac{\alpha}{r_L} B A$. If adapter up-projection $B$ is initialized to zero ($B = 0$), then $W_{\text{eff}}$ evaluated at initialization step $t=0$ identically equals the truncated base projection $W_r$.*

*Proof.* At initialization $t=0$, $B = 0 \in \mathbb{R}^{d_{\text{out}} \times r_L}$. For any arbitrary matrix $A \in \mathbb{R}^{r_L \times d_{\text{in}}}$:
$$B A = 0 \cdot A = 0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$
Consequently:
$$W_{\text{eff}} = U_r V_r + \frac{\alpha}{r_L} (0) = U_r V_r = W_r$$
Hence, output activations satisfy:
$$Y = X W_{\text{eff}}^\top + b = X W_r^\top + b$$
The initial forward pass preserves the exact mathematical characteristics of the truncated base projection without noise introduced by uncalibrated random adapter weights. $\square$

```

---

## SECTION 12 — Master Evidence Table

| Module | Cat | Artifact Count | Word Count | Equations | Citations | SHA256 Prefix | Verdict |
|---|:---:|---:|---:|---:|---:|:---:|:---:|
| `AdaptiveSVD` | 3 (Placeholders) | 729 | 1447 | 13 | 0 | `f51fc1671c` | **CONTAMINATED** |
| `AdaptiveSpanAttention` | 3 (Placeholders) | 622 | 1103 | 14 | 0 | `f0d1f2aac1` | **CONTAMINATED** |
| `AdaptiveSparseAttention` | 3 (Placeholders) | 590 | 1003 | 10 | 0 | `00895c672e` | **CONTAMINATED** |
| `Galerkin` | 3 (Placeholders) | 562 | 819 | 9 | 0 | `1a241d0cd4` | **CONTAMINATED** |
| `HALO` | 3 (Placeholders) | 542 | 1129 | 12 | 0 | `5fc9ea797f` | **CONTAMINATED** |
| `OH` | 3 (Placeholders) | 465 | 1156 | 12 | 0 | `bf29e27c27` | **CONTAMINATED** |
| `Primal` | 3 (Placeholders) | 542 | 1282 | 15 | 0 | `fbc77944ad` | **CONTAMINATED** |
| `SOAFusedAttention` | 3 (Placeholders) | 235 | 895 | 5 | 0 | `18ea59270f` | **CONTAMINATED** |
| `SqueezingHeadsAttentionDistillationLayer` | 3 (Placeholders) | 383 | 853 | 8 | 0 | `ca6a5dbc1f` | **CONTAMINATED** |
| `a3` | 1 (Double Char) | 826 | 1429 | 14 | 0 | `252210f9a9` | **CONTAMINATED** |
| `aa_svd_compressor` | 5 (SVD/LoRA Clone) | 332 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `aasvd` | 1 (Double Char) | 962 | 1551 | 20 | 0 | `0ca1ed510f` | **CONTAMINATED** |
| `adaptive_compressor` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `adaptive_ranks` | 5 (SVD/LoRA Clone) | 332 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `adaptive_span_attention` | 5 (SVD/LoRA Clone) | 332 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `adaptive_sparse_attention` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `adaptive_svd_attention` | 3 (Placeholders) | 254 | 987 | 10 | 0 | `ac2d41edf7` | **CONTAMINATED** |
| `adasplash` | 3 (Placeholders) | 696 | 1722 | 7 | 0 | `338a8a1953` | **CONTAMINATED** |
| `adasvd_compressor` | 5 (SVD/LoRA Clone) | 340 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `admm_compressor` | 5 (SVD/LoRA Clone) | 348 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `admm_pruner` | 5 (SVD/LoRA Clone) | 364 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `aegis_hybrid` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `aegis_spectron_fp8` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `aegis_v1` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `aegis_v2` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `aegis_v3` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `aegis_v4` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `aegis_v5` | 1 (Double Char) | 351 | 1087 | 12 | 0 | `14eb171c91` | **CONTAMINATED** |
| `aegis_v6` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `armor_pruner` | 5 (SVD/LoRA Clone) | 346 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `ast_pruner` | 5 (SVD/LoRA Clone) | 344 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `astro_quantizer` | 5 (SVD/LoRA Clone) | 348 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `asvd` | 3 (Placeholders) | 681 | 1377 | 7 | 0 | `cf28774b98` | **CONTAMINATED** |
| `bam_umoe` | 3 (Placeholders) | 357 | 981 | 11 | 0 | `80784d10fe` | **CONTAMINATED** |
| `block_transformer_hiba` | 5 (SVD/LoRA Clone) | 473 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `bonsai_pruner` | 5 (SVD/LoRA Clone) | 344 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `cachenotes_compressor` | 5 (SVD/LoRA Clone) | 348 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `chela_slay` | 3 (Placeholders) | 469 | 916 | 11 | 0 | `ebd609cb16` | **CONTAMINATED** |
| `cluster` | 3 (Placeholders) | 252 | 1301 | 11 | 0 | `5d905191e0` | **CONTAMINATED** |
| `come` | 5 (SVD/LoRA Clone) | 340 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `compot_compressor` | 5 (SVD/LoRA Clone) | 324 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `compressed_attention` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `compressed_mha` | 3 (Placeholders) | 299 | 1326 | 19 | 0 | `19f6c456cf` | **CONTAMINATED** |
| `coopq_quantizer` | 5 (SVD/LoRA Clone) | 344 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `cosformer` | 3 (Placeholders) | 459 | 1160 | 14 | 0 | `b5387b61d0` | **CONTAMINATED** |
| `cr_compressor` | 5 (SVD/LoRA Clone) | 324 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `cur_compressor` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `cur_linear` | 5 (SVD/LoRA Clone) | 332 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `cur_moe` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `cure` | 5 (SVD/LoRA Clone) | 332 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `decay_linear_attention` | 5 (SVD/LoRA Clone) | 522 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `delta_kv` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `direct_svd` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `dobi` | 5 (SVD/LoRA Clone) | 332 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `dobi_svd` | 1 (Double Char) | 620 | 1089 | 15 | 0 | `06fd87ed8d` | **CONTAMINATED** |
| `dynamic_entropy` | 5 (SVD/LoRA Clone) | 332 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `dynamic_rank` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `eat` | 1 (Double Char) | 567 | 1067 | 5 | 0 | `1bdc272f9c` | **CONTAMINATED** |
| `enlighten_compressor` | 5 (SVD/LoRA Clone) | 338 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `entropy_budget_allocation` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `erc_svd` | 5 (SVD/LoRA Clone) | 338 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `expert_choice_moe` | 1 (Double Char) | 290 | 935 | 15 | 0 | `d1b9c93f25` | **CONTAMINATED** |
| `factorised_matrix_state` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `faq_quantizer` | 5 (SVD/LoRA Clone) | 344 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `flap` | 5 (SVD/LoRA Clone) | 338 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `flash3` | 1 (Double Char) | 554 | 860 | 5 | 0 | `653e3a74ed` | **CONTAMINATED** |
| `flash_attention3` | 1 (Double Char) | 419 | 780 | 1 | 0 | `f8f9e83bf3` | **CONTAMINATED** |
| `flash_attention3_fp8` | 1 (Double Char) | 695 | 953 | 1 | 0 | `7d6523adc1` | **CONTAMINATED** |
| `flat_llm` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `frandi` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `full_attention` | 5 (SVD/LoRA Clone) | 322 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `fused_svd_kernel` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `fusion` | 1 (Double Char) | 409 | 907 | 10 | 0 | `20f480c7a2` | **CONTAMINATED** |
| `fusion_attn` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `galerkin_attention` | 5 (SVD/LoRA Clone) | 340 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `gated_delta_net` | 1 (Double Char) | 578 | 1010 | 9 | 0 | `736c5f6f0a` | **CONTAMINATED** |
| `gated_deltanet_block` | 1 (Double Char) | 515 | 812 | 2 | 0 | `d7daf4db84` | **CONTAMINATED** |
| `gated_linear_attention` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `gfws_d_compressor` | 5 (SVD/LoRA Clone) | 324 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `goldfinch_ahn` | 1 (Double Char) | 471 | 946 | 12 | 0 | `51b1c8680a` | **CONTAMINATED** |
| `gqa` | 3 (Placeholders) | 553 | 1154 | 8 | 0 | `a8395b6fcf` | **CONTAMINATED** |
| `granite_hybrid` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `grecodec` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `griffin_ssd` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `halo_regla` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `hamming_attention` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `hashed_linear` | 5 (SVD/LoRA Clone) | 332 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `hashed_net_compressor` | 5 (SVD/LoRA Clone) | 324 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `hgrn2_gated_deltanet` | 1 (Double Char) | 393 | 795 | 8 | 0 | `7e3a5bcd29` | **CONTAMINATED** |
| `hgrn2_layer` | 1 (Double Char) | 310 | 868 | 18 | 0 | `e8acb47cce` | **CONTAMINATED** |
| `hgrn2_outer` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `hla_attention` | 5 (SVD/LoRA Clone) | 340 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `huffman_coder` | 5 (SVD/LoRA Clone) | 332 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `hybrid_svd_layer` | 5 (SVD/LoRA Clone) | 332 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `hybridkv` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `hymba_transx` | 1 (Double Char) | 295 | 876 | 14 | 0 | `5659445a50` | **CONTAMINATED** |
| `infini` | 1 (Double Char) | 517 | 1181 | 8 | 0 | `a97402d6a2` | **CONTAMINATED** |
| `jamba` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `jamba_info` | 1 (Double Char) | 261 | 852 | 10 | 0 | `e0e9a67c9c` | **CONTAMINATED** |
| `kda_attention` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `kmeans_quantize` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `kronecker_compressor` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `kronecker_linear` | 5 (SVD/LoRA Clone) | 332 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `kronecker_tt_lora` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `kvtc` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `leka_v1` | 1 (Double Char) | 240 | 924 | 15 | 0 | `bf41e8288a` | **CONTAMINATED** |
| `leka_v2` | 1 (Double Char) | 259 | 943 | 14 | 0 | `4c467b38e8` | **CONTAMINATED** |
| `leka_v3` | 1 (Double Char) | 322 | 1146 | 17 | 0 | `d6845d9d09` | **CONTAMINATED** |
| `liftr` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `lillama_compressor` | 5 (SVD/LoRA Clone) | 344 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `linformer_attention` | 1 (Double Char) | 249 | 1145 | 5 | 0 | `c43cd4fcc2` | **CONTAMINATED** |
| `lola_attention` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `lora_attention` | 1 (Double Char) | 302 | 996 | 4 | 0 | `2a44b92bda` | **CONTAMINATED** |
| `lsc_compressor` | 5 (SVD/LoRA Clone) | 344 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `lsh_attention` | 5 (SVD/LoRA Clone) | 340 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `mamba` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `mamba_inject` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `mamba_shedder` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `mamba_svd_flash` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `mgie_svd` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `minimal_aurora` | 1 (Double Char) | 295 | 749 | 5 | 0 | `52045a38ce` | **CONTAMINATED** |
| `minimal_nova` | 1 (Double Char) | 583 | 1379 | 9 | 0 | `86dde7fd17` | **CONTAMINATED** |
| `minimal_prism` | 1 (Double Char) | 623 | 1413 | 9 | 0 | `ca5e954b9a` | **CONTAMINATED** |
| `minimal_svd` | 1 (Double Char) | 627 | 1235 | 9 | 0 | `ef8ed987f5` | **CONTAMINATED** |
| `minimal_svd_v2` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `mixattention_moska` | 1 (Double Char) | 702 | 850 | 8 | 0 | `40f5eff757` | **CONTAMINATED** |
| `mla_attention` | 1 (Double Char) | 268 | 1079 | 7 | 0 | `91beea93ef` | **CONTAMINATED** |
| `mlorq_compressor` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `moba` | 1 (Double Char) | 245 | 1302 | 16 | 0 | `4c5cb6dacc` | **CONTAMINATED** |
| `modegpt` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `moe_deltakv_paged` | 3 (Placeholders) | 247 | 1071 | 3 | 0 | `7dc392db3d` | **CONTAMINATED** |
| `nasc_attention` | 1 (Double Char) | 597 | 872 | 15 | 0 | `19d7b6d06b` | **CONTAMINATED** |
| `nexus` | 3 (Placeholders) | 275 | 813 | 0 | 0 | `cda3d4b886` | **CONTAMINATED** |
| `nha` | 3 (Placeholders) | 446 | 1155 | 16 | 0 | `0b6c611859` | **CONTAMINATED** |
| `nli_softmax` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `nmf_compressor` | 5 (SVD/LoRA Clone) | 344 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `nsa` | 1 (Double Char) | 447 | 1125 | 11 | 0 | `bf64acc198` | **CONTAMINATED** |
| `nystrom_attention` | 1 (Double Char) | 490 | 992 | 4 | 0 | `55cc86d697` | **CONTAMINATED** |
| `oats` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `optimal_subspace_projection` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `optimized_hybrid_attention` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `orion` | 3 (Placeholders) | 570 | 990 | 11 | 0 | `801a534e10` | **CONTAMINATED** |
| `padre_attention` | 3 (Placeholders) | 238 | 990 | 15 | 0 | `b3473e5369` | **CONTAMINATED** |
| `performer_attention` | 1 (Double Char) | 583 | 804 | 7 | 0 | `1c5536f55a` | **CONTAMINATED** |
| `phantom` | 1 (Double Char) | 646 | 1431 | 29 | 0 | `d32f735866` | **CONTAMINATED** |
| `progressive_token_pruning` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `projection_factorized_attn` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `proxsparse` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `proxsparse_pruner` | 5 (SVD/LoRA Clone) | 352 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `pruning_utils` | 5 (SVD/LoRA Clone) | 322 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `ptp` | 3 (Placeholders) | 474 | 1236 | 10 | 0 | `489bf191ac` | **CONTAMINATED** |
| `qcqa_attention` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `rala_attention` | 1 (Double Char) | 618 | 1146 | 7 | 0 | `cbf1ea9534` | **CONTAMINATED** |
| `reap_moe` | 1 (Double Char) | 365 | 894 | 19 | 0 | `097c289c53` | **CONTAMINATED** |
| `regla_attention` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `replace` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `replaceme_compressor` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `res_svd` | 1 (Double Char) | 272 | 840 | 8 | 0 | `8ac82ac30c` | **CONTAMINATED** |
| `roast_hashed_linear` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `rotational_slicegpt_int4` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `routing_attention` | 1 (Double Char) | 478 | 1159 | 6 | 0 | `04a30d9947` | **CONTAMINATED** |
| `rpc` | 1 (Double Char) | 514 | 878 | 8 | 0 | `f277a35206` | **CONTAMINATED** |
| `rpca` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `saes_svd` | 5 (SVD/LoRA Clone) | 322 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `saes_svd_v2` | 5 (SVD/LoRA Clone) | 338 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `salr` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `salt` | 3 (Placeholders) | 612 | 1037 | 5 | 0 | `4706190070` | **CONTAMINATED** |
| `samba_m3` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `samba_moe` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `sdrm` | 5 (SVD/LoRA Clone) | 338 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `simple_ssm` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `slicegpt` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `slicegpt_pca` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `slim_compressor` | 5 (SVD/LoRA Clone) | 346 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `smart_weighted_ensemble` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `soa_attn` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `soa_fused_module` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `soa_module` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `soka` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `span_expanded_attention` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `sparsegpt` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `spectra_v2` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `spectraformer_attention` | 5 (SVD/LoRA Clone) | 340 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `spectral_mamba` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `spectron` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `squeezing_heads` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `squeezing_heads_distill` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `squeezing_heads_distiller` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `subspace_projection` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `svd_attention` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `svd_compressed` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `svd_linear_attention` | 1 (Double Char) | 555 | 1198 | 17 | 0 | `8be64496b8` | **CONTAMINATED** |
| `svd_linear_decay` | 5 (SVD/LoRA Clone) | 332 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `svd_llm_v2_attention` | 1 (Double Char) | 587 | 992 | 4 | 0 | `928428a136` | **CONTAMINATED** |
| `svd_mamba` | 1 (Double Char) | 527 | 1191 | 22 | 0 | `f7bf104288` | **CONTAMINATED** |
| `svd_proj_attn` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `svd_projection_attention` | 5 (SVD/LoRA Clone) | 322 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `svd_sequence_attention` | 5 (SVD/LoRA Clone) | 340 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `swift_svd` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `swift_svd_act` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `template_attention` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `tlinformer_attention` | 5 (SVD/LoRA Clone) | 340 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `token_compressed_attention` | 1 (Double Char) | 364 | 826 | 8 | 0 | `5c5f2b2da9` | **CONTAMINATED** |
| `tomoe` | 5 (SVD/LoRA Clone) | 330 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `tpa_attention` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `tt` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `unified_rope` | 3 (Placeholders) | 468 | 1040 | 2 | 0 | `9ed6d5d3df` | **CONTAMINATED** |
| `uniql_compressor` | 5 (SVD/LoRA Clone) | 348 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `upi_mamba` | 5 (SVD/LoRA Clone) | 322 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `upi_stuffed_mamba` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `vig_attention` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `vspec` | 3 (Placeholders) | 231 | 955 | 20 | 0 | `6d97a73a10` | **CONTAMINATED** |
| `weighted_ensemble` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `wgqa` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `xattention` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `zerokv_neo` | 5 (SVD/LoRA Clone) | 336 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |
| `zeros_attention` | 5 (SVD/LoRA Clone) | 328 | 1221 | 24 | 0 | `b5c2351545` | **TEMPLATE CLONE** |

---

## SECTION 13 — Forensic Summary

- **Current HEAD commit:** `732df3b0f` (on branch `main`)
- **Templated forensic branch:** `templated_2026_09_18_forensic` (pushed to `origin/templated_2026_09_18_forensic` at commit `8b109264e`)
- **Last clean commit:** `732df3b0f`
- **Total lines in templated state:** ~542,000 lines across 217 module papers
- **Lines reverted by rollback to 732df3b0f:** 517,890 lines
- **Files that MUST be preserved:**
  - `scripts/verify_module_paper.py` (with upgraded anti-template checks)
  - `scripts/ast_extractor.py` (source-grounded AST extractor)
  - `scripts/audit_template_artifacts.py` (audit engine)
  - `audit/TEMPLATE_ARTIFACT_AUDIT_2026_09_18/` (forensic database)
  - `audit/CONTENT_AUTHENTICITY_AUDIT.md` (initial evidence)
  - `audit/ROLLBACK_AND_REARCHITECT_PLAN.md` (re-architect roadmap)
  - `audit/FULL_TEMPLATE_ARTIFACT_AUDIT_REPORT.md` (master report)
- **Files that MUST be reverted:**
  - All 146 artificial template clone directories in `paper/PerModulePaper/` generated via `enhanced_sections.py`
  - The synthetic string injection logic in `scripts/enhanced_sections.py`

---

## SECTION 14 — Final Verdict and Recommendation

### Explicit Answers to User Questions:
1. **How many of the 217 modules are pure template clones?** **146 modules (67.3%)** share the identical 1,221-word SVD/proximal/2:4/LoRA boilerplate.
2. **How many are contaminated with artifacts?** **217 modules (100.0%)** contain at least one synthetic artifact (`0.85` repeated numbers, unrendered variables, broken double percent signs, or zero citations).
3. **How many are potentially valid?** **0 modules (0.0%)** have verified, publication-grade math grounded to their source code.
4. **Is a full rollback required?** **YES.** Immediate rollback of all synthetic generator output is necessary to purge ungrounded data.
5. **Which specific modules could be salvaged (if any)?** None of the generated text in `paper/PerModulePaper/` is publishable as-is. However, the benchmark CSV results in `results/unified_*/summary.csv` and the technical docstrings in `Codes/frontend/dist/Docs/content/` are authentic and will be salvaged.
6. **What is the minimum viable path forward?**
   - Maintain `main` at clean commit `732df3b0f`.
   - Wire `scripts/ast_extractor.py` and curated Markdown specs from `Codes/frontend/dist/Docs/content/` into the paper builder.
   - Require 100% equation citation coverage (`<!-- source: <file> line N -->`) before compiling any PDF.

---

## SECTION 15 — Evidence Files Inventory

| File Path | Size (Bytes) | Lines | SHA256 Hash |
|---|---:|---:|:---:|
| `scripts/audit_template_artifacts.py` | 7290 | 191 | `488b358b4c9132dc...` |
| `audit/TEMPLATE_ARTIFACT_AUDIT_2026_09_18/per_module_findings.json` | 333998 | 13270 | `ab55f97c742d4704...` |
| `audit/TEMPLATE_ARTIFACT_AUDIT_2026_09_18/per_module_summary.csv` | 8425 | 218 | `d092b21e03dbf498...` |
| `audit/TEMPLATE_ARTIFACT_AUDIT_2026_09_18/cross_module_overlap.csv` | 13532 | 191 | `2068cfac46bcf7dd...` |
| `audit/TEMPLATE_ARTIFACT_AUDIT_2026_09_18/run.log` | 13130 | 227 | `4098154b795f6247...` |
| `audit/TEMPLATE_ARTIFACT_AUDIT_2026_09_18/FINAL_AUDIT.md` | 6682 | 125 | `9a9f5802c2ea75bb...` |
| `audit/CONTENT_AUTHENTICITY_AUDIT.md` | 8378 | 137 | `31dde5162b96d5e9...` |
| `audit/ROLLBACK_AND_REARCHITECT_PLAN.md` | 10455 | 191 | `a561ee252d8accb6...` |
