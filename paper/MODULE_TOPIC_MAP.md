# Module-Topic Taxonomy Mapping

This document maps all 217 evaluated attention and compression modules in `paper/PerModulePaper/` to our 12-topic literature taxonomy and operational diagnostic category.

## Taxonomy Codes
- **T1**: Low-rank / SVD attention
- **T2**: Linear attention (kernel / cosformer / performer family)
- **T3**: State-space models (Mamba, S4, S5, HGRN, Griffin, Jamba)
- **T4**: Mixture-of-Experts (Switch, Expert Choice, Mixtral, GShard)
- **T5**: Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda)
- **T6**: Post-training low-rank adaptation (LoRA, QLoRA, AdaLoRA, DoRA)
- **T7**: KV-cache compression / paging
- **T8**: Quantization (GPTQ, AWQ, FP8, SmoothQuant)
- **T9**: Efficient attention variants (MLA, NSA, FlashAttention, Halo)
- **T10**: Hybrid architectures (attention + SSM, MoE + SSM)
- **T11**: BERT/NLI-specific efficiency (DistilBERT, MiniLM, TinyBERT)
- **T12**: Other / Specialized Efficiency Operators

## Complete Module Mapping Table

| # | Module | Topic | Category | Priority | Topic Description |
|---|--------|:-----:|:--------:|:--------:|-------------------|
| 1 | `AdaptiveSVD` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 2 | `AdaptiveSpanAttention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 3 | `AdaptiveSparseAttention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 4 | `Galerkin` | T2 | Cat A | P2 | Linear attention (kernel / cosformer / performer family) |
| 5 | `HALO` | T9 | PASS | P3 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 6 | `OH` | T9 | PASS | P3 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 7 | `Primal` | T9 | PASS | P3 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 8 | `SOAFusedAttention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 9 | `SqueezingHeadsAttentionDistillationLayer` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 10 | `a3` | T1 | PASS | P1 | Low-rank / SVD attention |
| 11 | `aa_svd_compressor` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 12 | `aasvd` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 13 | `adaptive_compressor` | T1 | PASS | P3 | Low-rank / SVD attention |
| 14 | `adaptive_ranks` | T1 | PASS | P3 | Low-rank / SVD attention |
| 15 | `adaptive_span_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 16 | `adaptive_sparse_attention` | T8 | Cat C | P1 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 17 | `adaptive_svd_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 18 | `adasplash` | T9 | Cat A | P2 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 19 | `adasvd_compressor` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 20 | `admm_compressor` | T5 | PASS | P3 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 21 | `admm_pruner` | T5 | PASS | P1 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 22 | `aegis_hybrid` | T10 | PASS | P1 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 23 | `aegis_spectron_fp8` | T8 | PASS_LOW | P1 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 24 | `aegis_v1` | T1 | Cat B | P2 | Low-rank / SVD attention |
| 25 | `aegis_v2` | T1 | PASS | P1 | Low-rank / SVD attention |
| 26 | `aegis_v3` | T1 | PASS | P1 | Low-rank / SVD attention |
| 27 | `aegis_v4` | T1 | PASS | P1 | Low-rank / SVD attention |
| 28 | `aegis_v5` | T1 | PASS | P1 | Low-rank / SVD attention |
| 29 | `aegis_v6` | T1 | PASS | P1 | Low-rank / SVD attention |
| 30 | `armor_pruner` | T5 | PASS | P3 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 31 | `ast_pruner` | T5 | PASS | P3 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 32 | `astro_quantizer` | T5 | PASS | P3 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 33 | `asvd` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 34 | `bam_umoe` | T4 | Cat C | P1 | Mixture-of-Experts (Switch, Expert Choice, Mixtral, GShard) |
| 35 | `block_transformer_hiba` | T9 | Cat A | P2 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 36 | `bonsai_pruner` | T5 | Cat A | P2 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 37 | `cachenotes_compressor` | T7 | PASS | P3 | KV-cache compression / paging |
| 38 | `chela_slay` | T2 | Cat A | P2 | Linear attention (kernel / cosformer / performer family) |
| 39 | `cluster` | T2 | Cat A | P2 | Linear attention (kernel / cosformer / performer family) |
| 40 | `come` | T1 | PASS | P1 | Low-rank / SVD attention |
| 41 | `compot_compressor` | T1 | PASS | P3 | Low-rank / SVD attention |
| 42 | `compressed_attention` | T8 | PASS_LOW | P1 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 43 | `compressed_mha` | T9 | PASS_LOW | P1 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 44 | `coopq_quantizer` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 45 | `cosformer` | T2 | Cat A | P2 | Linear attention (kernel / cosformer / performer family) |
| 46 | `cr_compressor` | T1 | PASS | P3 | Low-rank / SVD attention |
| 47 | `cur_compressor` | T1 | PASS | P3 | Low-rank / SVD attention |
| 48 | `cur_linear` | T1 | PASS | P3 | Low-rank / SVD attention |
| 49 | `cur_moe` | T4 | PASS | P1 | Mixture-of-Experts (Switch, Expert Choice, Mixtral, GShard) |
| 50 | `cure` | T1 | PASS | P3 | Low-rank / SVD attention |
| 51 | `decay_linear_attention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 52 | `delta_kv` | T7 | PASS | P1 | KV-cache compression / paging |
| 53 | `direct_svd` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 54 | `dobi` | T1 | PASS | P1 | Low-rank / SVD attention |
| 55 | `dobi_svd` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 56 | `dynamic_entropy` | T1 | PASS | P3 | Low-rank / SVD attention |
| 57 | `dynamic_rank` | T1 | PASS | P1 | Low-rank / SVD attention |
| 58 | `eat` | T9 | Cat A | P2 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 59 | `enlighten_compressor` | T11 | PASS | P3 | BERT/NLI-specific efficiency (DistilBERT, MiniLM, TinyBERT) |
| 60 | `entropy_budget_allocation` | T1 | PASS | P3 | Low-rank / SVD attention |
| 61 | `erc_svd` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 62 | `expert_choice_moe` | T4 | Cat C | P1 | Mixture-of-Experts (Switch, Expert Choice, Mixtral, GShard) |
| 63 | `factorised_matrix_state` | T1 | PASS | P3 | Low-rank / SVD attention |
| 64 | `faq_quantizer` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 65 | `flap` | T5 | PASS | P1 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 66 | `flash3` | T9 | Cat A | P2 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 67 | `flash_attention3` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 68 | `flash_attention3_fp8` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 69 | `flat_llm` | T1 | PASS | P1 | Low-rank / SVD attention |
| 70 | `frandi` | T1 | PASS | P1 | Low-rank / SVD attention |
| 71 | `full_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 72 | `fused_svd_kernel` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 73 | `fusion` | T12 | PASS | P3 | Other / Specialized Efficiency Operators |
| 74 | `fusion_attn` | T8 | Cat B | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 75 | `galerkin_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 76 | `gated_delta_net` | T12 | PASS | P1 | Other / Specialized Efficiency Operators |
| 77 | `gated_deltanet_block` | T3 | PASS | P1 | State-space models (Mamba, S4, S5, HGRN, Griffin, Jamba) |
| 78 | `gated_linear_attention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 79 | `gfws_d_compressor` | T1 | PASS | P3 | Low-rank / SVD attention |
| 80 | `goldfinch_ahn` | T10 | PASS | P3 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 81 | `gqa` | T9 | Cat B | P2 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 82 | `granite_hybrid` | T10 | PASS | P1 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 83 | `grecodec` | T1 | PASS | P1 | Low-rank / SVD attention |
| 84 | `griffin_ssd` | T3 | Cat B | P2 | State-space models (Mamba, S4, S5, HGRN, Griffin, Jamba) |
| 85 | `halo_regla` | T9 | Cat A | P2 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 86 | `hamming_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 87 | `hashed_linear` | T2 | PASS | P3 | Linear attention (kernel / cosformer / performer family) |
| 88 | `hashed_net_compressor` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 89 | `hgrn2_gated_deltanet` | T3 | PASS | P1 | State-space models (Mamba, S4, S5, HGRN, Griffin, Jamba) |
| 90 | `hgrn2_layer` | T3 | PASS | P3 | State-space models (Mamba, S4, S5, HGRN, Griffin, Jamba) |
| 91 | `hgrn2_outer` | T3 | Cat A | P2 | State-space models (Mamba, S4, S5, HGRN, Griffin, Jamba) |
| 92 | `hla_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 93 | `huffman_coder` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 94 | `hybrid_svd_layer` | T10 | Cat A | P2 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 95 | `hybridkv` | T10 | PASS | P1 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 96 | `hymba_transx` | T10 | Cat C | P1 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 97 | `infini` | T1 | Cat B | P2 | Low-rank / SVD attention |
| 98 | `jamba` | T10 | Cat C | P1 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 99 | `jamba_info` | T10 | Cat C | P1 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 100 | `kda_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 101 | `kmeans_quantize` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 102 | `kronecker_compressor` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 103 | `kronecker_linear` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 104 | `kronecker_tt_lora` | T8 | Cat B | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 105 | `kvtc` | T7 | PASS | P1 | KV-cache compression / paging |
| 106 | `leka_v1` | T2 | Cat A | P2 | Linear attention (kernel / cosformer / performer family) |
| 107 | `leka_v2` | T2 | Cat B | P2 | Linear attention (kernel / cosformer / performer family) |
| 108 | `leka_v3` | T2 | Cat B | P2 | Linear attention (kernel / cosformer / performer family) |
| 109 | `liftr` | T11 | PASS | P1 | BERT/NLI-specific efficiency (DistilBERT, MiniLM, TinyBERT) |
| 110 | `lillama_compressor` | T1 | PASS | P3 | Low-rank / SVD attention |
| 111 | `linformer_attention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 112 | `lola_attention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 113 | `lora_attention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 114 | `lsc_compressor` | T1 | PASS | P3 | Low-rank / SVD attention |
| 115 | `lsh_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 116 | `mamba` | T3 | PASS_LOW | P1 | State-space models (Mamba, S4, S5, HGRN, Griffin, Jamba) |
| 117 | `mamba_inject` | T3 | PASS | P3 | State-space models (Mamba, S4, S5, HGRN, Griffin, Jamba) |
| 118 | `mamba_shedder` | T3 | PASS | P1 | State-space models (Mamba, S4, S5, HGRN, Griffin, Jamba) |
| 119 | `mamba_svd_flash` | T3 | Cat A | P2 | State-space models (Mamba, S4, S5, HGRN, Griffin, Jamba) |
| 120 | `mgie_svd` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 121 | `minimal_aurora` | T1 | PASS | P1 | Low-rank / SVD attention |
| 122 | `minimal_nova` | T1 | PASS | P1 | Low-rank / SVD attention |
| 123 | `minimal_prism` | T1 | PASS | P1 | Low-rank / SVD attention |
| 124 | `minimal_svd` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 125 | `minimal_svd_v2` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 126 | `mixattention_moska` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 127 | `mla_attention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 128 | `mlorq_compressor` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 129 | `moba` | T9 | Cat A | P2 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 130 | `modegpt` | T1 | PASS | P1 | Low-rank / SVD attention |
| 131 | `moe_deltakv_paged` | T4 | Cat B | P2 | Mixture-of-Experts (Switch, Expert Choice, Mixtral, GShard) |
| 132 | `nasc_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 133 | `nexus` | T2 | Cat A | P2 | Linear attention (kernel / cosformer / performer family) |
| 134 | `nha` | T2 | Cat A | P2 | Linear attention (kernel / cosformer / performer family) |
| 135 | `nli_softmax` | T11 | Cat A | P2 | BERT/NLI-specific efficiency (DistilBERT, MiniLM, TinyBERT) |
| 136 | `nmf_compressor` | T1 | PASS | P3 | Low-rank / SVD attention |
| 137 | `nsa` | T9 | Cat A | P2 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 138 | `nystrom_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 139 | `oats` | T1 | PASS | P1 | Low-rank / SVD attention |
| 140 | `optimal_subspace_projection` | T1 | PASS | P1 | Low-rank / SVD attention |
| 141 | `optimized_hybrid_attention` | T10 | PASS | P3 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 142 | `orion` | T1 | PASS | P1 | Low-rank / SVD attention |
| 143 | `padre_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 144 | `performer_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 145 | `phantom` | T1 | PASS_LOW | P1 | Low-rank / SVD attention |
| 146 | `progressive_token_pruning` | T5 | PASS | P3 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 147 | `projection_factorized_attn` | T8 | PASS | P1 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 148 | `proxsparse` | T5 | PASS | P1 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 149 | `proxsparse_pruner` | T5 | PASS | P3 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 150 | `pruning_utils` | T5 | PASS | P3 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 151 | `ptp` | T11 | Cat B | P2 | BERT/NLI-specific efficiency (DistilBERT, MiniLM, TinyBERT) |
| 152 | `qcqa_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 153 | `rala_attention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 154 | `reap_moe` | T4 | Cat C | P1 | Mixture-of-Experts (Switch, Expert Choice, Mixtral, GShard) |
| 155 | `regla_attention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 156 | `replace` | T5 | PASS | P1 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 157 | `replaceme_compressor` | T5 | PASS | P3 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 158 | `res_svd` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 159 | `roast_hashed_linear` | T5 | PASS | P3 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 160 | `rotational_slicegpt_int4` | T5 | PASS | P3 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 161 | `routing_attention` | T4 | Cat A | P2 | Mixture-of-Experts (Switch, Expert Choice, Mixtral, GShard) |
| 162 | `rpc` | T2 | Cat A | P2 | Linear attention (kernel / cosformer / performer family) |
| 163 | `rpca` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 164 | `saes_svd` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 165 | `saes_svd_v2` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 166 | `salr` | T6 | PASS | P1 | Post-training low-rank adaptation (LoRA, QLoRA, AdaLoRA, DoRA) |
| 167 | `salt` | T9 | PASS | P1 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 168 | `samba_m3` | T10 | Cat C | P1 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 169 | `samba_moe` | T10 | Cat C | P1 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 170 | `sdrm` | T1 | FAIL | P1 | Low-rank / SVD attention |
| 171 | `simple_ssm` | T3 | Cat B | P2 | State-space models (Mamba, S4, S5, HGRN, Griffin, Jamba) |
| 172 | `slicegpt` | T5 | PASS | P1 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 173 | `slicegpt_pca` | T5 | PASS | P1 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 174 | `slim_compressor` | T5 | PASS | P3 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 175 | `smart_weighted_ensemble` | T9 | Cat A | P2 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 176 | `soa_attn` | T8 | PASS | P1 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 177 | `soa_fused_module` | T9 | PASS | P3 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 178 | `soa_module` | T9 | PASS | P3 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 179 | `soka` | T1 | PASS | P1 | Low-rank / SVD attention |
| 180 | `span_expanded_attention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 181 | `sparsegpt` | T5 | PASS | P1 | Structured pruning / LLM compression (SliceGPT, SparseGPT, Wanda) |
| 182 | `spectra_v2` | T1 | Cat B | P2 | Low-rank / SVD attention |
| 183 | `spectraformer_attention` | T10 | PASS | P3 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 184 | `spectral_mamba` | T10 | FAIL | P1 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 185 | `spectron` | T9 | PASS | P1 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 186 | `squeezing_heads` | T11 | Cat A | P2 | BERT/NLI-specific efficiency (DistilBERT, MiniLM, TinyBERT) |
| 187 | `squeezing_heads_distill` | T11 | Cat A | P2 | BERT/NLI-specific efficiency (DistilBERT, MiniLM, TinyBERT) |
| 188 | `squeezing_heads_distiller` | T11 | Cat A | P2 | BERT/NLI-specific efficiency (DistilBERT, MiniLM, TinyBERT) |
| 189 | `subspace_projection` | T1 | PASS | P1 | Low-rank / SVD attention |
| 190 | `svd_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 191 | `svd_compressed` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 192 | `svd_linear_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 193 | `svd_linear_decay` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 194 | `svd_llm_v2_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 195 | `svd_mamba` | T3 | Cat A | P2 | State-space models (Mamba, S4, S5, HGRN, Griffin, Jamba) |
| 196 | `svd_proj_attn` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 197 | `svd_projection_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 198 | `svd_sequence_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 199 | `swift_svd` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 200 | `swift_svd_act` | T1 | Cat A | P2 | Low-rank / SVD attention |
| 201 | `template_attention` | T8 | Cat B | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 202 | `tlinformer_attention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 203 | `token_compressed_attention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 204 | `tomoe` | T4 | PASS | P1 | Mixture-of-Experts (Switch, Expert Choice, Mixtral, GShard) |
| 205 | `tpa_attention` | T10 | Cat C | P1 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 206 | `tt` | T8 | PASS | P1 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 207 | `unified_rope` | T2 | Cat A | P2 | Linear attention (kernel / cosformer / performer family) |
| 208 | `uniql_compressor` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 209 | `upi_mamba` | T10 | Cat B | P2 | Hybrid architectures (attention + SSM, MoE + SSM) |
| 210 | `upi_stuffed_mamba` | T3 | PASS | P3 | State-space models (Mamba, S4, S5, HGRN, Griffin, Jamba) |
| 211 | `vig_attention` | T8 | PASS | P3 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 212 | `vspec` | T9 | PASS | P1 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 213 | `weighted_ensemble` | T9 | Cat A | P2 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 214 | `wgqa` | T9 | Cat A | P2 | Efficient attention variants (MLA, NSA, FlashAttention, Halo) |
| 215 | `xattention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
| 216 | `zerokv_neo` | T7 | PASS | P3 | KV-cache compression / paging |
| 217 | `zeros_attention` | T8 | Cat A | P2 | Quantization (GPTQ, AWQ, FP8, SmoothQuant) |
