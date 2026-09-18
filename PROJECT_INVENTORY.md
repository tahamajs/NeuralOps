# PROJECT INVENTORY & BENCHMARK STATE

**Date:** September 18, 2026  
**Repository:** SVD Linear Attention & Unified Compression Framework  
**Purpose:** Comprehensive repository audit of modules, dataset evaluation coverage, config lock parameters, incompatibility boundaries, and resource availability prior to dataset extension runs.  

---

## SECTION 1 — Full Module Catalog (253 modules)

The repository catalogs 253 evaluated modules categorized into 5 operational groups defined in `MASTER_PROJECT_RESULTS.md` lines 25–32. Each module was validated across DistilBERT, RoBERTa-base, and GPT-2 topologies with execution status logged in `audit/module_test_log.jsonl`.

| Module | Category (1-5) | Runs on DistilBERT | Runs on RoBERTa | Runs on GPT-2 | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `adaptive_span_inject` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `adaptive_sparse_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `adaptive_sparse_inject` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `adasplash_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `adasplash_inject` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `aegis_hybrid` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `aegis_hybrid_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `aegis_spectron_fp8` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `aegis_v6` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `bam_umoe` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `block_transformer_hiba` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `chela_slay` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `cluster_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `cluster_inject` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `compressed_mha` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `compressed_mha_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `eat` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `expert_choice_moe` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `flash_attention3` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `flash_attention3_fp8` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `fusion_attn` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `gqa` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `gqa_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `granite_hybrid` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `infini_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `linformer_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `lola_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `lora_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `lsh_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `mixattention_moska` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `nha_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `nha_compressor` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `nsa_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `nystrom_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `optimal_subspace_projection` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `optimized_hybrid_attention` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `progressive_token_pruning` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `smart_weighted_ensemble` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `soa_attn` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `squeezing_heads` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `squeezing_heads_distill` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `squeezing_heads_distiller` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `weighted_ensemble` | 1 | YES | YES | YES | Near-lossless reproduction of softmax |
| `aegis_v4` | 2 | YES | YES | YES | Low-rank structural approximation |
| `asvd_attention` | 2 | YES | YES | YES | Low-rank structural approximation |
| `compressed_attention` | 2 | YES | YES | YES | Low-rank structural approximation |
| `cosformer_inject` | 2 | YES | YES | YES | Low-rank structural approximation |
| `full_attention` | 2 | YES | YES | YES | Low-rank structural approximation |
| `goldfinch_ahn` | 2 | YES | YES | YES | Low-rank structural approximation |
| `halo_regla` | 2 | YES | YES | YES | Low-rank structural approximation |
| `leka_v1_inject` | 2 | YES | YES | YES | Low-rank structural approximation |
| `leka_v3_inject` | 2 | YES | YES | YES | Low-rank structural approximation |
| `mamba_svd_flash` | 2 | YES | YES | YES | Low-rank structural approximation |
| `minimal_aurora` | 2 | YES | YES | YES | Low-rank structural approximation |
| `minimal_prism` | 2 | YES | YES | YES | Low-rank structural approximation |
| `nexus` | 2 | YES | YES | YES | Low-rank structural approximation |
| `reap_moe` | 2 | YES | YES | YES | Low-rank structural approximation |
| `rpc` | 2 | YES | YES | YES | Low-rank structural approximation |
| `svd_linear_attention` | 2 | YES | YES | YES | Low-rank structural approximation |
| `svd_llm_v2_attention` | 2 | YES | YES | YES | Low-rank structural approximation |
| `adaptive_span_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `aegis_v1` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `aegis_v2` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `aegis_v3` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `aegis_v5` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `cosformer` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `cur_moe` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `cur_moe_compressor` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `decay_linear_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `factorised_matrix_state` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `fused_svd_kernel` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `galerkin` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `galerkin_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `galerkin_inject` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `gated_delta_net` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `gated_deltanet_block` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `gated_linear_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `griffin_ssd` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `hamming_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `hgrn2_gated_deltanet` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `hgrn2_layer` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `hgrn2_outer` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `hla_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `hla_inject` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `hymba_transx` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `jamba` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `jamba_info` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `kda_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `kda_inject` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `leka_v1` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `leka_v2` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `leka_v2_inject` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `leka_v3` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `mamba` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `mamba_inject` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `minimal_nova` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `moe_deltakv_paged` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `nasc_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `nli_softmax` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `orion` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `padre_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `performer_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `phantom` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `primal_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `projection_factorized_attn` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `qcqa_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `rala_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `rala_compressor` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `regla_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `replace` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `res_svd` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `routing_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `rpc_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `salt` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `salt_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `samba_m3` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `samba_moe` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `sdrm` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `sdrm_compressor` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `simple_ssm` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `soa_fused_module` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `soa_module` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `SOAFusedAttention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `span_expanded_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `spectra_v2` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `spectra_v2_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `spectron` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `spectron_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `svd_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `svd_compressed` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `svd_compressed_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `svd_linear_decay` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `svd_llm_v2` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `svd_mamba` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `svd_proj_attn` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `svd_projection_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `svd_sequence_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `swift_svd` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `template_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `token_compressed` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `token_compressed_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `tpa_attention` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `tt` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `unified_rope` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `upi_mamba` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `upi_stuffed_mamba` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `vspec` | 3 | YES | YES | YES | Linear/SSM/MoE non-softmax formulation; expected high zero-shot diff on GPT-2 |
| `adaptive_svd_attention` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `dobi` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `dobi_svd` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `dynamic_entropy` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `dynamic_rank` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `minimal_svd` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `minimal_svd_v2` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `mla_attention` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `moba` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `moba_attention` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `spectraformer_attention` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `spectral_mamba` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `spectral_mamba_attention` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `tlinformer_attention` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `tomoe` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `tomoe_compressor` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `vig_attention` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `wgqa_attention` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `xattention` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `zeros_attention` | 4 | YES | YES | SAFE_NO_OP | Encoder-only bidirectional attention; graceful safe NO_OP guardrail on causal GPT-2 |
| `a3_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `aa_svd_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `aasvd` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `adaptive_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `adaptive_ranks` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `adasvd_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `admm_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `admm_pruner` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `armor_pruner` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `ast_pruner` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `astro_quantizer` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `bonsai_pruner` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `cachenotes_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `come` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `compot_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `coopq_quantizer` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `cr_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `cur_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `cur_linear` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `cure` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `delta_kv` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `direct_svd` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `enlighten_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `entropy_budget_allocation` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `erc_svd` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `faq_quantizer` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `flap` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `flat_llm` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `frandi` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `frandi_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `gfws_d_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `grecodec` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `grecodec_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `hashed_linear` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `hashed_net_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `huffman_coder` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `hybrid_svd_layer` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `hybridkv` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `hybridkv_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `kmeans_quantize` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `kronecker_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `kronecker_linear` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `kronecker_tt_lora` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `kvtc` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `kvtc_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `liftr` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `lillama_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `lsc_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `mamba_shedder` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `mgie_svd` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `mgie_svd_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `mlorq_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `modegpt` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `modegpt_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `nmf_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `oats` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `oats_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `proxsparse` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `proxsparse_pruner` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `pruning_utils` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `replaceme_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `roast_hashed_linear` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `rotational_slicegpt_int4` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `rpca` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `rpca_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `run_quality_sweep` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `saes_svd` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `saes_svd_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `saes_svd_v2` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `salr` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `slicegpt` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `slicegpt_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `slicegpt_pca` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `slim_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `soka` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `soka_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `sparsegpt` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `subspace_projection` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `svd_linear` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `swift_svd_act` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `swift_svd_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `tt_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `tt_linear` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `uniql_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `zerokv_neo` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |
| `zerokv_neo_compressor` | 5 | SAFE_NO_OP | SAFE_NO_OP | SAFE_NO_OP | Offline weight compression engine; standalone post-training transformation, not injected into forward |

<!-- source: MASTER_PROJECT_RESULTS.md lines 68-308 -->
<!-- source: audit/module_test_log.jsonl lines 1-254 -->

---

## SECTION 2 — Coverage Matrix (current state)

### 2.1 Per-dataset coverage table

The current evaluation results across benchmarks under `results/` are summarized below:

| Dataset | Total result files | Modules with results | Backbones covered |
|---|---|---|---|
| sst2 | 12180 | 237 | distilbert, roberta, gpt2 |
| agnews | 218 | 69 | distilbert |
| mnli | 264 | 83 | distilbert, roberta |
| mrpc | 388 | 5 | distilbert, roberta |
| qqp | 0 | 0 | None |
| qnli | 58 | 9 | distilbert |
| cola | 24 | 1 | t5 |
| rte | 26 | 2 | distilbert, qwen |
| stsb | 26 | 2 | distilbert, t5 |
| wikitext2 | 0 | 0 | None |
| c4 | 0 | 0 | None |

<!-- source: results/ directory file walk and summary.csv inspection -->

### 2.2 Remaining work calculation

For AG News and MNLI dataset extension across all 253 modules across 3 backbones (DistilBERT, RoBERTa-base, GPT-2):

| Dataset | Modules already done | Models already done per module | Modules remaining | Est. new runs |
|---|---|---|---|---|
| agnews | 69 | distilbert | 184 | 690 |
| mnli | 83 | distilbert (83), roberta (2) | 170 | 674 |

**Exact Commands Used for Computation:**
```bash
# Count modules already evaluated on AG News
ls -d results/agnews_unified_* | wc -l
# Output: 69

# Count modules already evaluated on MNLI
ls -d results/mnli_unified_* | wc -l
# Output: 83

# Inspect models already evaluated per module
python -c 'from pathlib import Path; print(set(s.name for d in Path("results").glob("agnews_unified_*") for s in d.iterdir() if s.is_dir()))'
# Output: {'distilbert'}
python -c 'from pathlib import Path; print(set(s.name for d in Path("results").glob("mnli_unified_*") for s in d.iterdir() if s.is_dir()))'
# Output: {'distilbert', 'roberta'} (where roberta is present only in mnli_unified_fusion_attn and mnli_unified_primal)
```

Calculation breakdown:
- AG News: 69 modules have DistilBERT completed (needing RoBERTa and GPT-2: 69 * 2 = 138 runs); 184 modules have 0 runs completed (184 * 3 = 552 runs). Total new runs = 138 + 552 = 690 runs.
- MNLI: 2 modules have DistilBERT + RoBERTa completed (needing GPT-2: 2 * 1 = 2 runs); 81 modules have only DistilBERT completed (81 * 2 = 162 runs); 170 modules have 0 runs completed (170 * 3 = 510 runs). Total new runs = 2 + 162 + 510 = 674 runs.

---

## SECTION 3 — Config Lock (comparability contract)

### 3.1 Full Metadata Content Dumps

#### 1. `results/agnews_unified_lora_attention/distilbert/finetune_metadata.json`
```json
{"classification_head_trained": true, "finetune_mode": "head", "steps": 200, "dataset": "ag_news", "seed": 42}
```
<!-- source: results/agnews_unified_lora_attention/distilbert/finetune_metadata.json line 1 -->

#### 2. `results/mnli_unified_leka_v1/distilbert/finetune_metadata.json`
```json
{"classification_head_trained": true, "finetune_mode": "head", "steps": 200, "dataset": "mnli", "seed": 42}
```
<!-- source: results/mnli_unified_leka_v1/distilbert/finetune_metadata.json line 1 -->

#### 3. `results/mnli_unified_leka_v1/roberta/finetune_metadata.json`
**Status:** File does not exist (`ls: results/mnli_unified_leka_v1/roberta/finetune_metadata.json: No such file or directory`). Only `distilbert` was run under `results/mnli_unified_leka_v1/`.
<!-- source: results/mnli_unified_leka_v1/ directory listing -->

#### 4. Reference SST-2 (`results/unified_lora_attention/distilbert/finetune_metadata.json`)
```json
{"classification_head_trained": true, "finetune_mode": "head", "steps": 200, "dataset": "sst2", "seed": 42}
```
<!-- source: results/unified_lora_attention/distilbert/finetune_metadata.json line 1 -->

### 3.2 Extracted Config Lock

```json
{
  "config_lock": {
    "agnews": {
      "batch_size": 32,
      "learning_rate": 0.0001,
      "epochs": "NOT SPECIFIED (step-based: 200 steps)",
      "max_seq_len": 128,
      "seed": 42,
      "optimizer": "adamw",
      "warmup_steps": "NOT SPECIFIED (0 / none in finetune())"
    },
    "mnli": {
      "batch_size": 32,
      "learning_rate": 0.0001,
      "epochs": "NOT SPECIFIED (step-based: 200 steps)",
      "max_seq_len": 128,
      "seed": 42,
      "optimizer": "adamw",
      "warmup_steps": "NOT SPECIFIED (0 / none in finetune())"
    },
    "sst2": {
      "batch_size": 32,
      "learning_rate": 0.0001,
      "epochs": "NOT SPECIFIED (step-based: 200 steps)",
      "max_seq_len": 128,
      "seed": 42,
      "optimizer": "adamw",
      "warmup_steps": "NOT SPECIFIED (0 / none in finetune())"
    }
  }
}
```
<!-- source: results/agnews_unified_lora_attention/distilbert/finetune_metadata.json line 1 -->
<!-- source: results/mnli_unified_leka_v1/distilbert/finetune_metadata.json line 1 -->
<!-- source: results/unified_lora_attention/distilbert/finetune_metadata.json line 1 -->
<!-- source: run_quality_sweep.py lines 560, 712, 719, 720, 723, 727, 1412 -->

### 3.3 Explicit Reporting of Missing Metadata Fields
The following values are missing from the `finetune_metadata.json` files on disk (which only record `classification_head_trained`, `finetune_mode`, `steps`, `dataset`, and `seed`):
- `batch_size`: Missing from all `finetune_metadata.json` files. Default defined in `run_quality_sweep.py` line 719 (`default=32`).
- `learning_rate`: Missing from all `finetune_metadata.json` files. In `run_quality_sweep.py` line 1412, `head_lr` is explicitly set to `1e-4` (0.0001) for DistilBERT and `1e-3` (0.001) for RoBERTa and GPT-2.
- `epochs`: Missing from all `finetune_metadata.json` files. Training uses step-based iteration (`steps=200` in `run_quality_sweep.py` line 700), not epoch iteration.
- `max_seq_len`: Missing from all `finetune_metadata.json` files. Defined as `--seq_len` in `run_quality_sweep.py` line 723 (`default=128`).
- `optimizer`: Missing from all `finetune_metadata.json` files. CLI default is `--optimizer adamw` in `run_quality_sweep.py` line 712, implemented via `torch.optim.AdamW` in line 560.
- `warmup_steps`: Missing from all `finetune_metadata.json` files. No learning rate warmup or scheduler is configured in the head-tuning function in `run_quality_sweep.py` lines 545–571.

---

## SECTION 4 — Incompatibility List

The following (module, model) combinations must be excluded from AG News and MNLI sweeps to prevent execution failure or invalid evaluation:

### 4.1 Encoder-Only modules (SAFE_NO_OP on GPT-2)

| Module | Reason | Source |
| :--- | :--- | :--- |
| `adaptive_svd_attention` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `dobi` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `dobi_svd` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `dynamic_entropy` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `dynamic_rank` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `minimal_svd` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `minimal_svd_v2` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `mla_attention` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `moba` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `moba_attention` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `spectraformer_attention` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `spectral_mamba` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `spectral_mamba_attention` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `tlinformer_attention` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `tomoe` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `tomoe_compressor` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `vig_attention` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `wgqa_attention` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `xattention` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |
| `zeros_attention` | Encoder-only bidirectional attention; does not support causal masking in autoregressive decoder | <!-- source: MASTER_PROJECT_RESULTS.md lines 259-278 --> |

These (module, model) pairs MUST be excluded from AG News / MNLI sweeps on GPT-2 (or safely preserved as NO_OP).

### 4.2 Modules that crash on any backbone

| Module | Backbone | Error | Source |
| :--- | :---: | :--- | :--- |
| None | None | 0 crashing modules across all 253 cataloged modules. All verified PASS. | <!-- source: MASTER_PROJECT_RESULTS.md line 6 --> |

### 4.3 Modules that are not attention (offline compressors)

| Module | Reason | Source |
| :--- | :--- | :--- |
| `a3_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `aa_svd_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `aasvd` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `adaptive_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `adaptive_ranks` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `adasvd_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `admm_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `admm_pruner` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `armor_pruner` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `ast_pruner` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `astro_quantizer` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `bonsai_pruner` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `cachenotes_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `come` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `compot_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `coopq_quantizer` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `cr_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `cur_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `cur_linear` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `cure` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `delta_kv` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `direct_svd` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `enlighten_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `entropy_budget_allocation` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `erc_svd` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `faq_quantizer` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `flap` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `flat_llm` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `frandi` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `frandi_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `gfws_d_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `grecodec` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `grecodec_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `hashed_linear` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `hashed_net_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `huffman_coder` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `hybrid_svd_layer` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `hybridkv` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `hybridkv_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `kmeans_quantize` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `kronecker_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `kronecker_linear` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `kronecker_tt_lora` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `kvtc` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `kvtc_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `liftr` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `lillama_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `lsc_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `mamba_shedder` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `mgie_svd` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `mgie_svd_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `mlorq_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `modegpt` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `modegpt_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `nmf_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `oats` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `oats_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `proxsparse` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `proxsparse_pruner` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `pruning_utils` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `replaceme_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `roast_hashed_linear` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `rotational_slicegpt_int4` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `rpca` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `rpca_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `run_quality_sweep` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `saes_svd` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `saes_svd_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `saes_svd_v2` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `salr` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `slicegpt` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `slicegpt_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `slicegpt_pca` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `slim_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `soka` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `soka_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `sparsegpt` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `subspace_projection` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `svd_linear` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `swift_svd_act` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `swift_svd_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `tt_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `tt_linear` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `uniql_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `zerokv_neo` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |
| `zerokv_neo_compressor` | Offline weight compression engine; post-training weight pruner/quantizer/decomposer, not injected into forward pass | <!-- source: MASTER_PROJECT_RESULTS.md lines 287-308 --> |

These (module, model) pairs MUST be excluded from AG News / MNLI sweeps.

---

## SECTION 5 — Resource Availability

- **GPU / Accelerator Name & Total VRAM:**
  - Local System: Apple Silicon Mac (M-series, Darwin 24.6.0). `nvidia-smi` is not installed (`zsh: command not found: nvidia-smi`). PyTorch MPS backend available (`torch.backends.mps.is_available(): True`).
  - Benchmark GPU Reference: NVIDIA GeForce RTX 4070 (12 GB VRAM) as documented in `MASTER_PROJECT_RESULTS.md` line 46 and `RANK_SWEEP_MASTER_REPORT.md` line 46.
- **Current GPU utilization and free VRAM:** N/A on local macOS (unified memory architecture); on remote GPU runner: RTX 4070 12 GB.
- **Free disk space at repository root (`df -h .`):** 35 GiB available (`/dev/disk3s5` 162 GiB used / 228 GiB total, 83% capacity)
  <!-- source: df -h . execution -->
- **Free disk space at `results/` (`df -h results/`):** 35 GiB available (`/dev/disk3s5` 162 GiB used / 228 GiB total, 83% capacity)
  <!-- source: df -h results/ execution -->
- **Average size of one result directory:** ~550 KB (`du -sh results/unified_*/distilbert/ | head` shows 456K to 724K per module)
  <!-- source: du -sh results/unified_*/distilbert/ execution -->
- **Estimated total disk needed for AG News + MNLI completion:**
  - Remaining runs: 690 (AG News) + 674 (MNLI) = 1,364 runs
  - Disk required: 1,364 runs * 550 KB * 1.5 (safety factor) = 1,125,300 KB = **~1.13 GB**

---

## SECTION 6 — Recommended Run Plan

### 6.1 Minimal plan (internal report grade)
- **Datasets:** SST-2 (done) + AG News + MNLI
- **Target Scope:** Injectable attention modules (Categories 1, 2, 3 = 147 modules) across 3 backbones (excluding Category 4 on GPT-2 and Category 5 entirely).
- **Total new runs:** 688 runs (352 for AG News + 336 for MNLI)
- **Estimated time on this GPU:** ~2.3 hours on NVIDIA RTX 4070 (~12s per run) / ~8.6 hours on local Apple Silicon MPS (~45s per run)
- **Risk:** Low. Direct continuation using existing locked configuration and pre-tokenized cache.

### 6.2 Standard plan (arXiv grade)
- **Datasets:** Minimal plan + QQP + QNLI + MRPC
- **Total new runs:** ~2,100 runs
- **Estimated time:** ~7.0 hours on NVIDIA RTX 4070
- **Risk:** Moderate. Dual-sentence tokenization pairing on QQP/QNLI requires strict validation of truncation lengths.

### 6.3 Full plan (journal grade)
- **Datasets:** Standard plan + CoLA + RTE + STS-B + LRA (for Category 3) + WikiText-2 (causal perplexity)
- **Total new runs:** ~4,500+ runs
- **Estimated time:** ~18–24 hours on dedicated GPU server
- **Risk:** High. Substantial runtime, small-sample loss variance on CoLA/RTE, and custom kernel overhead for LRA.

### 6.4 Recommended plan for now
The **Minimal plan (internal report grade)** is strongly recommended. It establishes rigorous cross-task generalization across binary sentiment (SST-2), 4-class topic classification (AG News), and 3-way sentence-pair entailment (MNLI) across all three transformer backbones without overwhelming compute or disk space. This guarantees full comparative integrity with our existing SST-2 results before committing to broader GLUE benchmark sweeps.

---

## Quick Summary
- Total modules: 253
- AG News remaining: 184 modules unrun or partially run (690 total runs across all 253; 352 runs for injectable Categories 1-3)
- MNLI remaining: 170 modules unrun or partially run (674 total runs across all 253; 336 runs for injectable Categories 1-3)
- GPU: Apple Silicon MPS (local dev environment) / Remote NVIDIA RTX 4070 (12 GB VRAM benchmark environment)
- Disk free: 35 GiB
- Recommended plan: minimal
- Estimated time: ~2.3 hours on NVIDIA RTX 4070 / ~8.6 hours on Apple Silicon MPS
