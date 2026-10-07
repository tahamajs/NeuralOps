# Directory Reorganization (Step 6) Documentation

## Overview
The `neuralops` attention and compression modules have been reorganized from flat directories into categorized domain-specific subpackages while maintaining 100% backwards compatibility via `_LEGACY_MAP` shims.

## Attention Modules Mapping (114 files)

| Legacy Flat Path | Categorized Subpackage Path |
|---|---|
| `neuralops/attentions/adaptive_span_attention.py` | `neuralops/attentions/adaptive/adaptive_span_attention.py` |
| `neuralops/attentions/adaptive_sparse_attention.py` | `neuralops/attentions/sparse/adaptive_sparse_attention.py` |
| `neuralops/attentions/adaptive_svd_attention.py` | `neuralops/attentions/svd/adaptive_svd_attention.py` |
| `neuralops/attentions/adasplash_attention.py` | `neuralops/attentions/fast_kernels/adasplash_attention.py` |
| `neuralops/attentions/asvd_attention.py` | `neuralops/attentions/svd/asvd_attention.py` |
| `neuralops/attentions/bam_umoe.py` | `neuralops/attentions/sparse/bam_umoe.py` |
| `neuralops/attentions/block_transformer_hiba.py` | `neuralops/attentions/adaptive/block_transformer_hiba.py` |
| `neuralops/attentions/chela_slay.py` | `neuralops/attentions/linear/chela_slay.py` |
| `neuralops/attentions/cluster_attention.py` | `neuralops/attentions/sparse/cluster_attention.py` |
| `neuralops/attentions/compressed_attention.py` | `neuralops/attentions/svd/compressed_attention.py` |
| `neuralops/attentions/compressed_mha_attention.py` | `neuralops/attentions/svd/compressed_mha_attention.py` |
| `neuralops/attentions/cosformer.py` | `neuralops/attentions/linear/cosformer.py` |
| `neuralops/attentions/decay_linear_attention.py` | `neuralops/attentions/gated/decay_linear_attention.py` |
| `neuralops/attentions/eat.py` | `neuralops/attentions/adaptive/eat.py` |
| `neuralops/attentions/expert_choice_moe.py` | `neuralops/attentions/sparse/expert_choice_moe.py` |
| `neuralops/attentions/factorised_matrix_state.py` | `neuralops/attentions/svd/factorised_matrix_state.py` |
| `neuralops/attentions/flash_attention3.py` | `neuralops/attentions/fast_kernels/flash_attention3.py` |
| `neuralops/attentions/flash_attention3_fp8.py` | `neuralops/attentions/fast_kernels/flash_attention3_fp8.py` |
| `neuralops/attentions/full_attention.py` | `neuralops/attentions/dense/full_attention.py` |
| `neuralops/attentions/fused_svd_kernel.py` | `neuralops/attentions/svd/fused_svd_kernel.py` |
| `neuralops/attentions/fusion_attn.py` | `neuralops/attentions/hybrid/fusion_attn.py` |
| `neuralops/attentions/galerkin_attention.py` | `neuralops/attentions/linear/galerkin_attention.py` |
| `neuralops/attentions/gated_delta_net.py` | `neuralops/attentions/gated/gated_delta_net.py` |
| `neuralops/attentions/gated_deltanet_block.py` | `neuralops/attentions/gated/gated_deltanet_block.py` |
| `neuralops/attentions/gated_hybrid_kernel_attention.py` | `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py` |
| `neuralops/attentions/gated_linear_attention.py` | `neuralops/attentions/gated/gated_linear_attention.py` |
| `neuralops/attentions/goldfinch_ahn.py` | `neuralops/attentions/ssm/goldfinch_ahn.py` |
| `neuralops/attentions/gqa_attention.py` | `neuralops/attentions/modern/gqa_attention.py` |
| `neuralops/attentions/granite_hybrid.py` | `neuralops/attentions/hybrid/granite_hybrid.py` |
| `neuralops/attentions/griffin_ssd.py` | `neuralops/attentions/ssm/griffin_ssd.py` |
| `neuralops/attentions/halo_regla.py` | `neuralops/attentions/hybrid/halo_regla.py` |
| `neuralops/attentions/hamming_attention.py` | `neuralops/attentions/sparse/hamming_attention.py` |
| `neuralops/attentions/hgrn2_gated_deltanet.py` | `neuralops/attentions/gated/hgrn2_gated_deltanet.py` |
| `neuralops/attentions/hgrn2_gated_layer.py` | `neuralops/attentions/gated/hgrn2_gated_layer.py` |
| `neuralops/attentions/hgrn2_gated_outer.py` | `neuralops/attentions/gated/hgrn2_gated_outer.py` |
| `neuralops/attentions/hla_attention.py` | `neuralops/attentions/gated/hla_attention.py` |
| `neuralops/attentions/hybrid_attention.py` | `neuralops/attentions/hybrid/hybrid_attention.py` |
| `neuralops/attentions/hybrid_local_kernel_attention.py` | `neuralops/attentions/hybrid/hybrid_local_kernel_attention.py` |
| `neuralops/attentions/hymba_transx.py` | `neuralops/attentions/ssm/hymba_transx.py` |
| `neuralops/attentions/infini_attention.py` | `neuralops/attentions/adaptive/infini_attention.py` |
| `neuralops/attentions/jamba.py` | `neuralops/attentions/ssm/jamba.py` |
| `neuralops/attentions/jamba_info.py` | `neuralops/attentions/ssm/jamba_info.py` |
| `neuralops/attentions/kda_attention.py` | `neuralops/attentions/gated/kda_attention.py` |
| `neuralops/attentions/landmark_local_attention.py` | `neuralops/attentions/hybrid/landmark_local_attention.py` |
| `neuralops/attentions/linformer_attention.py` | `neuralops/attentions/linear/linformer_attention.py` |
| `neuralops/attentions/lola_attention.py` | `neuralops/attentions/adaptive/lola_attention.py` |
| `neuralops/attentions/lora_attention.py` | `neuralops/attentions/svd/lora_attention.py` |
| `neuralops/attentions/lsh_attention.py` | `neuralops/attentions/sparse/lsh_attention.py` |
| `neuralops/attentions/mamba_block.py` | `neuralops/attentions/ssm/mamba_block.py` |
| `neuralops/attentions/mamba_svd_flash.py` | `neuralops/attentions/ssm/mamba_svd_flash.py` |
| `neuralops/attentions/minimal_nova.py` | `neuralops/attentions/svd/minimal_nova.py` |
| `neuralops/attentions/minimal_svd.py` | `neuralops/attentions/svd/minimal_svd.py` |
| `neuralops/attentions/mixattention_moska.py` | `neuralops/attentions/hybrid/mixattention_moska.py` |
| `neuralops/attentions/mla_attention.py` | `neuralops/attentions/modern/mla_attention.py` |
| `neuralops/attentions/moba_attention.py` | `neuralops/attentions/sparse/moba_attention.py` |
| `neuralops/attentions/moe_deltakv_paged.py` | `neuralops/attentions/sparse/moe_deltakv_paged.py` |
| `neuralops/attentions/naive_svd.py` | `neuralops/attentions/svd/naive_svd.py` |
| `neuralops/attentions/nasc_attention.py` | `neuralops/attentions/linear/nasc_attention.py` |
| `neuralops/attentions/nexus.py` | `neuralops/attentions/ssm/nexus.py` |
| `neuralops/attentions/nha_attention.py` | `neuralops/attentions/modern/nha_attention.py` |
| `neuralops/attentions/nsa_attention.py` | `neuralops/attentions/sparse/nsa_attention.py` |
| `neuralops/attentions/nystrom_attention.py` | `neuralops/attentions/linear/nystrom_attention.py` |
| `neuralops/attentions/optimized_hybrid_attention.py` | `neuralops/attentions/hybrid/optimized_hybrid_attention.py` |
| `neuralops/attentions/orion.py` | `neuralops/attentions/ssm/orion.py` |
| `neuralops/attentions/padre_attention.py` | `neuralops/attentions/sparse/padre_attention.py` |
| `neuralops/attentions/performer_attention.py` | `neuralops/attentions/linear/performer_attention.py` |
| `neuralops/attentions/phantom.py` | `neuralops/attentions/ssm/phantom.py` |
| `neuralops/attentions/positive_kernel_linear_attention.py` | `neuralops/attentions/hybrid/positive_kernel_linear_attention.py` |
| `neuralops/attentions/primal_attention.py` | `neuralops/attentions/linear/primal_attention.py` |
| `neuralops/attentions/progressive_token_pruning.py` | `neuralops/attentions/adaptive/progressive_token_pruning.py` |
| `neuralops/attentions/projection_factorized_attn.py` | `neuralops/attentions/svd/projection_factorized_attn.py` |
| `neuralops/attentions/qcqa_attention.py` | `neuralops/attentions/modern/qcqa_attention.py` |
| `neuralops/attentions/rala_attention.py` | `neuralops/attentions/linear/rala_attention.py` |
| `neuralops/attentions/reap_moe.py` | `neuralops/attentions/sparse/reap_moe.py` |
| `neuralops/attentions/refined_landmark_attention.py` | `neuralops/attentions/hybrid/refined_landmark_attention.py` |
| `neuralops/attentions/regla_attention.py` | `neuralops/attentions/hybrid/regla_attention.py` |
| `neuralops/attentions/residual_svd_attention.py` | `neuralops/attentions/svd/residual_svd_attention.py` |
| `neuralops/attentions/routing_attention.py` | `neuralops/attentions/sparse/routing_attention.py` |
| `neuralops/attentions/rpc_attention.py` | `neuralops/attentions/linear/rpc_attention.py` |
| `neuralops/attentions/salt_attention.py` | `neuralops/attentions/svd/salt_attention.py` |
| `neuralops/attentions/samba_m3.py` | `neuralops/attentions/ssm/samba_m3.py` |
| `neuralops/attentions/samba_moe.py` | `neuralops/attentions/ssm/samba_moe.py` |
| `neuralops/attentions/selective_ssm.py` | `neuralops/attentions/ssm/selective_ssm.py` |
| `neuralops/attentions/simple_ssm.py` | `neuralops/attentions/ssm/simple_ssm.py` |
| `neuralops/attentions/soa_attn.py` | `neuralops/attentions/fast_kernels/soa_attn.py` |
| `neuralops/attentions/span_expanded_attention.py` | `neuralops/attentions/adaptive/span_expanded_attention.py` |
| `neuralops/attentions/spectra_v2.py` | `neuralops/attentions/linear/spectra_v2.py` |
| `neuralops/attentions/spectraformer_attention.py` | `neuralops/attentions/linear/spectraformer_attention.py` |
| `neuralops/attentions/spectral_mamba.py` | `neuralops/attentions/ssm/spectral_mamba.py` |
| `neuralops/attentions/spectron.py` | `neuralops/attentions/linear/spectron.py` |
| `neuralops/attentions/spectron_fp8_attention.py` | `neuralops/attentions/hybrid/spectron_fp8_attention.py` |
| `neuralops/attentions/split_head_hybrid_attention.py` | `neuralops/attentions/hybrid/split_head_hybrid_attention.py` |
| `neuralops/attentions/squeezing_heads_distill.py` | `neuralops/attentions/adaptive/squeezing_heads_distill.py` |
| `neuralops/attentions/svd_attention.py` | `neuralops/attentions/svd/svd_attention.py` |
| `neuralops/attentions/svd_compressed.py` | `neuralops/attentions/svd/svd_compressed.py` |
| `neuralops/attentions/svd_linear_attention.py` | `neuralops/attentions/svd/svd_linear_attention.py` |
| `neuralops/attentions/svd_linear_decay.py` | `neuralops/attentions/svd/svd_linear_decay.py` |
| `neuralops/attentions/svd_llm_v2_attention.py` | `neuralops/attentions/svd/svd_llm_v2_attention.py` |
| `neuralops/attentions/svd_mamba.py` | `neuralops/attentions/svd/svd_mamba.py` |
| `neuralops/attentions/svd_projection_attention.py` | `neuralops/attentions/svd/svd_projection_attention.py` |
| `neuralops/attentions/template_attention.py` | `neuralops/attentions/experimental/template_attention.py` |
| `neuralops/attentions/tlinformer_attention.py` | `neuralops/attentions/linear/tlinformer_attention.py` |
| `neuralops/attentions/token_compressed_attention.py` | `neuralops/attentions/adaptive/token_compressed_attention.py` |
| `neuralops/attentions/tpa_attention.py` | `neuralops/attentions/linear/tpa_attention.py` |
| `neuralops/attentions/tri_branch_hybrid_attention.py` | `neuralops/attentions/hybrid/tri_branch_hybrid_attention.py` |
| `neuralops/attentions/triton_cosformer.py` | `neuralops/attentions/linear/triton_cosformer.py` |
| `neuralops/attentions/triton_svd_linear_attention.py` | `neuralops/attentions/svd/triton_svd_linear_attention.py` |
| `neuralops/attentions/unified_rope.py` | `neuralops/attentions/modern/unified_rope.py` |
| `neuralops/attentions/upi_stuffed_mamba.py` | `neuralops/attentions/ssm/upi_stuffed_mamba.py` |
| `neuralops/attentions/vig_attention.py` | `neuralops/attentions/modern/vig_attention.py` |
| `neuralops/attentions/vspec.py` | `neuralops/attentions/ssm/vspec.py` |
| `neuralops/attentions/wgqa_attention.py` | `neuralops/attentions/modern/wgqa_attention.py` |
| `neuralops/attentions/xattention.py` | `neuralops/attentions/fast_kernels/xattention.py` |
| `neuralops/attentions/zeros_attention.py` | `neuralops/attentions/dense/zeros_attention.py` |

## Compression Engine Modules Mapping (73 files)

| Legacy Flat Path | Categorized Subpackage Path |
|---|---|
| `neuralops/compression_engine/a3_compressor.py` | `neuralops/compression_engine/low_rank/a3_compressor.py` |
| `neuralops/compression_engine/aa_svd_compressor.py` | `neuralops/compression_engine/svd/aa_svd_compressor.py` |
| `neuralops/compression_engine/aatc_xkv_compressor.py` | `neuralops/compression_engine/kv_cache/aatc_xkv_compressor.py` |
| `neuralops/compression_engine/adaptive_ranks.py` | `neuralops/compression_engine/svd/adaptive_ranks.py` |
| `neuralops/compression_engine/adasvd_compressor.py` | `neuralops/compression_engine/svd/adasvd_compressor.py` |
| `neuralops/compression_engine/admm_compressor.py` | `neuralops/compression_engine/pruning/admm_compressor.py` |
| `neuralops/compression_engine/admm_pruner.py` | `neuralops/compression_engine/pruning/admm_pruner.py` |
| `neuralops/compression_engine/armor_pruner.py` | `neuralops/compression_engine/pruning/armor_pruner.py` |
| `neuralops/compression_engine/ast_pruner.py` | `neuralops/compression_engine/pruning/ast_pruner.py` |
| `neuralops/compression_engine/astro_quantizer.py` | `neuralops/compression_engine/quantization/astro_quantizer.py` |
| `neuralops/compression_engine/bonsai_pruner.py` | `neuralops/compression_engine/pruning/bonsai_pruner.py` |
| `neuralops/compression_engine/cachenotes_compressor.py` | `neuralops/compression_engine/kv_cache/cachenotes_compressor.py` |
| `neuralops/compression_engine/come_compressor.py` | `neuralops/compression_engine/low_rank/come_compressor.py` |
| `neuralops/compression_engine/compot_compressor.py` | `neuralops/compression_engine/low_rank/compot_compressor.py` |
| `neuralops/compression_engine/coopq_quantizer.py` | `neuralops/compression_engine/quantization/coopq_quantizer.py` |
| `neuralops/compression_engine/cr_compressor.py` | `neuralops/compression_engine/low_rank/cr_compressor.py` |
| `neuralops/compression_engine/cur_compressor.py` | `neuralops/compression_engine/low_rank/cur_compressor.py` |
| `neuralops/compression_engine/cur_moe_compressor.py` | `neuralops/compression_engine/low_rank/cur_moe_compressor.py` |
| `neuralops/compression_engine/delta_kv.py` | `neuralops/compression_engine/kv_cache/delta_kv.py` |
| `neuralops/compression_engine/direct_svd.py` | `neuralops/compression_engine/svd/direct_svd.py` |
| `neuralops/compression_engine/dobi_svd.py` | `neuralops/compression_engine/svd/dobi_svd.py` |
| `neuralops/compression_engine/drone_compressor.py` | `neuralops/compression_engine/low_rank/drone_compressor.py` |
| `neuralops/compression_engine/dynamic_rank.py` | `neuralops/compression_engine/svd/dynamic_rank.py` |
| `neuralops/compression_engine/enlighten_compressor.py` | `neuralops/compression_engine/low_rank/enlighten_compressor.py` |
| `neuralops/compression_engine/entropy_budget_allocation.py` | `neuralops/compression_engine/svd/entropy_budget_allocation.py` |
| `neuralops/compression_engine/erc_svd.py` | `neuralops/compression_engine/svd/erc_svd.py` |
| `neuralops/compression_engine/faq_quantizer.py` | `neuralops/compression_engine/quantization/faq_quantizer.py` |
| `neuralops/compression_engine/flap.py` | `neuralops/compression_engine/pruning/flap.py` |
| `neuralops/compression_engine/flat_llm.py` | `neuralops/compression_engine/experimental/flat_llm.py` |
| `neuralops/compression_engine/frandi_compressor.py` | `neuralops/compression_engine/low_rank/frandi_compressor.py` |
| `neuralops/compression_engine/fwsvd_compressor.py` | `neuralops/compression_engine/svd/fwsvd_compressor.py` |
| `neuralops/compression_engine/gfws_d_compressor.py` | `neuralops/compression_engine/svd/gfws_d_compressor.py` |
| `neuralops/compression_engine/grecodec_compressor.py` | `neuralops/compression_engine/low_rank/grecodec_compressor.py` |
| `neuralops/compression_engine/hashed_net_compressor.py` | `neuralops/compression_engine/experimental/hashed_net_compressor.py` |
| `neuralops/compression_engine/huffman_coder.py` | `neuralops/compression_engine/quantization/huffman_coder.py` |
| `neuralops/compression_engine/hybrid_svd_layer.py` | `neuralops/compression_engine/svd/hybrid_svd_layer.py` |
| `neuralops/compression_engine/hybridkv_compressor.py` | `neuralops/compression_engine/kv_cache/hybridkv_compressor.py` |
| `neuralops/compression_engine/kmeans_quantize.py` | `neuralops/compression_engine/quantization/kmeans_quantize.py` |
| `neuralops/compression_engine/kronecker_compressor.py` | `neuralops/compression_engine/tensor_decomp/kronecker_compressor.py` |
| `neuralops/compression_engine/kronecker_tt_lora.py` | `neuralops/compression_engine/tensor_decomp/kronecker_tt_lora.py` |
| `neuralops/compression_engine/kvtc_compressor.py` | `neuralops/compression_engine/kv_cache/kvtc_compressor.py` |
| `neuralops/compression_engine/liftr.py` | `neuralops/compression_engine/experimental/liftr.py` |
| `neuralops/compression_engine/lillama_compressor.py` | `neuralops/compression_engine/experimental/lillama_compressor.py` |
| `neuralops/compression_engine/lsc_compressor.py` | `neuralops/compression_engine/low_rank/lsc_compressor.py` |
| `neuralops/compression_engine/mamba_shedder.py` | `neuralops/compression_engine/pruning/mamba_shedder.py` |
| `neuralops/compression_engine/mgie_svd_compressor.py` | `neuralops/compression_engine/low_rank/mgie_svd_compressor.py` |
| `neuralops/compression_engine/mlorq_compressor.py` | `neuralops/compression_engine/low_rank/mlorq_compressor.py` |
| `neuralops/compression_engine/modegpt_compressor.py` | `neuralops/compression_engine/low_rank/modegpt_compressor.py` |
| `neuralops/compression_engine/nmf_compressor.py` | `neuralops/compression_engine/low_rank/nmf_compressor.py` |
| `neuralops/compression_engine/oats_compressor.py` | `neuralops/compression_engine/low_rank/oats_compressor.py` |
| `neuralops/compression_engine/optimal_subspace_projection.py` | `neuralops/compression_engine/svd/optimal_subspace_projection.py` |
| `neuralops/compression_engine/proxsparse.py` | `neuralops/compression_engine/pruning/proxsparse.py` |
| `neuralops/compression_engine/residual_svd_compressor.py` | `neuralops/compression_engine/svd/residual_svd_compressor.py` |
| `neuralops/compression_engine/roast_hashed_linear.py` | `neuralops/compression_engine/experimental/roast_hashed_linear.py` |
| `neuralops/compression_engine/rotational_slicegpt_int4.py` | `neuralops/compression_engine/quantization/rotational_slicegpt_int4.py` |
| `neuralops/compression_engine/rpca_compressor.py` | `neuralops/compression_engine/low_rank/rpca_compressor.py` |
| `neuralops/compression_engine/saes_svd_v2.py` | `neuralops/compression_engine/svd/saes_svd_v2.py` |
| `neuralops/compression_engine/salr.py` | `neuralops/compression_engine/low_rank/salr.py` |
| `neuralops/compression_engine/sdrm_compressor.py` | `neuralops/compression_engine/low_rank/sdrm_compressor.py` |
| `neuralops/compression_engine/slicegpt.py` | `neuralops/compression_engine/quantization/slicegpt.py` |
| `neuralops/compression_engine/slicegpt_pca.py` | `neuralops/compression_engine/quantization/slicegpt_pca.py` |
| `neuralops/compression_engine/slim_compressor.py` | `neuralops/compression_engine/low_rank/slim_compressor.py` |
| `neuralops/compression_engine/soka_compressor.py` | `neuralops/compression_engine/low_rank/soka_compressor.py` |
| `neuralops/compression_engine/sparsegpt.py` | `neuralops/compression_engine/pruning/sparsegpt.py` |
| `neuralops/compression_engine/subspace_projection.py` | `neuralops/compression_engine/svd/subspace_projection.py` |
| `neuralops/compression_engine/svd_linear.py` | `neuralops/compression_engine/svd/svd_linear.py` |
| `neuralops/compression_engine/svd_llm_compressor.py` | `neuralops/compression_engine/low_rank/svd_llm_compressor.py` |
| `neuralops/compression_engine/swift_svd.py` | `neuralops/compression_engine/svd/swift_svd.py` |
| `neuralops/compression_engine/swift_svd_act.py` | `neuralops/compression_engine/svd/swift_svd_act.py` |
| `neuralops/compression_engine/tomoe_compressor.py` | `neuralops/compression_engine/low_rank/tomoe_compressor.py` |
| `neuralops/compression_engine/tt_compressor.py` | `neuralops/compression_engine/tensor_decomp/tt_compressor.py` |
| `neuralops/compression_engine/uniql_compressor.py` | `neuralops/compression_engine/low_rank/uniql_compressor.py` |
| `neuralops/compression_engine/zerokv_neo_compressor.py` | `neuralops/compression_engine/kv_cache/zerokv_neo_compressor.py` |

## Registry Key Preservation Proof

- **Total Attention Registry Keys Registered**: 159 / 159 (100% preserved)
- **Total Compressor Registry Keys Registered**: 73 / 73 (100% preserved)
- **Deprecated Aliases Verified**: 641 / 641 passed (0 failures)
- **Old Flat Imports**: Fully functional via `sys.modules` backward compatibility shims.
- **New Subpackage Imports**: Fully functional via subpackage top-level exports.
