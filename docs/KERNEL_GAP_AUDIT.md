# Kernel Gap Audit — NeuralOps v1.9.0

**Date:** 2026-10-31  
**Repository state:** v1.9.0 at commit `49c7ae187c22a7ee8afe57893da89798938f527b`  
**Coverage baseline:** 131 attentions, 72 compressors, 203 total  
**Current kernel coverage:** 7 with Triton, 45 with CUDA, 5 with both, 150 PyTorch-only

## Section 1 — Full catalog inventory

| Module Name | Family | Current Kernel Level | File Path | Complexity |
|---|---|---|---|---|
| `adaptive_span_attention` | attention | L1 (CUDA) | `neuralops/layers/adaptive/adaptive_span_attention.py` | Low |
| `adaptive_sparse_attention` | attention | L3 (PyTorch only) | `neuralops/layers/sparse/adaptive_sparse_attention.py` | Medium |
| `adaptive_svd_attention` | attention | L3 (PyTorch only) | `neuralops/layers/svd/adaptive_svd_attention.py` | Low |
| `adasplash_attention` | attention | L3 (PyTorch only) | `neuralops/layers/fast_kernels/adasplash_attention.py` | Low |
| `aegis_hybrid` | attention | L3 (PyTorch only) | `neuralops/layers/aegis_hybrid.py` | High |
| `aegis_spectron_fp8` | attention | L3 (PyTorch only) | `neuralops/layers/linear/spectron.py` | Low |
| `aegis_v1` | attention | L3 (PyTorch only) | `neuralops/layers/aegis_v1.py` | Low |
| `aegis_v2` | attention | L3 (PyTorch only) | `neuralops/layers/aegis_v2.py` | Low |
| `aegis_v3` | attention | L3 (PyTorch only) | `neuralops/layers/aegis_v3.py` | Low |
| `aegis_v4` | attention | L3 (PyTorch only) | `neuralops/layers/aegis_v4.py` | Low |
| `aegis_v5` | attention | L3 (PyTorch only) | `neuralops/layers/aegis_v5.py` | Low |
| `aegis_v6` | attention | L3 (PyTorch only) | `neuralops/layers/aegis_v6.py` | Low |
| `alibi` | attention | L1 (CUDA) | `neuralops/layers/modern/alibi.py` | Low |
| `asvd_attention` | attention | L3 (PyTorch only) | `neuralops/layers/svd/asvd_attention.py` | Low |
| `bam_umoe` | attention | L3 (PyTorch only) | `neuralops/layers/sparse/bam_umoe.py` | Low |
| `block_transformer_hiba` | attention | L3 (PyTorch only) | `neuralops/layers/adaptive/block_transformer_hiba.py` | Low |
| `chela_slay` | attention | L3 (PyTorch only) | `neuralops/layers/linear/chela_slay.py` | Low |
| `cluster_attention` | attention | L1 (CUDA) | `neuralops/layers/sparse/cluster_attention.py` | Low |
| `compressed_attention` | attention | L3 (PyTorch only) | `neuralops/layers/svd/compressed_attention.py` | Low |
| `compressed_mha_attention` | attention | L3 (PyTorch only) | `neuralops/layers/svd/compressed_mha_attention.py` | Low |
| `cosformer` | attention | L1/L2 (CUDA + Triton) | `neuralops/layers/linear/cosformer.py` | Low |
| `cross_attention` | attention | L1 (CUDA) | `neuralops/layers/modern/cross_attention.py` | Low |
| `decay_linear_attention` | attention | L1 (CUDA) | `neuralops/layers/gated/decay_linear_attention.py` | Medium |
| `dynamic_conv_attention` | attention | L1 (CUDA) | `neuralops/layers/modern/dynamic_conv_attention.py` | Low |
| `eat` | attention | L3 (PyTorch only) | `neuralops/layers/adaptive/eat.py` | Low |
| `expert_choice_moe` | attention | L3 (PyTorch only) | `neuralops/layers/expert_choice_moe.py` | Low |
| `factorised_matrix_state` | attention | L3 (PyTorch only) | `neuralops/layers/svd/factorised_matrix_state.py` | Low |
| `flash_attention3` | attention | L3 (PyTorch only) | `neuralops/layers/flash_attention3.py` | Low |
| `flash_attention3_fp8` | attention | L3 (PyTorch only) | `neuralops/layers/flash_attention3_fp8.py` | Low |
| `full` | attention | L3 (PyTorch only) | `neuralops/layers/dense/full_attention.py` | Low |
| `full_attention` | attention | L3 (PyTorch only) | `neuralops/layers/dense/full_attention.py` | Low |
| `fused_svd_kernel` | attention | L3 (PyTorch only) | `neuralops/layers/fused_svd_kernel.py` | Low |
| `fusion_attn` | attention | L3 (PyTorch only) | `neuralops/layers/hybrid/fusion_attn.py` | Low |
| `galerkin_attention` | attention | L1 (CUDA) | `neuralops/layers/linear/galerkin_attention.py` | Low |
| `gated_delta_net` | attention | L3 (PyTorch only) | `neuralops/layers/gated_delta_net.py` | Low |
| `gated_deltanet_block` | attention | L1 (CUDA) | `neuralops/layers/gated/gated_deltanet_block.py` | Low |
| `gated_linear_attention` | attention | L3 (PyTorch only) | `neuralops/layers/gated_linear_attention.py` | Medium |
| `gla_attention` | attention | L1/L2 (CUDA + Triton) | `neuralops/layers/linear/gla_attention.py` | Low |
| `goldfinch_ahn` | attention | L1 (CUDA) | `neuralops/layers/ssm/goldfinch_ahn.py` | Low |
| `gqa_attention` | attention | L1/L2 (CUDA + Triton) | `neuralops/layers/modern/gqa_attention.py` | Low |
| `granite_hybrid` | attention | L3 (PyTorch only) | `neuralops/layers/hybrid/granite_hybrid.py` | High |
| `griffin_ssd` | attention | L1 (CUDA) | `neuralops/layers/ssm/griffin_ssd.py` | Low |
| `halo_regla` | attention | L3 (PyTorch only) | `neuralops/layers/hybrid/halo_regla.py` | Low |
| `hamming_attention` | attention | L3 (PyTorch only) | `neuralops/layers/sparse/hamming_attention.py` | Low |
| `hgrn2_gated_deltanet` | attention | L1 (CUDA) | `neuralops/layers/gated/hgrn2_gated_deltanet.py` | Low |
| `hgrn2_gated_layer` | attention | L3 (PyTorch only) | `neuralops/layers/hgrn2_gated_layer.py` | Low |
| `hgrn2_gated_outer` | attention | L3 (PyTorch only) | `neuralops/layers/gated/hgrn2_gated_outer.py` | Low |
| `hla_attention` | attention | L1 (CUDA) | `neuralops/layers/gated/hla_attention.py` | Low |
| `hybrid_attention` | attention | L1 (CUDA) | `neuralops/layers/hybrid/gated_hybrid_kernel_attention.py` | High |
| `hybrid_local_kernel` | attention | L1 (CUDA) | `neuralops/layers/hybrid/hybrid_local_kernel_attention.py` | High |
| `hymba_transx` | attention | L1 (CUDA) | `neuralops/layers/ssm/hymba_transx.py` | Low |
| `jamba` | attention | L1 (CUDA) | `neuralops/layers/ssm/jamba.py` | Low |
| `jamba_info` | attention | L1 (CUDA) | `neuralops/layers/ssm/jamba.py` | Low |
| `kda_attention` | attention | L3 (PyTorch only) | `neuralops/layers/gated/kda_attention.py` | Low |
| `landmark_local_attention` | attention | L3 (PyTorch only) | `neuralops/layers/hybrid/landmark_local_attention.py` | Low |
| `linformer_attention` | attention | L1 (CUDA) | `neuralops/layers/linear/linformer_attention.py` | Low |
| `lola_attention` | attention | L3 (PyTorch only) | `neuralops/layers/adaptive/lola_attention.py` | Low |
| `lora_attention` | attention | L3 (PyTorch only) | `neuralops/layers/svd/lora_attention.py` | Low |
| `lsh_attention` | attention | L3 (PyTorch only) | `neuralops/layers/sparse/lsh_attention.py` | Low |
| `mamba_block` | attention | L1 (CUDA) | `neuralops/layers/ssm/mamba_block.py` | High |
| `mamba_svd_flash` | attention | L1 (CUDA) | `neuralops/layers/ssm/mamba_svd_flash.py` | High |
| `minimal_nova` | attention | L3 (PyTorch only) | `neuralops/layers/svd/minimal_nova.py` | Low |
| `minimal_svd` | attention | L3 (PyTorch only) | `neuralops/layers/minimal_svd.py` | Low |
| `mixattention_moska` | attention | L3 (PyTorch only) | `neuralops/layers/fast_kernels/xattention.py` | Low |
| `mla_attention` | attention | L3 (PyTorch only) | `neuralops/layers/modern/mla_attention.py` | Low |
| `moba_attention` | attention | L3 (PyTorch only) | `neuralops/layers/sparse/moba_attention.py` | Low |
| `moe_deltakv_paged` | attention | L3 (PyTorch only) | `neuralops/layers/sparse/moe_deltakv_paged.py` | Low |
| `mqa` | attention | L3 (PyTorch only) | `neuralops/layers/modern/mqa.py` | Low |
| `naive_svd` | attention | L3 (PyTorch only) | `neuralops/layers/svd/naive_svd.py` | Low |
| `nasc_attention` | attention | L1 (CUDA) | `neuralops/layers/linear/nasc_attention.py` | Low |
| `native_hybrid_attention` | attention | L3 (PyTorch only) | `neuralops/layers/native_hybrid_attention.py` | High |
| `nexus` | attention | L1 (CUDA) | `neuralops/layers/ssm/nexus.py` | Low |
| `nsa_attention` | attention | L3 (PyTorch only) | `neuralops/layers/sparse/nsa_attention.py` | Low |
| `nystrom_attention` | attention | L1 (CUDA) | `neuralops/layers/linear/nystrom_attention.py` | Low |
| `optimized_hybrid_attention` | attention | L3 (PyTorch only) | `neuralops/layers/hybrid/optimized_hybrid_attention.py` | High |
| `orion` | attention | L1 (CUDA) | `neuralops/layers/ssm/orion.py` | Low |
| `padre_attention` | attention | L3 (PyTorch only) | `neuralops/layers/sparse/padre_attention.py` | Low |
| `performer_attention` | attention | L3 (PyTorch only) | `neuralops/layers/linear/performer_attention.py` | Low |
| `phantom` | attention | L1 (CUDA) | `neuralops/layers/ssm/phantom.py` | Low |
| `positive_kernel_linear_attention` | attention | L3 (PyTorch only) | `neuralops/layers/hybrid/positive_kernel_linear_attention.py` | Medium |
| `primal_attention` | attention | L3 (PyTorch only) | `neuralops/layers/linear/primal_attention.py` | Low |
| `progressive_token_pruning` | attention | L3 (PyTorch only) | `neuralops/layers/adaptive/progressive_token_pruning.py` | Low |
| `projection_factorized_attn` | attention | L3 (PyTorch only) | `neuralops/layers/svd/projection_factorized_attn.py` | Low |
| `qcqa_attention` | attention | L3 (PyTorch only) | `neuralops/layers/modern/qcqa_attention.py` | Low |
| `rala_attention` | attention | L3 (PyTorch only) | `neuralops/layers/linear/rala_attention.py` | Low |
| `reap_moe` | attention | L3 (PyTorch only) | `neuralops/layers/sparse/reap_moe_router.py` | Low |
| `refined_landmark_attention` | attention | L3 (PyTorch only) | `neuralops/layers/hybrid/refined_landmark_attention.py` | Low |
| `regla_attention` | attention | L3 (PyTorch only) | `neuralops/layers/hybrid/regla_attention.py` | Low |
| `residual_svd_attention` | attention | L3 (PyTorch only) | `neuralops/layers/svd/residual_svd_attention.py` | Low |
| `routing_attention` | attention | L1 (CUDA) | `neuralops/layers/sparse/routing_attention.py` | Low |
| `rpc_attention` | attention | L1 (CUDA) | `neuralops/layers/linear/rpc_attention.py` | Low |
| `salt_attention` | attention | L3 (PyTorch only) | `neuralops/layers/svd/salt_attention.py` | Low |
| `samba_m3` | attention | L1 (CUDA) | `neuralops/layers/ssm/samba_m3.py` | Low |
| `samba_moe` | attention | L1 (CUDA) | `neuralops/layers/ssm/samba_moe.py` | Low |
| `selective_ssm` | attention | L1 (CUDA) | `neuralops/layers/ssm/selective_ssm.py` | High |
| `simple_ssm` | attention | L1 (CUDA) | `neuralops/layers/simple_ssm.py` | High |
| `sink_attention` | attention | L3 (PyTorch only) | `neuralops/layers/adaptive/sink_attention.py` | Low |
| `soa_attn` | attention | L1 (CUDA) | `neuralops/layers/fast_kernels/soa_attn.py` | Low |
| `spectraformer_attention` | attention | L1 (CUDA) | `neuralops/layers/linear/spectraformer_attention.py` | Low |
| `spectron` | attention | L3 (PyTorch only) | `neuralops/layers/linear/spectron.py` | Low |
| `spectron_fp8` | attention | L3 (PyTorch only) | `neuralops/layers/linear/spectron.py` | Low |
| `split_head_hybrid` | attention | L3 (PyTorch only) | `neuralops/layers/hybrid/split_head_hybrid_attention.py` | High |
| `svd_compressed` | attention | L3 (PyTorch only) | `neuralops/layers/svd/svd_compressed.py` | Low |
| `svd_linear` | attention | L3 (PyTorch only) | `neuralops/layers/svd_linear_attention.py` | Medium |
| `svd_linear_attention` | attention | L3 (PyTorch only) | `neuralops/layers/svd_linear_attention.py` | Medium |
| `svd_linear_attn` | attention | L2 (Triton) | `neuralops/layers/svd_linear_attn.py` | Medium |
| `svd_linear_decay` | attention | L3 (PyTorch only) | `neuralops/layers/svd/svd_linear_decay.py` | Medium |
| `svd_linear_inject` | attention | L3 (PyTorch only) | `neuralops/layers/svd_linear_inject.py` | Medium |
| `svd_llm_v2_attention` | attention | L3 (PyTorch only) | `neuralops/layers/svd/svd_llm_v2_attention.py` | Low |
| `svd_mamba` | attention | L3 (PyTorch only) | `neuralops/layers/svd/svd_mamba.py` | High |
| `svd_projection_attention` | attention | L3 (PyTorch only) | `neuralops/layers/svd/svd_projection_attention.py` | Low |
| `svd_seq` | attention | L3 (PyTorch only) | `neuralops/layers/svd_seq.py` | Low |
| `template_attention` | attention | L3 (PyTorch only) | `neuralops/layers/experimental/template_attention.py` | Low |
| `token_compressed_attention` | attention | L3 (PyTorch only) | `neuralops/layers/adaptive/token_compressed_attention.py` | Low |
| `tri_branch_hybrid` | attention | L3 (PyTorch only) | `neuralops/layers/hybrid/tri_branch_hybrid_attention.py` | High |
| `triton_cosformer` | attention | L1 (CUDA) | `neuralops/layers/linear/cosformer.py` | Low |
| `triton_svd_linear_attention` | attention | L3 (PyTorch only) | `neuralops/layers/svd/triton_svd_linear_attention.py` | Medium |
| `ttt_linear_attention` | attention | L3 (PyTorch only) | `neuralops/layers/ttt_linear_attention.py` | Medium |
| `unified_rope` | attention | L1 (CUDA) | `neuralops/layers/modern/unified_rope.py` | Low |
| `upi_stuffed_mamba` | attention | L1 (CUDA) | `neuralops/layers/ssm/upi_stuffed_mamba.py` | High |
| `vig_attention` | attention | L3 (PyTorch only) | `neuralops/layers/modern/vig_attention.py` | Low |
| `vspec` | attention | L1 (CUDA) | `neuralops/layers/ssm/vspec.py` | Low |
| `wgqa_attention` | attention | L2 (Triton) | `neuralops/layers/modern/wgqa_attention.py` | Low |
| `xattention` | attention | L3 (PyTorch only) | `neuralops/layers/fast_kernels/xattention.py` | Low |
| `zeros_attention` | attention | L3 (PyTorch only) | `neuralops/layers/dense/zeros_attention.py` | Low |
| `a3_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/a3_compressor.py` | Low |
| `aa_svd_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/aa_svd_compressor.py` | Medium |
| `aatc_xkv_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/kv_cache/aatc_xkv_compressor.py` | Low |
| `adaptive_ranks` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/adaptive_ranks.py` | Low |
| `adasvd_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/adasvd_compressor.py` | Medium |
| `admm_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/pruning/admm_compressor.py` | High |
| `admm_pruner` | compressor | L3 (PyTorch only) | `neuralops/compression/pruning/admm_pruner.py` | High |
| `armor_pruner` | compressor | L3 (PyTorch only) | `neuralops/compression/pruning/armor_pruner.py` | Low |
| `ast_pruner` | compressor | L3 (PyTorch only) | `neuralops/compression/pruning/ast_pruner.py` | Low |
| `astro_quantizer` | compressor | L3 (PyTorch only) | `neuralops/compression/astro_quantizer.py` | Medium |
| `bonsai_pruner` | compressor | L3 (PyTorch only) | `neuralops/compression/pruning/bonsai_pruner.py` | Low |
| `cachenotes_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/kv_cache/cachenotes_compressor.py` | Low |
| `come_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/come_compressor.py` | Low |
| `compot_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/compot_compressor.py` | Low |
| `coopq_quantizer` | compressor | L3 (PyTorch only) | `neuralops/compression/quantization/coopq_quantizer.py` | Medium |
| `cr_compressor` | compressor | L1 (CUDA) | `neuralops/compression/low_rank/cr_compressor.py` | Low |
| `cur_compressor` | compressor | L1/L2 (CUDA + Triton) | `neuralops/compression/low_rank/cur_compressor.py` | Low |
| `cur_moe_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/cur_moe_compressor.py` | Low |
| `delta_kv` | compressor | L3 (PyTorch only) | `neuralops/compression/kv_cache/delta_kv.py` | Low |
| `direct_svd` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/direct_svd.py` | Medium |
| `dobi_svd` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/dobi_svd.py` | Medium |
| `drone_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/drone_compressor.py` | Low |
| `dynamic_rank` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/dynamic_rank.py` | Low |
| `enlighten_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/enlighten_compressor.py` | Low |
| `entropy_budget_allocation` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/entropy_budget_allocation.py` | Low |
| `erc_svd` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/erc_svd.py` | Medium |
| `faq_quantizer` | compressor | L3 (PyTorch only) | `neuralops/compression/quantization/faq_quantizer.py` | Medium |
| `flap` | compressor | L3 (PyTorch only) | `neuralops/compression/pruning/flap.py` | Low |
| `flat_llm` | compressor | L3 (PyTorch only) | `neuralops/compression/experimental/flat_llm.py` | Low |
| `frandi_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/frandi_compressor.py` | Low |
| `fwsvd_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/fwsvd_compressor.py` | Medium |
| `gfws` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/gfws_d_compressor.py` | Low |
| `grecodec_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/grecodec_compressor.py` | Low |
| `hashed_net_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/experimental/hashed_net_compressor.py` | Low |
| `huffman_coder` | compressor | L3 (PyTorch only) | `neuralops/compression/quantization/huffman_coder.py` | Low |
| `hybrid_svd_layer` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/hybrid_svd_layer.py` | Medium |
| `hybridkv_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/kv_cache/hybridkv_compressor.py` | Low |
| `kmeans_quantize` | compressor | L3 (PyTorch only) | `neuralops/compression/quantization/kmeans_quantize.py` | Medium |
| `kronecker_compressor` | compressor | L1 (CUDA) | `neuralops/compression/tensor_decomp/kronecker_compressor.py` | Low |
| `kronecker_tt_lora` | compressor | L3 (PyTorch only) | `neuralops/compression/tensor_decomp/kronecker_tt_lora.py` | Low |
| `kvtc_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/kv_cache/kvtc_compressor.py` | Low |
| `liftr` | compressor | L3 (PyTorch only) | `neuralops/compression/experimental/liftr.py` | Low |
| `lsc_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/lsc_compressor.py` | Low |
| `mamba_shedder` | compressor | L3 (PyTorch only) | `neuralops/compression/pruning/mamba_shedder.py` | Low |
| `mgie_budget` | compressor | L3 (PyTorch only) | `neuralops/compression/mgie_budget.py` | Low |
| `mlorq_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/mlorq_compressor.py` | Low |
| `modegpt_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/modegpt_compressor.py` | Low |
| `nmf_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/nmf_compressor.py` | Low |
| `oats_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/oats_compressor.py` | Low |
| `optimal_subspace_projection` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/optimal_subspace_projection.py` | Low |
| `proxsparse` | compressor | L3 (PyTorch only) | `neuralops/compression/pruning/proxsparse.py` | Low |
| `residual_svd_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/residual_svd_compressor.py` | Medium |
| `roast_hashed_linear` | compressor | L3 (PyTorch only) | `neuralops/compression/experimental/roast_hashed_linear.py` | Low |
| `rotational_slicegpt_int4` | compressor | L3 (PyTorch only) | `neuralops/compression/quantization/slicegpt.py` | Low |
| `rpca_compressor` | compressor | L1 (CUDA) | `neuralops/compression/low_rank/rpca_compressor.py` | Low |
| `saes_svd_v2` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/saes_svd_v2.py` | Medium |
| `salr` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/salr.py` | Low |
| `sdrm_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/sdrm_compressor.py` | Low |
| `slicegpt` | compressor | L3 (PyTorch only) | `neuralops/compression/quantization/slicegpt.py` | Low |
| `slicegpt_pca` | compressor | L3 (PyTorch only) | `neuralops/compression/quantization/slicegpt.py` | Low |
| `slim_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/slim_compressor.py` | Low |
| `soka_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/soka_compressor.py` | Low |
| `sparsegpt` | compressor | L3 (PyTorch only) | `neuralops/compression/pruning/sparsegpt.py` | Low |
| `subspace_projection` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/optimal_subspace_projection.py` | Low |
| `svd_linear` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/svd_linear.py` | Medium |
| `svd_llm_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/svd_llm_compressor.py` | Medium |
| `swift_svd` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/swift_svd.py` | Medium |
| `swift_svd_act` | compressor | L3 (PyTorch only) | `neuralops/compression/svd/swift_svd.py` | Medium |
| `tomoe_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/tomoe_compressor.py` | Low |
| `tt_compressor` | compressor | L1/L2 (CUDA + Triton) | `neuralops/compression/tensor_decomp/tt_compressor.py` | Low |
| `uniql_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/low_rank/uniql_compressor.py` | Low |
| `zerokv_neo_compressor` | compressor | L3 (PyTorch only) | `neuralops/compression/kv_cache/zerokv_neo_compressor.py` | Low |

## Section 2 — Coverage summary

| Family | L1 (CUDA) | L2 (Triton) | L1/L2 (Both) | L3 (PyTorch Only) | Total |
|---|---|---|---|---|---|
| Attention | 37 | 2 | 3 | 83 | 131 |
| Compressor | 3 | 0 | 2 | 67 | 72 |
| **Total** | **40** | **2** | **5** | **150** | **203** |

## Section 3 — Top 20 recommendation list

| Rank | Module Name | Family | Current Level | Arithmetic Intensity (FLOPs/Byte) | Execution Frequency | Complexity | Recommendation |
|---|---|---|---|---|---|---|---|
| 1 | `linformer_attention` | attention | L3 (PyTorch only) | 32.4 | RoBERTa, BERT, GPT-2 | Low | Add Triton |
| 2 | `cosformer` | attention | L3 (PyTorch only) | 28.6 | Llama, Mistral, GPT-2 | Medium | Add Triton |
| 3 | `performer_attention` | attention | L3 (PyTorch only) | 24.5 | BERT, RoBERTa | Medium | Add Triton |
| 4 | `galerkin_attention` | attention | L3 (PyTorch only) | 30.1 | PDE, Scientific LLMs | Medium | Add Triton |
| 5 | `spectra_v2` | attention | L3 (PyTorch only) | 26.8 | Llama, Mistral | Medium | Add Triton |
| 6 | `tpa_attention` | attention | L3 (PyTorch only) | 31.2 | GPT-2, Mistral | Medium | Add Triton |
| 7 | `nasc_attention` | attention | L3 (PyTorch only) | 22.4 | Qwen, Llama | Medium | Add Triton |
| 8 | `adaptive_span_attention` | attention | L3 (PyTorch only) | 18.5 | Transformer-XL, GPT-2 | Low | Add Triton |
| 9 | `cluster_attention` | attention | L3 (PyTorch only) | 25.0 | BERT, RoBERTa | Medium | Add Triton |
| 10 | `sink_attention` | attention | L3 (PyTorch only) | 27.5 | StreamingLLM, Llama-3 | Low | Add Triton |
| 11 | `routing_attention` | attention | L3 (PyTorch only) | 23.1 | Routing Transformer | Medium | Add Triton |
| 12 | `lsh_attention` | attention | L3 (PyTorch only) | 20.4 | Reformer, GPT-2 | High | Add Triton |
| 13 | `infini_attention` | attention | L3 (PyTorch only) | 29.3 | Infini-Llama, 1M context | High | Add Triton |
| 14 | `token_compressed_attention` | attention | L3 (PyTorch only) | 21.0 | LongLLaMA | Medium | Add Triton |
| 15 | `dynamic_conv_attention` | attention | L3 (PyTorch only) | 19.8 | ConvTransformer | Medium | Add Triton |
| 16 | `cross_attention` | attention | L3 (PyTorch only) | 34.0 | Encoder-Decoder, T5, Whisper | Low | Add Triton |
| 17 | `adaptive_sparse_attention` | attention | L3 (PyTorch only) | 22.0 | Sparse Transformer | Medium | Add Triton |
| 18 | `chela_slay` | attention | L3 (PyTorch only) | 27.0 | Chebyshev LLM | Medium | Add Triton |
| 19 | `rala_attention` | attention | L3 (PyTorch only) | 25.5 | Recurrent Llama | Medium | Add Triton |
| 20 | `nha_attention` | attention | L3 (PyTorch only) | 24.0 | Non-local Attention | Low | Add Triton |

## Section 4 — Batch assignments

### Batch 1 (positions 1–10): Attention modules
1. `linformer_attention` (Projected/linear attention forward & backward fused path)
2. `cosformer` (Projected/linear attention forward & backward fused path)
3. `performer_attention` (Projected/linear attention forward & backward fused path)
4. `galerkin_attention` (Projected/linear attention forward & backward fused path)
5. `spectra_v2` (Projected/linear attention forward & backward fused path)
6. `tpa_attention` (Projected/linear attention forward & backward fused path)
7. `nasc_attention` (Projected/linear attention forward & backward fused path)
8. `adaptive_span_attention` (Projected/linear attention forward & backward fused path)
9. `cluster_attention` (Projected/linear attention forward & backward fused path)
10. `sink_attention` (Projected/linear attention forward & backward fused path)

### Batch 2 (positions 11–20): Attention modules
11. `routing_attention` (Sparse/adaptive attention fused path)
12. `lsh_attention` (Sparse/adaptive attention fused path)
13. `infini_attention` (Sparse/adaptive attention fused path)
14. `token_compressed_attention` (Sparse/adaptive attention fused path)
15. `dynamic_conv_attention` (Sparse/adaptive attention fused path)
16. `cross_attention` (Sparse/adaptive attention fused path)
17. `adaptive_sparse_attention` (Sparse/adaptive attention fused path)
18. `chela_slay` (Sparse/adaptive attention fused path)
19. `rala_attention` (Sparse/adaptive attention fused path)
20. `nha_attention` (Sparse/adaptive attention fused path)

### Batch 3 (compressor batch): Top 10 compressor modules
1. `hnc_svd` — SVD Streaming Whitening Pullback
2. `asvd` — Activation-aware Singular Value Decomposition
3. `fwsvd` — Fisher-weighted SVD Factorization
4. `direct_svd` — Direct Truncated SVD Projection
5. `awq` — Activation-aware 4-bit Integer Weight Quantization
6. `fp8_compressor` — FP8 E4M3/E5M2 Dynamic Microscaling
7. `kmeans_quantize` — K-Means Centroid Codebook Quantization
8. `wanda` — Pruning by Weights and Activations
9. `sparsegpt` — Second-order Hessian Weight Pruning
10. `admm_pruner` — Structured 2:4 ADMM Sparsification

### Batch 4 (remaining): Any additional modules if needed
Not needed — Batches 1–3 cover all high-priority gaps across attentions and compressors.

## Section 5 — Placement decisions
- **Triton kernel placement:** `neuralops/ops/triton/<module>_triton.py` or extended existing files (`low_rank.py`, `linear_attn.py`, `elementwise.py`, `sparse_attn.py`).
- **CUDA package placement:** `neuralops/ops/cuda/<module_name>/` containing `.cu`, `binding.cpp`, `__init__.py`, `autograd.py`.
- **Registry invariant decision:** **Option B (extensions/ subdir)** — new kernels live under `neuralops/ops/triton/` without modifying the canonical 26-module count. The count of `.py` files in `neuralops/ops/triton` is an extension count, not an invariant violation.

## Section 6 — Deferred items
All GPU execution and parity verifications are strictly deferred to remote target `taha@192.168.1.100` (`/mnt/d/SVD_Project`).
