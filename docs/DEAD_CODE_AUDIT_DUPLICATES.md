# Duplicate Code Audit (Round 2)

## Duplicate Function Bodies (129 groups)

### Group `[841940092f26]` — 75 instances

- `neuralops/attentions/ssm/griffin_ssd.py:46` (`_cfg_get`)
- `neuralops/attentions/ssm/samba_m3.py:35` (`_cfg_get`)
- `neuralops/attentions/ssm/orion.py:44` (`_cfg_get`)
- `neuralops/attentions/ssm/spectral_mamba.py:55` (`_cfg_get`)
- `neuralops/attentions/experimental/template_attention.py:30` (`_cfg_get`)
- `neuralops/attentions/linear/tpa_attention.py:48` (`_cfg_get`)
- `neuralops/attentions/linear/spectra_v2.py:48` (`_cfg_get`)
- `neuralops/attentions/linear/spectraformer_attention.py:38` (`_cfg_get`)
- `neuralops/attentions/linear/spectron.py:55` (`_cfg_get`)
- `neuralops/attentions/linear/rpc_attention.py:32` (`_cfg_get`)
- `neuralops/attentions/svd/svd_projection_attention.py:41` (`_cfg_get`)
- `neuralops/attentions/svd/fused_svd_kernel.py:442` (`_cfg_get`)
- `neuralops/attentions/svd/svd_attention.py:52` (`_cfg_get`)
- `neuralops/attentions/svd/svd_compressed.py:44` (`_cfg_get`)
- `neuralops/attentions/svd/svd_llm_v2_attention.py:31` (`_cfg_get`)
- `neuralops/attentions/svd/svd_mamba.py:26` (`_cfg_get`)
- `neuralops/attentions/svd/svd_linear_attention.py:79` (`_cfg_get`)
- `neuralops/attentions/gated/hgrn2_gated_deltanet.py:42` (`_cfg_get`)
- `neuralops/attentions/gated/hgrn2_gated_layer.py:63` (`_cfg_get`)
- `neuralops/attentions/sparse/routing_attention.py:31` (`_cfg_get`)
- `neuralops/attentions/sparse/hamming_attention.py:47` (`_cfg_get`)
- `neuralops/attentions/modern/gqa_attention.py:28` (`_cfg_get`)
- `neuralops/attentions/modern/qcqa_attention.py:30` (`_cfg_get`)
- `neuralops/attentions/modern/nha_attention.py:32` (`_cfg_get`)
- `neuralops/attentions/adaptive/squeezing_heads_distill.py:26` (`_cfg_get`)
- `neuralops/attentions/hybrid/regla_attention.py:24` (`_cfg_get`)
- `neuralops/attentions/hybrid/granite_hybrid.py:28` (`_cfg_get`)
- `neuralops/attentions/hybrid/halo_regla.py:36` (`_cfg_get`)
- `neuralops/compression_engine/experimental/liftr.py:25` (`_cfg_get`)
- `neuralops/compression_engine/experimental/lillama_compressor.py:29` (`_cfg_get`)
- `neuralops/compression_engine/experimental/flat_llm.py:29` (`_cfg_get`)
- `neuralops/compression_engine/experimental/hashed_net_compressor.py:406` (`_cfg_get`)
- `neuralops/compression_engine/tensor_decomp/kronecker_compressor.py:504` (`_cfg_get`)
- `neuralops/compression_engine/tensor_decomp/tt_compressor.py:48` (`_cfg_get`)
- `neuralops/compression_engine/tensor_decomp/kronecker_tt_lora.py:27` (`_cfg_get`)
- `neuralops/compression_engine/quantization/slicegpt.py:54` (`_cfg_get`)
- `neuralops/compression_engine/quantization/faq_quantizer.py:24` (`_cfg_get`)
- `neuralops/compression_engine/quantization/rotational_slicegpt_int4.py:38` (`_cfg_get`)
- `neuralops/compression_engine/quantization/kmeans_quantize.py:29` (`_cfg_get`)
- `neuralops/compression_engine/quantization/slicegpt_pca.py:182` (`_cfg_get`)
- `neuralops/compression_engine/svd/saes_svd_v2.py:34` (`_cfg_get`)
- `neuralops/compression_engine/svd/optimal_subspace_projection.py:37` (`_cfg_get`)
- `neuralops/compression_engine/svd/swift_svd.py:24` (`_cfg_get`)
- `neuralops/compression_engine/svd/erc_svd.py:34` (`_cfg_get`)
- `neuralops/compression_engine/svd/aa_svd_compressor.py:65` (`_cfg_get`)
- `neuralops/compression_engine/svd/subspace_projection.py:44` (`_cfg_get`)
- `neuralops/compression_engine/svd/svd_linear.py:31` (`_cfg_get`)
- `neuralops/compression_engine/svd/swift_svd_act.py:40` (`_cfg_get`)
- `neuralops/compression_engine/svd/hybrid_svd_layer.py:33` (`_cfg_get`)
- `neuralops/compression_engine/svd/adasvd_compressor.py:83` (`_cfg_get`)
- `neuralops/compression_engine/pruning/admm_compressor.py:28` (`_cfg_get`)
- `neuralops/compression_engine/pruning/mamba_shedder.py:48` (`_cfg_get`)
- `neuralops/compression_engine/pruning/flap.py:31` (`_cfg_get`)
- `neuralops/compression_engine/pruning/proxsparse.py:48` (`_cfg_get`)
- `neuralops/compression_engine/pruning/sparsegpt.py:38` (`_cfg_get`)
- `neuralops/compression_engine/pruning/armor_pruner.py:34` (`_cfg_get`)
- `neuralops/compression_engine/pruning/admm_pruner.py:36` (`_cfg_get`)
- `neuralops/compression_engine/kv_cache/delta_kv.py:155` (`_cfg_get`)
- `neuralops/compression_engine/kv_cache/kvtc_compressor.py:40` (`_cfg_get`)
- `neuralops/compression_engine/kv_cache/zerokv_neo_compressor.py:119` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/rpca_compressor.py:23` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/frandi_compressor.py:26` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/sdrm_compressor.py:25` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/mgie_svd_compressor.py:26` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/cur_compressor.py:248` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/lsc_compressor.py:37` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/oats_compressor.py:192` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/salr.py:136` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/uniql_compressor.py:25` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/soka_compressor.py:31` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/tomoe_compressor.py:386` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/modegpt_compressor.py:181` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/slim_compressor.py:36` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/nmf_compressor.py:300` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/mlorq_compressor.py:322` (`_cfg_get`)

### Group `[792248e7c928]` — 39 instances

- `neuralops/attentions/ssm/upi_stuffed_mamba.py:87` (`reset_parameters`)
- `neuralops/attentions/ssm/jamba.py:111` (`reset_parameters`)
- `neuralops/attentions/ssm/hymba_transx.py:231` (`reset_parameters`)
- `neuralops/attentions/ssm/jamba_info.py:226` (`reset_parameters`)
- `neuralops/attentions/ssm/spectral_mamba.py:154` (`reset_parameters`)
- `neuralops/attentions/fast_kernels/xattention.py:182` (`reset_parameters`)
- `neuralops/attentions/fast_kernels/flash_attention3_fp8.py:144` (`reset_parameters`)
- `neuralops/attentions/fast_kernels/flash_attention3.py:134` (`reset_parameters`)
- `neuralops/attentions/linear/nystrom_attention.py:107` (`reset_parameters`)
- `neuralops/attentions/linear/nasc_attention.py:122` (`reset_parameters`)
- `neuralops/attentions/linear/spectra_v2.py:139` (`reset_parameters`)
- `neuralops/attentions/linear/spectraformer_attention.py:120` (`reset_parameters`)
- `neuralops/attentions/linear/performer_attention.py:192` (`reset_parameters`)
- `neuralops/attentions/linear/primal_attention.py:129` (`reset_parameters`)
- `neuralops/attentions/linear/rala_attention.py:116` (`reset_parameters`)
- `neuralops/attentions/linear/galerkin_attention.py:122` (`reset_parameters`)
- `neuralops/attentions/dense/full_attention.py:110` (`reset_parameters`)
- `neuralops/attentions/dense/zeros_attention.py:123` (`reset_parameters`)
- `neuralops/attentions/svd/factorised_matrix_state.py:140` (`reset_parameters`)
- `neuralops/attentions/gated/kda_attention.py:112` (`reset_parameters`)
- `neuralops/attentions/gated/decay_linear_attention.py:135` (`reset_parameters`)
- `neuralops/attentions/sparse/padre_attention.py:120` (`reset_parameters`)
- `neuralops/attentions/sparse/cluster_attention.py:110` (`reset_parameters`)
- `neuralops/attentions/sparse/reap_moe.py:154` (`reset_parameters`)
- `neuralops/attentions/sparse/nsa_attention.py:109` (`reset_parameters`)
- `neuralops/attentions/sparse/hamming_attention.py:171` (`reset_parameters`)
- `neuralops/attentions/sparse/adaptive_sparse_attention.py:110` (`reset_parameters`)
- `neuralops/attentions/sparse/expert_choice_moe.py:125` (`reset_parameters`)
- `neuralops/attentions/modern/vig_attention.py:125` (`reset_parameters`)
- `neuralops/attentions/modern/mla_attention.py:126` (`reset_parameters`)
- `neuralops/attentions/modern/wgqa_attention.py:129` (`reset_parameters`)
- `neuralops/attentions/modern/qcqa_attention.py:159` (`reset_parameters`)
- `neuralops/attentions/modern/nha_attention.py:126` (`reset_parameters`)
- `neuralops/attentions/adaptive/progressive_token_pruning.py:136` (`reset_parameters`)
- `neuralops/attentions/adaptive/infini_attention.py:116` (`reset_parameters`)
- `neuralops/attentions/adaptive/span_expanded_attention.py:118` (`reset_parameters`)
- `neuralops/attentions/adaptive/block_transformer_hiba.py:126` (`reset_parameters`)
- `neuralops/attentions/hybrid/regla_attention.py:123` (`reset_parameters`)
- `neuralops/attentions/hybrid/fusion_attn.py:133` (`reset_parameters`)

### Group `[14bb4e9cec57]` — 30 instances

- `neuralops/attentions/ssm/upi_stuffed_mamba.py:101` (`freeze_base_weights`)
- `neuralops/attentions/ssm/nexus.py:146` (`freeze_base_weights`)
- `neuralops/attentions/ssm/jamba_info.py:239` (`freeze_base_weights`)
- `neuralops/attentions/ssm/spectral_mamba.py:168` (`freeze_base_weights`)
- `neuralops/attentions/fast_kernels/xattention.py:196` (`freeze_base_weights`)
- `neuralops/attentions/fast_kernels/soa_attn.py:197` (`freeze_base_weights`)
- `neuralops/attentions/linear/nasc_attention.py:136` (`freeze_base_weights`)
- `neuralops/attentions/linear/spectra_v2.py:153` (`freeze_base_weights`)
- `neuralops/attentions/linear/spectraformer_attention.py:134` (`freeze_base_weights`)
- `neuralops/attentions/linear/primal_attention.py:143` (`freeze_base_weights`)
- `neuralops/attentions/linear/rala_attention.py:129` (`freeze_base_weights`)
- `neuralops/attentions/linear/galerkin_attention.py:135` (`freeze_base_weights`)
- `neuralops/attentions/dense/full_attention.py:124` (`freeze_base_weights`)
- `neuralops/attentions/dense/zeros_attention.py:137` (`freeze_base_weights`)
- `neuralops/attentions/svd/lora_attention.py:152` (`freeze_base_weights`)
- `neuralops/attentions/sparse/padre_attention.py:134` (`freeze_base_weights`)
- `neuralops/attentions/sparse/moe_deltakv_paged.py:228` (`freeze_base_weights`)
- `neuralops/attentions/sparse/cluster_attention.py:124` (`freeze_base_weights`)
- `neuralops/attentions/sparse/reap_moe.py:168` (`freeze_base_weights`)
- `neuralops/attentions/sparse/hamming_attention.py:185` (`freeze_base_weights`)
- `neuralops/attentions/sparse/adaptive_sparse_attention.py:124` (`freeze_base_weights`)
- `neuralops/attentions/modern/vig_attention.py:139` (`freeze_base_weights`)
- `neuralops/attentions/modern/mla_attention.py:139` (`freeze_base_weights`)
- `neuralops/attentions/modern/wgqa_attention.py:143` (`freeze_base_weights`)
- `neuralops/attentions/modern/qcqa_attention.py:173` (`freeze_base_weights`)
- `neuralops/attentions/modern/nha_attention.py:139` (`freeze_base_weights`)
- `neuralops/attentions/adaptive/progressive_token_pruning.py:149` (`freeze_base_weights`)
- `neuralops/attentions/adaptive/span_expanded_attention.py:132` (`freeze_base_weights`)
- `neuralops/attentions/adaptive/adaptive_span_attention.py:245` (`freeze_base_weights`)
- `neuralops/attentions/hybrid/regla_attention.py:136` (`freeze_base_weights`)

### Group `[89b8206730c4]` — 29 instances

- `neuralops/attentions/ssm/nexus.py:95` (`__init__`)
- `neuralops/attentions/ssm/hymba_transx.py:194` (`__init__`)
- `neuralops/attentions/ssm/spectral_mamba.py:117` (`__init__`)
- `neuralops/attentions/fast_kernels/xattention.py:145` (`__init__`)
- `neuralops/attentions/fast_kernels/flash_attention3_fp8.py:107` (`__init__`)
- `neuralops/attentions/fast_kernels/flash_attention3.py:97` (`__init__`)
- `neuralops/attentions/linear/nystrom_attention.py:70` (`__init__`)
- `neuralops/attentions/linear/nasc_attention.py:85` (`__init__`)
- `neuralops/attentions/linear/spectra_v2.py:102` (`__init__`)
- `neuralops/attentions/linear/spectraformer_attention.py:83` (`__init__`)
- `neuralops/attentions/linear/primal_attention.py:92` (`__init__`)
- `neuralops/attentions/linear/rala_attention.py:79` (`__init__`)
- `neuralops/attentions/linear/galerkin_attention.py:85` (`__init__`)
- `neuralops/attentions/dense/full_attention.py:73` (`__init__`)
- `neuralops/attentions/dense/zeros_attention.py:86` (`__init__`)
- `neuralops/attentions/svd/factorised_matrix_state.py:103` (`__init__`)
- `neuralops/attentions/gated/kda_attention.py:75` (`__init__`)
- `neuralops/attentions/gated/decay_linear_attention.py:98` (`__init__`)
- `neuralops/attentions/sparse/padre_attention.py:83` (`__init__`)
- `neuralops/attentions/sparse/cluster_attention.py:73` (`__init__`)
- `neuralops/attentions/sparse/reap_moe.py:117` (`__init__`)
- `neuralops/attentions/sparse/nsa_attention.py:72` (`__init__`)
- `neuralops/attentions/sparse/hamming_attention.py:134` (`__init__`)
- `neuralops/attentions/sparse/adaptive_sparse_attention.py:73` (`__init__`)
- `neuralops/attentions/sparse/expert_choice_moe.py:88` (`__init__`)
- `neuralops/attentions/modern/qcqa_attention.py:122` (`__init__`)
- `neuralops/attentions/adaptive/span_expanded_attention.py:81` (`__init__`)
- `neuralops/attentions/adaptive/adaptive_span_attention.py:195` (`__init__`)
- `neuralops/attentions/adaptive/block_transformer_hiba.py:89` (`__init__`)

### Group `[7739ff3a80a4]` — 16 instances

- `neuralops/attentions/ssm/vspec.py:682` (`_orient`)
- `neuralops/attentions/ssm/jamba_info.py:709` (`_orient`)
- `neuralops/attentions/fast_kernels/xattention.py:528` (`_orient`)
- `neuralops/attentions/linear/nasc_attention.py:479` (`_orient`)
- `neuralops/attentions/dense/full_attention.py:430` (`_orient`)
- `neuralops/attentions/dense/zeros_attention.py:365` (`_orient`)
- `neuralops/attentions/svd/compressed_attention.py:374` (`_orient`)
- `neuralops/attentions/svd/compressed_mha_attention.py:367` (`_orient`)
- `neuralops/attentions/svd/minimal_svd.py:537` (`_orient`)
- `neuralops/attentions/svd/minimal_nova.py:553` (`_orient`)
- `neuralops/attentions/sparse/hamming_attention.py:513` (`_orient`)
- `neuralops/attentions/modern/vig_attention.py:471` (`_orient`)
- `neuralops/attentions/modern/wgqa_attention.py:484` (`_orient`)
- `neuralops/attentions/adaptive/lola_attention.py:602` (`_orient`)
- `neuralops/attentions/hybrid/split_head_hybrid_attention.py:872` (`_orient`)
- `neuralops/attentions/hybrid/hybrid_local_kernel_attention.py:798` (`_orient`)

### Group `[9d03062f8c2a]` — 15 instances

- `neuralops/attentions/ssm/hymba_transx.py:253` (`forward`)
- `neuralops/attentions/ssm/spectral_mamba.py:178` (`forward`)
- `neuralops/attentions/linear/nystrom_attention.py:135` (`forward`)
- `neuralops/attentions/linear/spectra_v2.py:163` (`forward`)
- `neuralops/attentions/linear/spectraformer_attention.py:144` (`forward`)
- `neuralops/attentions/linear/primal_attention.py:153` (`forward`)
- `neuralops/attentions/linear/rala_attention.py:139` (`forward`)
- `neuralops/attentions/linear/galerkin_attention.py:150` (`forward`)
- `neuralops/attentions/svd/factorised_matrix_state.py:170` (`forward`)
- `neuralops/attentions/gated/kda_attention.py:140` (`forward`)
- `neuralops/attentions/gated/decay_linear_attention.py:165` (`forward`)
- `neuralops/attentions/sparse/padre_attention.py:144` (`forward`)
- `neuralops/attentions/sparse/nsa_attention.py:137` (`forward`)
- `neuralops/attentions/modern/qcqa_attention.py:183` (`forward`)
- `neuralops/attentions/adaptive/span_expanded_attention.py:142` (`forward`)

### Group `[5eeca637e99b]` — 14 instances

- `neuralops/attentions/ssm/nexus.py:926` (`_orient`)
- `neuralops/attentions/linear/linformer_attention.py:688` (`_orient`)
- `neuralops/attentions/svd/lora_attention.py:514` (`_orient`)
- `neuralops/attentions/gated/gated_linear_attention.py:483` (`_orient`)
- `neuralops/attentions/sparse/moba_attention.py:513` (`_orient`)
- `neuralops/attentions/sparse/lsh_attention.py:648` (`_orient`)
- `neuralops/attentions/sparse/bam_umoe.py:700` (`_orient`)
- `neuralops/attentions/modern/mla_attention.py:515` (`_orient`)
- `neuralops/attentions/adaptive/eat.py:634` (`_orient`)
- `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:1162` (`_orient`)
- `neuralops/attentions/hybrid/mixattention_moska.py:525` (`_orient`)
- `neuralops/attentions/hybrid/landmark_local_attention.py:767` (`_orient`)
- `neuralops/attentions/hybrid/refined_landmark_attention.py:806` (`_orient`)
- `neuralops/attentions/hybrid/positive_kernel_linear_attention.py:663` (`_orient`)

### Group `[18d70b943131]` — 12 instances

- `neuralops/attentions/ssm/goldfinch_ahn.py:111` (`reset_parameters`)
- `neuralops/attentions/ssm/mamba_svd_flash.py:100` (`reset_parameters`)
- `neuralops/attentions/linear/cosformer.py:114` (`reset_parameters`)
- `neuralops/attentions/linear/tlinformer_attention.py:124` (`reset_parameters`)
- `neuralops/attentions/gated/gated_delta_net.py:250` (`reset_parameters`)
- `neuralops/attentions/gated/gated_linear_attention.py:122` (`reset_parameters`)
- `neuralops/attentions/sparse/moba_attention.py:110` (`reset_parameters`)
- `neuralops/attentions/modern/gqa_attention.py:136` (`reset_parameters`)
- `neuralops/attentions/adaptive/eat.py:113` (`reset_parameters`)
- `neuralops/attentions/hybrid/optimized_hybrid_attention.py:140` (`reset_parameters`)
- `neuralops/attentions/hybrid/mixattention_moska.py:99` (`reset_parameters`)
- `neuralops/attentions/hybrid/halo_regla.py:131` (`reset_parameters`)

### Group `[bcd971752b4d]` — 11 instances

- `neuralops/attentions/ssm/goldfinch_ahn.py:124` (`freeze_base_weights`)
- `neuralops/attentions/ssm/mamba_svd_flash.py:113` (`freeze_base_weights`)
- `neuralops/attentions/linear/cosformer.py:127` (`freeze_base_weights`)
- `neuralops/attentions/gated/gated_delta_net.py:263` (`freeze_base_weights`)
- `neuralops/attentions/gated/gated_linear_attention.py:135` (`freeze_base_weights`)
- `neuralops/attentions/sparse/moba_attention.py:123` (`freeze_base_weights`)
- `neuralops/attentions/modern/gqa_attention.py:149` (`freeze_base_weights`)
- `neuralops/attentions/adaptive/eat.py:126` (`freeze_base_weights`)
- `neuralops/attentions/hybrid/optimized_hybrid_attention.py:153` (`freeze_base_weights`)
- `neuralops/attentions/hybrid/mixattention_moska.py:112` (`freeze_base_weights`)
- `neuralops/attentions/hybrid/halo_regla.py:144` (`freeze_base_weights`)

### Group `[8747c2c1d595]` — 10 instances

- `neuralops/attentions/ssm/vspec.py:31` (`_cfg_get`)
- `neuralops/attentions/ssm/selective_ssm.py:31` (`_cfg_get`)
- `neuralops/attentions/ssm/upi_stuffed_mamba.py:38` (`_cfg_get`)
- `neuralops/attentions/ssm/samba_moe.py:31` (`_cfg_get`)
- `neuralops/attentions/fast_kernels/xattention.py:37` (`_cfg_get`)
- `neuralops/attentions/dense/zeros_attention.py:31` (`_cfg_get`)
- `neuralops/attentions/svd/salt_attention.py:25` (`_cfg_get`)
- `neuralops/attentions/modern/unified_rope.py:28` (`_cfg_get`)
- `neuralops/attentions/modern/vig_attention.py:31` (`_cfg_get`)
- `neuralops/attentions/modern/wgqa_attention.py:35` (`_cfg_get`)

### Group `[1fbbc209b94e]` — 9 instances

- `neuralops/attentions/ssm/goldfinch_ahn.py:912` (`_sz`)
- `neuralops/attentions/ssm/mamba_svd_flash.py:1208` (`_sz`)
- `neuralops/attentions/linear/chela_slay.py:1124` (`_sz`)
- `neuralops/attentions/svd/adaptive_svd_attention.py:815` (`_sz`)
- `neuralops/attentions/gated/hgrn2_gated_deltanet.py:305` (`_sz`)
- `neuralops/attentions/gated/gated_delta_net.py:1021` (`_sz`)
- `neuralops/attentions/gated/gated_linear_attention.py:773` (`_sz`)
- `neuralops/attentions/modern/gqa_attention.py:802` (`_sz`)
- `neuralops/attentions/hybrid/tri_branch_hybrid_attention.py:1016` (`_sz`)

### Group `[654f56e4627d]` — 9 instances

- `neuralops/attentions/ssm/samba_m3.py:165` (`unfreeze_base_weights`)
- `neuralops/attentions/linear/nystrom_attention.py:130` (`unfreeze_base_weights`)
- `neuralops/attentions/linear/performer_attention.py:215` (`unfreeze_base_weights`)
- `neuralops/attentions/linear/galerkin_attention.py:145` (`unfreeze_base_weights`)
- `neuralops/attentions/gated/kda_attention.py:135` (`unfreeze_base_weights`)
- `neuralops/attentions/sparse/nsa_attention.py:132` (`unfreeze_base_weights`)
- `neuralops/attentions/sparse/adaptive_sparse_attention.py:134` (`unfreeze_base_weights`)
- `neuralops/attentions/modern/nha_attention.py:149` (`unfreeze_base_weights`)
- `neuralops/attentions/hybrid/fusion_attn.py:155` (`unfreeze_base_weights`)

### Group `[80be42332041]` — 8 instances

- `neuralops/attentions/ssm/goldfinch_ahn.py:722` (`_to_bool`)
- `neuralops/attentions/ssm/mamba_svd_flash.py:824` (`_to_bool`)
- `neuralops/attentions/gated/gated_delta_net.py:870` (`_to_bool`)
- `neuralops/attentions/gated/gated_linear_attention.py:650` (`_to_bool`)
- `neuralops/attentions/modern/gqa_attention.py:666` (`_to_bool`)
- `neuralops/attentions/hybrid/mixattention_moska.py:650` (`_to_bool`)
- `neuralops/attentions/hybrid/fusion_attn.py:726` (`_to_bool`)
- `neuralops/attentions/hybrid/tri_branch_hybrid_attention.py:852` (`_to_bool`)

### Group `[2d77db7748fd]` — 8 instances

- `neuralops/attentions/dense/full_attention.py:29` (`apply_rotary_pos_emb`)
- `neuralops/attentions/svd/compressed_attention.py:29` (`apply_rotary_pos_emb`)
- `neuralops/attentions/svd/compressed_mha_attention.py:29` (`apply_rotary_pos_emb`)
- `neuralops/attentions/svd/lora_attention.py:37` (`apply_rotary_pos_emb`)
- `neuralops/attentions/svd/svd_mamba.py:59` (`apply_rotary_pos_emb`)
- `neuralops/attentions/sparse/cluster_attention.py:29` (`apply_rotary_pos_emb`)
- `neuralops/attentions/sparse/adaptive_sparse_attention.py:29` (`apply_rotary_pos_emb`)
- `neuralops/attentions/hybrid/granite_hybrid.py:66` (`apply_rotary_pos_emb`)

### Group `[f4f5ae28a649]` — 8 instances

- `neuralops/compression_engine/quantization/huffman_coder.py:25` (`_cfg_get`)
- `neuralops/compression_engine/svd/dynamic_rank.py:246` (`_cfg_get`)
- `neuralops/compression_engine/svd/entropy_budget_allocation.py:24` (`_cfg_get`)
- `neuralops/compression_engine/svd/direct_svd.py:24` (`_cfg_get`)
- `neuralops/compression_engine/svd/dobi_svd.py:118` (`_cfg_get`)
- `neuralops/compression_engine/kv_cache/hybridkv_compressor.py:25` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/grecodec_compressor.py:25` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/enlighten_compressor.py:22` (`_cfg_get`)

### Group `[7f5c46e22947]` — 7 instances

- `neuralops/attentions/ssm/goldfinch_ahn.py:36` (`_env_flag`)
- `neuralops/attentions/ssm/hymba_transx.py:33` (`_env_flag`)
- `neuralops/attentions/ssm/griffin_ssd.py:33` (`_env_flag`)
- `neuralops/attentions/linear/cosformer.py:29` (`_env_flag`)
- `neuralops/attentions/gated/hgrn2_gated_deltanet.py:29` (`_env_flag`)
- `neuralops/attentions/gated/gated_linear_attention.py:42` (`_env_flag`)
- `neuralops/attentions/hybrid/tri_branch_hybrid_attention.py:36` (`_env_flag`)

### Group `[fe3e2c6dabb1]` — 7 instances

- `neuralops/attentions/ssm/goldfinch_ahn.py:139` (`forward`)
- `neuralops/attentions/ssm/mamba_svd_flash.py:128` (`forward`)
- `neuralops/attentions/linear/tlinformer_attention.py:146` (`forward`)
- `neuralops/attentions/linear/chela_slay.py:217` (`forward`)
- `neuralops/attentions/gated/gated_delta_net.py:273` (`forward`)
- `neuralops/attentions/modern/gqa_attention.py:159` (`forward`)
- `neuralops/attentions/hybrid/mixattention_moska.py:122` (`forward`)

### Group `[d2b1c0258d74]` — 7 instances

- `neuralops/attentions/linear/linformer_attention.py:181` (`freeze_base_weights`)
- `neuralops/attentions/linear/rpc_attention.py:175` (`freeze_base_weights`)
- `neuralops/attentions/sparse/routing_attention.py:154` (`freeze_base_weights`)
- `neuralops/attentions/sparse/lsh_attention.py:195` (`freeze_base_weights`)
- `neuralops/attentions/hybrid/landmark_local_attention.py:184` (`freeze_base_weights`)
- `neuralops/attentions/hybrid/refined_landmark_attention.py:214` (`freeze_base_weights`)
- `neuralops/attentions/hybrid/positive_kernel_linear_attention.py:200` (`freeze_base_weights`)

### Group `[d22e2a41ea9c]` — 7 instances

- `neuralops/attentions/dense/full_attention.py:134` (`forward`)
- `neuralops/attentions/sparse/moe_deltakv_paged.py:238` (`forward`)
- `neuralops/attentions/sparse/cluster_attention.py:134` (`forward`)
- `neuralops/attentions/sparse/hamming_attention.py:195` (`forward`)
- `neuralops/attentions/modern/nha_attention.py:154` (`forward`)
- `neuralops/attentions/adaptive/progressive_token_pruning.py:159` (`forward`)
- `neuralops/attentions/hybrid/fusion_attn.py:160` (`forward`)

### Group `[3a280a2b97be]` — 7 instances

- `neuralops/compression_engine/quantization/faq_quantizer.py:333` (`_is_gqa_asymmetric`)
- `neuralops/compression_engine/svd/gfws_d_compressor.py:205` (`_is_gqa_asymmetric`)
- `neuralops/compression_engine/svd/svd_linear.py:219` (`_is_gqa_asymmetric`)
- `neuralops/compression_engine/svd/direct_svd.py:148` (`_is_gqa_asymmetric`)
- `neuralops/compression_engine/pruning/ast_pruner.py:289` (`_is_gqa_asymmetric`)
- `neuralops/compression_engine/kv_cache/kvtc_compressor.py:449` (`_is_gqa_asymmetric`)
- `neuralops/compression_engine/low_rank/grecodec_compressor.py:301` (`_is_gqa_asymmetric`)

### Group `[ae5f9cc2834b]` — 6 instances

- `neuralops/attentions/ssm/hymba_transx.py:244` (`freeze_base_weights`)
- `neuralops/attentions/linear/nystrom_attention.py:121` (`freeze_base_weights`)
- `neuralops/attentions/linear/performer_attention.py:206` (`freeze_base_weights`)
- `neuralops/attentions/gated/kda_attention.py:126` (`freeze_base_weights`)
- `neuralops/attentions/sparse/nsa_attention.py:123` (`freeze_base_weights`)
- `neuralops/attentions/hybrid/fusion_attn.py:146` (`freeze_base_weights`)

### Group `[9c175bcc253c]` — 6 instances

- `neuralops/attentions/ssm/griffin_ssd.py:38` (`_env_float`)
- `neuralops/attentions/gated/hgrn2_gated_outer.py:46` (`_env_float`)
- `neuralops/attentions/gated/hgrn2_gated_deltanet.py:34` (`_env_float`)
- `neuralops/attentions/gated/hgrn2_gated_layer.py:55` (`_env_float`)
- `neuralops/attentions/gated/gated_linear_attention.py:47` (`_env_float`)
- `neuralops/attentions/hybrid/tri_branch_hybrid_attention.py:41` (`_env_float`)

### Group `[b81d348dda4b]` — 6 instances

- `neuralops/attentions/fast_kernels/xattention.py:206` (`forward`)
- `neuralops/attentions/linear/nasc_attention.py:146` (`forward`)
- `neuralops/attentions/dense/zeros_attention.py:147` (`forward`)
- `neuralops/attentions/modern/vig_attention.py:149` (`forward`)
- `neuralops/attentions/modern/wgqa_attention.py:153` (`forward`)
- `neuralops/attentions/adaptive/adaptive_span_attention.py:255` (`forward`)

### Group `[c1a34c208d2b]` — 6 instances

- `neuralops/attentions/fast_kernels/flash_attention3_fp8.py:58` (`repeat_kv`)
- `neuralops/attentions/fast_kernels/flash_attention3.py:48` (`repeat_kv`)
- `neuralops/attentions/svd/factorised_matrix_state.py:45` (`repeat_kv`)
- `neuralops/attentions/gated/decay_linear_attention.py:49` (`repeat_kv`)
- `neuralops/attentions/sparse/expert_choice_moe.py:39` (`repeat_kv`)
- `neuralops/attentions/hybrid/granite_hybrid.py:55` (`repeat_kv`)

### Group `[32426c91b68b]` — 6 instances

- `neuralops/attentions/fast_kernels/flash_attention3_fp8.py:158` (`freeze_base_weights`)
- `neuralops/attentions/fast_kernels/flash_attention3.py:148` (`freeze_base_weights`)
- `neuralops/attentions/svd/factorised_matrix_state.py:154` (`freeze_base_weights`)
- `neuralops/attentions/gated/decay_linear_attention.py:149` (`freeze_base_weights`)
- `neuralops/attentions/sparse/expert_choice_moe.py:139` (`freeze_base_weights`)
- `neuralops/attentions/adaptive/block_transformer_hiba.py:140` (`freeze_base_weights`)

### Group `[909bfb1b8e5c]` — 6 instances

- `neuralops/attentions/fast_kernels/flash_attention3_fp8.py:168` (`unfreeze_base_weights`)
- `neuralops/attentions/fast_kernels/flash_attention3.py:158` (`unfreeze_base_weights`)
- `neuralops/attentions/svd/factorised_matrix_state.py:164` (`unfreeze_base_weights`)
- `neuralops/attentions/gated/decay_linear_attention.py:159` (`unfreeze_base_weights`)
- `neuralops/attentions/sparse/expert_choice_moe.py:149` (`unfreeze_base_weights`)
- `neuralops/attentions/adaptive/block_transformer_hiba.py:150` (`unfreeze_base_weights`)

### Group `[78000334b494]` — 5 instances

- `neuralops/attentions/ssm/goldfinch_ahn.py:74` (`__init__`)
- `neuralops/attentions/ssm/mamba_svd_flash.py:63` (`__init__`)
- `neuralops/attentions/gated/gated_delta_net.py:213` (`__init__`)
- `neuralops/attentions/sparse/moba_attention.py:73` (`__init__`)
- `neuralops/attentions/hybrid/mixattention_moska.py:62` (`__init__`)

### Group `[675cca0b767c]` — 5 instances

- `neuralops/attentions/ssm/goldfinch_ahn.py:134` (`unfreeze_base_weights`)
- `neuralops/attentions/ssm/mamba_svd_flash.py:123` (`unfreeze_base_weights`)
- `neuralops/attentions/linear/chela_slay.py:212` (`unfreeze_base_weights`)
- `neuralops/attentions/adaptive/eat.py:136` (`unfreeze_base_weights`)
- `neuralops/attentions/hybrid/optimized_hybrid_attention.py:163` (`unfreeze_base_weights`)

### Group `[eb9ed767f4a9]` — 5 instances

- `neuralops/attentions/ssm/phantom.py:48` (`apply_rotary_pos_emb`)
- `neuralops/attentions/linear/nystrom_attention.py:35` (`apply_rotary_pos_emb`)
- `neuralops/attentions/linear/spectraformer_attention.py:46` (`apply_rotary_pos_emb`)
- `neuralops/attentions/sparse/padre_attention.py:46` (`apply_rotary_pos_emb`)
- `neuralops/attentions/sparse/nsa_attention.py:37` (`apply_rotary_pos_emb`)

### Group `[83e3126764c4]` — 5 instances

- `neuralops/attentions/fast_kernels/flash_attention3_fp8.py:69` (`apply_rotary_pos_emb`)
- `neuralops/attentions/fast_kernels/flash_attention3.py:59` (`apply_rotary_pos_emb`)
- `neuralops/attentions/svd/factorised_matrix_state.py:65` (`apply_rotary_pos_emb`)
- `neuralops/attentions/gated/decay_linear_attention.py:60` (`apply_rotary_pos_emb`)
- `neuralops/attentions/sparse/expert_choice_moe.py:50` (`apply_rotary_pos_emb`)

### Group `[1c2d0f098734]` — 5 instances

- `neuralops/attentions/linear/linformer_attention.py:133` (`__init__`)
- `neuralops/attentions/sparse/routing_attention.py:106` (`__init__`)
- `neuralops/attentions/sparse/lsh_attention.py:147` (`__init__`)
- `neuralops/attentions/hybrid/refined_landmark_attention.py:166` (`__init__`)
- `neuralops/attentions/hybrid/positive_kernel_linear_attention.py:152` (`__init__`)

### Group `[1f32d03515ad]` — 5 instances

- `neuralops/attentions/linear/linformer_attention.py:171` (`weight`)
- `neuralops/attentions/sparse/routing_attention.py:144` (`weight`)
- `neuralops/attentions/sparse/lsh_attention.py:185` (`weight`)
- `neuralops/attentions/hybrid/refined_landmark_attention.py:204` (`weight`)
- `neuralops/attentions/hybrid/positive_kernel_linear_attention.py:190` (`weight`)

### Group `[09c081f19db0]` — 5 instances

- `neuralops/attentions/linear/linformer_attention.py:369` (`_detect_hidden_size`)
- `neuralops/attentions/sparse/lsh_attention.py:361` (`_detect_hidden_size`)
- `neuralops/attentions/hybrid/landmark_local_attention.py:356` (`_detect_hidden_size`)
- `neuralops/attentions/hybrid/refined_landmark_attention.py:401` (`_detect_hidden_size`)
- `neuralops/attentions/hybrid/positive_kernel_linear_attention.py:346` (`_detect_hidden_size`)

### Group `[aa97494feb22]` — 5 instances

- `neuralops/attentions/linear/linformer_attention.py:391` (`_detect_num_heads`)
- `neuralops/attentions/sparse/lsh_attention.py:383` (`_detect_num_heads`)
- `neuralops/attentions/hybrid/landmark_local_attention.py:378` (`_detect_num_heads`)
- `neuralops/attentions/hybrid/refined_landmark_attention.py:423` (`_detect_num_heads`)
- `neuralops/attentions/hybrid/positive_kernel_linear_attention.py:368` (`_detect_num_heads`)

### Group `[9ab1a6c648e9]` — 5 instances

- `neuralops/attentions/linear/linformer_attention.py:411` (`_detect_num_kv_heads`)
- `neuralops/attentions/sparse/lsh_attention.py:403` (`_detect_num_kv_heads`)
- `neuralops/attentions/hybrid/landmark_local_attention.py:398` (`_detect_num_kv_heads`)
- `neuralops/attentions/hybrid/refined_landmark_attention.py:443` (`_detect_num_kv_heads`)
- `neuralops/attentions/hybrid/positive_kernel_linear_attention.py:388` (`_detect_num_kv_heads`)

### Group `[91344f9d2849]` — 4 instances

- `neuralops/attentions/ssm/nexus.py:43` (`apply_rotary_pos_emb`)
- `neuralops/attentions/svd/residual_svd_attention.py:32` (`apply_rotary_pos_emb`)
- `neuralops/attentions/sparse/routing_attention.py:44` (`apply_rotary_pos_emb`)
- `neuralops/attentions/hybrid/regla_attention.py:32` (`apply_rotary_pos_emb`)

### Group `[95366463001d]` — 4 instances

- `neuralops/attentions/linear/linformer_attention.py:84` (`_detect_decoder`)
- `neuralops/attentions/sparse/lsh_attention.py:98` (`_detect_decoder`)
- `neuralops/attentions/hybrid/refined_landmark_attention.py:88` (`_detect_decoder`)
- `neuralops/attentions/hybrid/positive_kernel_linear_attention.py:96` (`_detect_decoder`)

### Group `[24d9e915d39c]` — 4 instances

- `neuralops/attentions/linear/cosformer.py:47` (`repeat_kv`)
- `neuralops/attentions/linear/galerkin_attention.py:35` (`repeat_kv`)
- `neuralops/attentions/svd/svd_mamba.py:49` (`repeat_kv`)
- `neuralops/attentions/hybrid/fusion_attn.py:38` (`repeat_kv`)

### Group `[3a9edd25a6d1]` — 4 instances

- `neuralops/attentions/svd/fused_svd_kernel.py:380` (`weight`)
- `neuralops/compression_engine/svd/dynamic_rank.py:302` (`weight`)
- `neuralops/compression_engine/svd/svd_linear.py:141` (`weight`)
- `neuralops/compression_engine/low_rank/grecodec_compressor.py:113` (`weight`)

### Group `[839ee6a668e3]` — 4 instances

- `neuralops/attentions/svd/minimal_svd.py:171` (`weight`)
- `neuralops/attentions/svd/minimal_nova.py:191` (`weight`)
- `neuralops/attentions/hybrid/split_head_hybrid_attention.py:328` (`weight`)
- `neuralops/attentions/hybrid/hybrid_local_kernel_attention.py:248` (`weight`)

### Group `[79c7b8737baf]` — 4 instances

- `neuralops/attentions/sparse/routing_attention.py:164` (`forward`)
- `neuralops/attentions/hybrid/landmark_local_attention.py:194` (`forward`)
- `neuralops/attentions/hybrid/refined_landmark_attention.py:230` (`forward`)
- `neuralops/attentions/hybrid/positive_kernel_linear_attention.py:210` (`forward`)

### Group `[6eb4ce393b91]` — 4 instances

- `neuralops/attentions/sparse/lsh_attention.py:653` (`_resize_gqa`)
- `neuralops/attentions/hybrid/landmark_local_attention.py:772` (`_resize_gqa`)
- `neuralops/attentions/hybrid/refined_landmark_attention.py:811` (`_resize_gqa`)
- `neuralops/attentions/hybrid/positive_kernel_linear_attention.py:668` (`_resize_gqa`)

### Group `[9470fe2c89af]` — 4 instances

- `neuralops/attentions/sparse/reap_moe.py:75` (`_detect_decoder`)
- `neuralops/attentions/modern/qcqa_attention.py:80` (`_detect_decoder`)
- `neuralops/attentions/hybrid/spectron_fp8_attention.py:71` (`_detect_decoder`)
- `neuralops/attentions/hybrid/hybrid_local_kernel_attention.py:70` (`_detect_decoder`)

### Group `[b84197e91bda]` — 3 instances

- `neuralops/data/prepare_datasets.py:141` (`__getitem__`)
- `neuralops/data/synthetic_generator.py:173` (`__getitem__`)
- `neuralops/data/synthetic_generator.py:228` (`__getitem__`)

### Group `[b933e7c150ca]` — 3 instances

- `neuralops/attentions/ssm/vspec.py:48` (`apply_rotary_pos_emb`)
- `neuralops/attentions/modern/vig_attention.py:40` (`apply_rotary_pos_emb`)
- `neuralops/attentions/modern/wgqa_attention.py:44` (`apply_rotary_pos_emb`)

### Group `[6d2c86c85533]` — 3 instances

- `neuralops/attentions/ssm/phantom.py:149` (`freeze_base_weights`)
- `neuralops/attentions/svd/minimal_svd.py:159` (`freeze_base_weights`)
- `neuralops/attentions/svd/minimal_nova.py:179` (`freeze_base_weights`)

### Group `[7a7626a183d9]` — 3 instances

- `neuralops/attentions/ssm/orion.py:259` (`unfreeze_base_weights`)
- `neuralops/attentions/linear/linformer_attention.py:191` (`unfreeze_base_weights`)
- `neuralops/attentions/hybrid/refined_landmark_attention.py:224` (`unfreeze_base_weights`)

### Group `[7a070e26b939]` — 3 instances

- `neuralops/attentions/ssm/spectral_mamba.py:63` (`_unpack_weight_bias`)
- `neuralops/attentions/linear/spectra_v2.py:56` (`_unpack_weight_bias`)
- `neuralops/attentions/hybrid/granite_hybrid.py:36` (`_unpack_weight_bias`)

### Group `[e0b5c2ad6391]` — 3 instances

- `neuralops/attentions/experimental/template_attention.py:142` (`forward`)
- `neuralops/attentions/fast_kernels/soa_attn.py:207` (`forward`)
- `neuralops/attentions/svd/svd_attention.py:153` (`forward`)

### Group `[e71cfcd2a95c]` — 3 instances

- `neuralops/attentions/fast_kernels/flash_attention3_fp8.py:174` (`forward`)
- `neuralops/attentions/fast_kernels/flash_attention3.py:164` (`forward`)
- `neuralops/attentions/adaptive/block_transformer_hiba.py:156` (`forward`)

### Group `[7e81b756f9de]` — 3 instances

- `neuralops/attentions/linear/linformer_attention.py:430` (`_detect_head_dim`)
- `neuralops/attentions/sparse/lsh_attention.py:422` (`_detect_head_dim`)
- `neuralops/attentions/hybrid/positive_kernel_linear_attention.py:407` (`_detect_head_dim`)

### Group `[104ebf767ef4]` — 3 instances

- `neuralops/attentions/linear/nasc_attention.py:25` (`apply_rotary_pos_emb`)
- `neuralops/attentions/svd/minimal_svd.py:17` (`apply_rotary_pos_emb`)
- `neuralops/attentions/svd/minimal_nova.py:17` (`apply_rotary_pos_emb`)

### Group `[ef228ee31e25]` — 3 instances

- `neuralops/attentions/linear/tlinformer_attention.py:48` (`_cfg_get`)
- `neuralops/attentions/gated/gated_delta_net.py:43` (`_cfg_get`)
- `neuralops/attentions/adaptive/token_compressed_attention.py:51` (`_cfg_get`)

### Group `[94afc6fb4786]` — 3 instances

- `neuralops/attentions/linear/tlinformer_attention.py:87` (`__init__`)
- `neuralops/attentions/modern/gqa_attention.py:99` (`__init__`)
- `neuralops/attentions/hybrid/optimized_hybrid_attention.py:103` (`__init__`)

### Group `[0edb136652d1]` — 3 instances

- `neuralops/attentions/linear/rala_attention.py:31` (`apply_rotary_pos_emb`)
- `neuralops/attentions/sparse/reap_moe.py:30` (`apply_rotary_pos_emb`)
- `neuralops/attentions/modern/qcqa_attention.py:38` (`apply_rotary_pos_emb`)

### Group `[a231a7ea7239]` — 3 instances

- `neuralops/attentions/svd/minimal_svd.py:91` (`reset_parameters`)
- `neuralops/attentions/svd/minimal_nova.py:101` (`reset_parameters`)
- `neuralops/attentions/hybrid/hybrid_local_kernel_attention.py:174` (`reset_parameters`)

### Group `[6115da2c380c]` — 3 instances

- `neuralops/attentions/sparse/reap_moe.py:178` (`forward`)
- `neuralops/attentions/sparse/expert_choice_moe.py:155` (`forward`)
- `neuralops/attentions/hybrid/regla_attention.py:146` (`forward`)

### Group `[dea7fb67258d]` — 3 instances

- `neuralops/compression_engine/tensor_decomp/kronecker_compressor.py:614` (`_is_gqa_asymmetric`)
- `neuralops/compression_engine/tensor_decomp/kronecker_tt_lora.py:487` (`_is_gqa_asymmetric`)
- `neuralops/compression_engine/quantization/kmeans_quantize.py:474` (`_is_gqa_asymmetric`)

### Group `[c803034452c7]` — 3 instances

- `neuralops/compression_engine/quantization/coopq_quantizer.py:123` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/cr_compressor.py:25` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/cur_moe_compressor.py:24` (`_cfg_get`)

### Group `[6ed5ff73d424]` — 3 instances

- `neuralops/compression_engine/quantization/slicegpt_pca.py:290` (`_is_gqa_asymmetric`)
- `neuralops/compression_engine/low_rank/rpca_compressor.py:120` (`_is_gqa_asymmetric`)
- `neuralops/compression_engine/low_rank/salr.py:241` (`_is_gqa_asymmetric`)

### Group `[37baf7b5d4ba]` — 3 instances

- `neuralops/compression_engine/svd/optimal_subspace_projection.py:294` (`_is_gqa_asymmetric`)
- `neuralops/compression_engine/low_rank/mgie_svd_compressor.py:125` (`_is_gqa_asymmetric`)
- `neuralops/compression_engine/low_rank/mlorq_compressor.py:421` (`_is_gqa_asymmetric`)

### Group `[5f51e8878dea]` — 3 instances

- `neuralops/compression_engine/svd/gfws_d_compressor.py:19` (`_cfg_get`)
- `neuralops/compression_engine/svd/adaptive_ranks.py:39` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/compot_compressor.py:25` (`_cfg_get`)

### Group `[5a4dadeec0fa]` — 3 instances

- `neuralops/compression_engine/pruning/ast_pruner.py:47` (`_cfg_get`)
- `neuralops/compression_engine/pruning/bonsai_pruner.py:45` (`_cfg_get`)
- `neuralops/compression_engine/low_rank/a3_compressor.py:38` (`_cfg_get`)

### Group `[3981ec168db0]` — 2 instances

- `neuralops/core/common.py:357` (`reset_parameters`)
- `neuralops/attentions/hybrid/spectron_fp8_attention.py:203` (`reset_parameters`)

### Group `[eac58ba1a9c7]` — 2 instances

- `neuralops/core/common.py:379` (`weight`)
- `neuralops/attentions/hybrid/spectron_fp8_attention.py:216` (`weight`)

### Group `[66592d85d16a]` — 2 instances

- `neuralops/core/common.py:492` (`reset_parameters`)
- `neuralops/compression_engine/svd/residual_svd_compressor.py:101` (`reset_parameters`)

### Group `[b8fcfa8f358f]` — 2 instances

- `neuralops/core/common.py:510` (`weight`)
- `neuralops/compression_engine/svd/residual_svd_compressor.py:119` (`weight`)

### Group `[b29bd1f280aa]` — 2 instances

- `neuralops/core/utils.py:52` (`count_parameters`)
- `neuralops/utils/helpers.py:47` (`count_parameters`)

### Group `[9ebbe5e03766]` — 2 instances

- `neuralops/utils/layer_norm_fix.py:71` (`extra_repr`)
- `neuralops/utils/layer_norm_fix.py:142` (`extra_repr`)

### Group `[73555cca7b8b]` — 2 instances

- `neuralops/tracking/backends.py:61` (`close`)
- `neuralops/tracking/backends.py:118` (`close`)

### Group `[e4600a386d12]` — 2 instances

- `neuralops/attentions/ssm/goldfinch_ahn.py:44` (`apply_rotary_pos_emb`)
- `neuralops/attentions/ssm/mamba_svd_flash.py:33` (`apply_rotary_pos_emb`)

### Group `[08004474b646]` — 2 instances

- `neuralops/attentions/ssm/phantom.py:224` (`forward`)
- `neuralops/attentions/ssm/orion.py:265` (`forward`)

### Group `[ebc9181cfea1]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:33` (`apply_rotary_pos_emb`)
- `neuralops/attentions/adaptive/infini_attention.py:38` (`apply_rotary_pos_emb`)

### Group `[d4b81be7b68f]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:194` (`_detect_decoder`)
- `neuralops/attentions/adaptive/infini_attention.py:206` (`_detect_decoder`)

### Group `[11112fdb075d]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:205` (`_detect_return_format`)
- `neuralops/attentions/adaptive/infini_attention.py:217` (`_detect_return_format`)

### Group `[d1f738e1d046]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:213` (`_read_cache`)
- `neuralops/attentions/adaptive/infini_attention.py:228` (`_read_cache`)

### Group `[48ee985ee60c]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:241` (`_write_cache`)
- `neuralops/attentions/adaptive/infini_attention.py:256` (`_write_cache`)

### Group `[b328e1b79805]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:630` (`inject_jamba`)
- `neuralops/attentions/adaptive/infini_attention.py:644` (`inject_jamba`)

### Group `[abf58c262e85]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:84` (`__init__`)
- `neuralops/attentions/adaptive/infini_attention.py:89` (`__init__`)

### Group `[c8d3e12b67e2]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:124` (`freeze_base_weights`)
- `neuralops/attentions/adaptive/infini_attention.py:129` (`freeze_base_weights`)

### Group `[0fab0f704cf1]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:130` (`unfreeze_base_weights`)
- `neuralops/attentions/adaptive/infini_attention.py:135` (`unfreeze_base_weights`)

### Group `[66784a64dbc1]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:134` (`forward`)
- `neuralops/attentions/adaptive/infini_attention.py:139` (`forward`)

### Group `[087327c02fee]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:160` (`_ssm_accepts_past_state`)
- `neuralops/attentions/adaptive/infini_attention.py:169` (`_ssm_accepts_past_state`)

### Group `[0d488b69567f]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:168` (`forward`)
- `neuralops/attentions/adaptive/infini_attention.py:177` (`forward`)

### Group `[fa5b51dba6c0]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:355` (`aux_loss`)
- `neuralops/attentions/adaptive/infini_attention.py:372` (`aux_loss`)

### Group `[6ce5358b8ecf]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:359` (`reset_aux_loss`)
- `neuralops/attentions/adaptive/infini_attention.py:376` (`reset_aux_loss`)

### Group `[c0f46fcade32]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:363` (`freeze_base_weights`)
- `neuralops/attentions/adaptive/infini_attention.py:380` (`freeze_base_weights`)

### Group `[72982fb8327e]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:367` (`unfreeze_base_weights`)
- `neuralops/attentions/adaptive/infini_attention.py:384` (`unfreeze_base_weights`)

### Group `[d4fc2260af82]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:371` (`copy_weights`)
- `neuralops/attentions/adaptive/infini_attention.py:388` (`copy_weights`)

### Group `[75e6d27ee562]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:475` (`_format_output`)
- `neuralops/attentions/adaptive/infini_attention.py:492` (`_format_output`)

### Group `[b2c6b94da911]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:481` (`_normalize_head_mask`)
- `neuralops/attentions/adaptive/infini_attention.py:498` (`_normalize_head_mask`)

### Group `[02ab5c0fbc8a]` — 2 instances

- `neuralops/attentions/ssm/hymba_transx.py:744` (`_prepare_valid_mask`)
- `neuralops/attentions/hybrid/spectron_fp8_attention.py:864` (`_prepare_valid_mask`)

### Group `[c069b958b575]` — 2 instances

- `neuralops/attentions/fast_kernels/adasplash_attention.py:356` (`_relative_position_bucket`)
- `neuralops/attentions/sparse/adaptive_sparse_attention.py:471` (`_relative_position_bucket`)

### Group `[ec2ada9579d0]` — 2 instances

- `neuralops/attentions/fast_kernels/adasplash_attention.py:384` (`compute_bias`)
- `neuralops/attentions/sparse/adaptive_sparse_attention.py:499` (`compute_bias`)

### Group `[55c6d8659d6e]` — 2 instances

- `neuralops/attentions/linear/linformer_attention.py:197` (`forward`)
- `neuralops/attentions/sparse/lsh_attention.py:205` (`forward`)

### Group `[0939914cf899]` — 2 instances

- `neuralops/attentions/linear/cosformer.py:86` (`__init__`)
- `neuralops/attentions/hybrid/halo_regla.py:103` (`__init__`)

### Group `[a735bddc5dc7]` — 2 instances

- `neuralops/attentions/linear/tpa_attention.py:261` (`forward`)
- `neuralops/attentions/adaptive/token_compressed_attention.py:148` (`forward`)

### Group `[c06c46f42312]` — 2 instances

- `neuralops/attentions/linear/spectron.py:219` (`apply_rotary_pos_emb`)
- `neuralops/attentions/adaptive/squeezing_heads_distill.py:37` (`apply_rotary_pos_emb`)

### Group `[8233ea8dea4f]` — 2 instances

- `neuralops/attentions/linear/tlinformer_attention.py:59` (`apply_rotary_pos_emb`)
- `neuralops/attentions/gated/gated_delta_net.py:185` (`apply_rotary_pos_emb`)

### Group `[16a2d10d04b3]` — 2 instances

- `neuralops/attentions/linear/tlinformer_attention.py:635` (`_prepare_valid_mask`)
- `neuralops/attentions/hybrid/optimized_hybrid_attention.py:590` (`_prepare_valid_mask`)

### Group `[8af3004dff56]` — 2 instances

- `neuralops/attentions/linear/triton_cosformer.py:63` (`apply_rotary_pos_emb`)
- `neuralops/attentions/svd/triton_svd_linear_attention.py:66` (`apply_rotary_pos_emb`)

### Group `[9fb4c1571b5e]` — 2 instances

- `neuralops/attentions/linear/chela_slay.py:127` (`repeat_kv`)
- `neuralops/attentions/gated/gated_delta_net.py:36` (`repeat_kv`)

### Group `[5d1e0bce6cfa]` — 2 instances

- `neuralops/attentions/svd/compressed_attention.py:214` (`reset_lora_parameters`)
- `neuralops/attentions/svd/compressed_mha_attention.py:200` (`reset_lora_parameters`)

### Group `[d27521b51d7a]` — 2 instances

- `neuralops/attentions/svd/svd_projection_attention.py:49` (`_unpack_weight_bias`)
- `neuralops/attentions/svd/svd_mamba.py:34` (`_unpack_weight_bias`)

### Group `[8ce4fd8c743e]` — 2 instances

- `neuralops/attentions/svd/svd_attention.py:63` (`apply_rotary_pos_emb`)
- `neuralops/attentions/svd/svd_compressed.py:168` (`apply_rotary_pos_emb`)

### Group `[a1a919b9ab04]` — 2 instances

- `neuralops/attentions/svd/svd_attention.py:111` (`__init__`)
- `neuralops/attentions/gated/hla_attention.py:58` (`__init__`)

### Group `[2e52d5179cf0]` — 2 instances

- `neuralops/attentions/svd/minimal_svd.py:181` (`forward`)
- `neuralops/attentions/hybrid/split_head_hybrid_attention.py:338` (`forward`)

### Group `[bba98daa1dde]` — 2 instances

- `neuralops/attentions/svd/salt_attention.py:34` (`apply_rotary_pos_emb`)
- `neuralops/attentions/modern/unified_rope.py:37` (`apply_rotary_pos_emb`)

### Group `[7348bb339088]` — 2 instances

- `neuralops/attentions/gated/gated_linear_attention.py:58` (`apply_rotary_pos_emb`)
- `neuralops/attentions/hybrid/mixattention_moska.py:26` (`apply_rotary_pos_emb`)

### Group `[7f080a9bdf42]` — 2 instances

- `neuralops/attentions/sparse/padre_attention.py:37` (`_cfg_get`)
- `neuralops/compression_engine/experimental/roast_hashed_linear.py:25` (`_cfg_get`)

### Group `[44d9114030f9]` — 2 instances

- `neuralops/attentions/sparse/moba_attention.py:27` (`apply_rotary_pos_emb`)
- `neuralops/attentions/modern/mla_attention.py:40` (`apply_rotary_pos_emb`)

### Group `[d6afdfaf6dcf]` — 2 instances

- `neuralops/attentions/sparse/moba_attention.py:835` (`_sz`)
- `neuralops/attentions/hybrid/mixattention_moska.py:857` (`_sz`)

### Group `[1ce766614366]` — 2 instances

- `neuralops/attentions/modern/vig_attention.py:88` (`__init__`)
- `neuralops/attentions/modern/wgqa_attention.py:92` (`__init__`)

### Group `[2dbd43f1addf]` — 2 instances

- `neuralops/attentions/modern/nha_attention.py:89` (`__init__`)
- `neuralops/attentions/adaptive/progressive_token_pruning.py:99` (`__init__`)

### Group `[096d63e65f70]` — 2 instances

- `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py:519` (`_detect_decoder`)
- `neuralops/attentions/hybrid/hybrid_attention.py:108` (`_detect_decoder`)

### Group `[5b18b00a6132]` — 2 instances

- `neuralops/attentions/hybrid/landmark_local_attention.py:417` (`_detect_head_dim`)
- `neuralops/attentions/hybrid/refined_landmark_attention.py:462` (`_detect_head_dim`)

### Group `[4380f9409225]` — 2 instances

- `neuralops/attentions/hybrid/landmark_local_attention.py:459` (`transpose_for_scores`)
- `neuralops/attentions/hybrid/refined_landmark_attention.py:496` (`transpose_for_scores`)

### Group `[45eaa3c42cf0]` — 2 instances

- `neuralops/compression_engine/experimental/lillama_compressor.py:63` (`_is_conv1d`)
- `neuralops/compression_engine/kv_cache/delta_kv.py:180` (`_is_conv1d`)

### Group `[dfe3406912a8]` — 2 instances

- `neuralops/compression_engine/experimental/hashed_net_compressor.py:236` (`_sz`)
- `neuralops/compression_engine/quantization/kmeans_quantize.py:364` (`_sz`)

### Group `[2860924dc3a3]` — 2 instances

- `neuralops/compression_engine/quantization/slicegpt.py:302` (`_detect_lora`)
- `neuralops/compression_engine/quantization/slicegpt_pca.py:476` (`_detect_lora`)

### Group `[e5c52e7b5b0c]` — 2 instances

- `neuralops/compression_engine/quantization/astro_quantizer.py:35` (`_cfg_get`)
- `neuralops/compression_engine/kv_cache/cachenotes_compressor.py:42` (`_cfg_get`)

### Group `[643cc81b7da2]` — 2 instances

- `neuralops/compression_engine/svd/optimal_subspace_projection.py:199` (`forward`)
- `neuralops/compression_engine/low_rank/oats_compressor.py:183` (`forward`)

### Group `[62424b387bdc]` — 2 instances

- `neuralops/compression_engine/svd/fwsvd_compressor.py:116` (`_is_target_module`)
- `neuralops/compression_engine/low_rank/drone_compressor.py:84` (`_is_target_module`)

### Group `[2685d0c70a55]` — 2 instances

- `neuralops/compression_engine/svd/aa_svd_compressor.py:161` (`get_weight`)
- `neuralops/compression_engine/low_rank/a3_compressor.py:125` (`get_weight`)

### Group `[977d4de76ab5]` — 2 instances

- `neuralops/compression_engine/pruning/sparsegpt.py:154` (`make_hook`)
- `neuralops/compression_engine/low_rank/oats_compressor.py:360` (`make_hook`)

### Group `[240db7134257]` — 2 instances

- `neuralops/compression_engine/pruning/sparsegpt.py:155` (`hook`)
- `neuralops/compression_engine/low_rank/oats_compressor.py:361` (`hook`)

### Group `[46c7abf22ef6]` — 2 instances

- `neuralops/compression_engine/kv_cache/cachenotes_compressor.py:57` (`__init__`)
- `neuralops/compression_engine/low_rank/come_compressor.py:217` (`__init__`)

### Group `[0262c9375050]` — 2 instances

- `neuralops/compression_engine/kv_cache/cachenotes_compressor.py:142` (`forward`)
- `neuralops/compression_engine/low_rank/come_compressor.py:304` (`forward`)

### Group `[3bacf310a2e2]` — 2 instances

- `neuralops/core/triton_kernels/_types.py:46` (`log_fallback_reason`)
- `neuralops/core/triton_kernels/_common.py:58` (`log_fallback_reason`)

## Duplicate Class Bodies (12 groups)

### Group `[ca679bc0ceaf]` — 5 instances

- `neuralops/attentions/ssm/spectral_mamba.py:116` (`LoRALinear`)
- `neuralops/attentions/linear/primal_attention.py:91` (`LoRALinear`)
- `neuralops/attentions/linear/rala_attention.py:78` (`LoRALinear`)
- `neuralops/attentions/modern/qcqa_attention.py:121` (`LoRALinear`)
- `neuralops/attentions/adaptive/span_expanded_attention.py:80` (`LoRALinear`)

### Group `[4068b352f0f0]` — 3 instances

- `neuralops/attentions/fast_kernels/xattention.py:144` (`LoRALinear`)
- `neuralops/attentions/linear/nasc_attention.py:84` (`LoRALinear`)
- `neuralops/attentions/dense/zeros_attention.py:85` (`LoRALinear`)

### Group `[1e56926a21fe]` — 3 instances

- `neuralops/attentions/linear/nystrom_attention.py:69` (`LoRALinear`)
- `neuralops/attentions/gated/kda_attention.py:74` (`LoRALinear`)
- `neuralops/attentions/sparse/nsa_attention.py:71` (`LoRALinear`)

### Group `[e6972cd0b3ad]` — 3 instances

- `neuralops/attentions/dense/full_attention.py:72` (`LoRALinear`)
- `neuralops/attentions/sparse/cluster_attention.py:72` (`LoRALinear`)
- `neuralops/attentions/sparse/hamming_attention.py:133` (`LoRALinear`)

### Group `[2e826315de68]` — 2 instances

- `neuralops/attentions/ssm/goldfinch_ahn.py:73` (`LoRALinear`)
- `neuralops/attentions/ssm/mamba_svd_flash.py:62` (`LoRALinear`)

### Group `[d8d43d8314f1]` — 2 instances

- `neuralops/attentions/ssm/jamba.py:83` (`LoRALinear`)
- `neuralops/attentions/adaptive/infini_attention.py:88` (`LoRALinear`)

### Group `[0b403aeba65c]` — 2 instances

- `neuralops/attentions/fast_kernels/flash_attention3_fp8.py:104` (`LoRALinear`)
- `neuralops/attentions/fast_kernels/flash_attention3.py:94` (`LoRALinear`)

### Group `[404629151de5]` — 2 instances

- `neuralops/attentions/linear/spectraformer_attention.py:80` (`LoRALinear`)
- `neuralops/attentions/sparse/padre_attention.py:80` (`LoRALinear`)

### Group `[c3ee225ba2e8]` — 2 instances

- `neuralops/attentions/svd/factorised_matrix_state.py:100` (`LoRALinear`)
- `neuralops/attentions/gated/decay_linear_attention.py:95` (`LoRALinear`)

### Group `[0bb15fa4074a]` — 2 instances

- `neuralops/attentions/gated/gated_delta_net.py:212` (`LoRALinear`)
- `neuralops/attentions/hybrid/mixattention_moska.py:61` (`LoRALinear`)

### Group `[a3fbd97d13a1]` — 2 instances

- `neuralops/attentions/sparse/routing_attention.py:97` (`FactorizedLoRALinear`)
- `neuralops/attentions/hybrid/positive_kernel_linear_attention.py:142` (`FactorizedLoRALinear`)

### Group `[2897a7117722]` — 2 instances

- `neuralops/attentions/modern/vig_attention.py:85` (`LoRALinear`)
- `neuralops/attentions/modern/wgqa_attention.py:89` (`LoRALinear`)

## Recommendations

- Where identical boilerplates exist (e.g. standard RoPE embedding or projection helpers), retain local self-containment for attention standalone independence, or extract shared utilities into `neuralops.core`.
