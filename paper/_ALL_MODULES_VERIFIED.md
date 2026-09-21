# All Modules Verified — Final Summary

**Generated**: 2026-09-21T08:49:48.048275

**Total Modules**: 217

## Classification Breakdown

| Classification | Count | % |
|----------------|-------|---|
| VALID | 46 | 21.2% |
| QUESTIONABLE | 1 | 0.5% |
| COLLAPSED | 0 | 0.0% |
| FALLBACK | 142 | 65.4% |
| BROKEN | 4 | 1.8% |
| NO_DATA | 24 | 11.1% |

**Modules with fabricated numbers**: 152


## ✅ VALID Modules (46)

- `SOAFusedAttention`: 76.30% @ 1.145x
- `a3`: 100.00% @ 1.094x
- `aa_svd_compressor`: 85.80% @ 1.133x
- `aasvd`: 94.50% @ 1.016x
- `adaptive_compressor`: 84.50% @ 1.002x
- `adaptive_ranks`: 84.50% @ 1.002x
- `adaptive_svd_attention`: 74.60% @ 1.152x
- `aegis_hybrid`: 93.00% @ 1.050x
- `aegis_spectron_fp8`: 79.59% @ 1.223x
- `aegis_v2`: 84.17% @ 1.181x
- `aegis_v3`: 90.10% @ 1.197x
- `aegis_v4`: 89.80% @ 1.197x
- `aegis_v5`: 89.80% @ 1.197x
- `aegis_v6`: 93.30% @ 1.252x
- `come`: 91.50% @ 1.241x
- `compressed_attention`: 78.60% @ 1.110x
- `compressed_mha`: 76.60% @ 1.110x
- `cur_moe`: 94.30% @ 1.056x
- `fused_svd_kernel`: 91.90% @ 1.076x
- `hybrid_svd_layer`: 77.18% @ 1.907x
- `lola_attention`: 77.98% @ 1.227x
- `mgie_svd`: 60.09% @ 1.003x
- `minimal_aurora`: 88.30% @ 1.152x
- `minimal_nova`: 88.30% @ 1.152x
- `minimal_prism`: 88.30% @ 1.152x
- `minimal_svd`: 88.30% @ 1.152x
- `minimal_svd_v2`: 88.30% @ 1.152x
- `orion`: 81.08% @ 1.060x
- `phantom`: 72.59% @ 1.227x
- `projection_factorized_attn`: 94.50% @ 1.065x
- `res_svd`: 92.70% @ 1.036x
- `rpca`: 60.21% @ 1.003x
- `salt`: 91.40% @ 1.076x
- `sdrm`: 91.50% @ 1.212x
- `soa_attn`: 93.90% @ 1.111x
- `soa_fused_module`: 81.77% @ 1.162x
- `soa_module`: 89.22% @ 1.111x
- `soka`: 60.78% @ 1.003x
- `spectron`: 91.90% @ 1.152x
- `subspace_projection`: 59.52% @ 1.003x
- `svd_compressed`: 94.80% @ 1.066x
- `svd_linear_decay`: 94.50% @ 1.065x
- `svd_proj_attn`: 76.15% @ 1.227x
- `swift_svd`: 89.30% @ 1.344x
- `swift_svd_act`: 60.09% @ 1.003x
- `vspec`: 93.50% @ 1.036x

## ❌ NO_DATA Modules — Need GPU Runs (24)

- `AdaptiveSVD`
- `AdaptiveSpanAttention`
- `AdaptiveSparseAttention`
- `HALO`
- `OH`
- `SqueezingHeadsAttentionDistillationLayer`
- `cr_compressor`
- `cur_compressor`
- `enlighten_compressor`
- `fusion`
- `gfws_d_compressor`
- `hashed_net_compressor`
- `kronecker_compressor`
- `mlorq_compressor`
- `padre_attention`
- `pruning_utils`
- `qcqa_attention`
- `rala_attention`
- `routing_attention`
- `saes_svd`
- `span_expanded_attention`
- `token_compressed_attention`
- `tpa_attention`
- `vig_attention`

## ⚠️ Modules with Fabricated Numbers (152)

- `AdaptiveSVD`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36]
- `AdaptiveSpanAttention`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36]
- `AdaptiveSparseAttention`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36]
- `Galerkin`: [90.71, 90.71, 90.71, 90.25, 90.71, 71.22]
- `HALO`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36, 90.25, 78.78]
- `OH`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36, 90.25, 78.78, 89.5, 76.5, 77.5]
- `Primal`: [78.56, 78.56, 78.78]
- `SOAFusedAttention`: [90.71, 78.56]
- `SqueezingHeadsAttentionDistillationLayer`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36]
- `adaptive_compressor`: [79.36]
- `adaptive_span_attention`: [90.71, 90.48, 77.52, 85.09]
- `adaptive_sparse_attention`: [67.55, 78.56, 90.71, 79.36, 67.89, 78.56]
- `adaptive_svd_attention`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36]
- `adasvd_compressor`: [90.71, 86.01, 79.36, 63.76, 67.55, 78.56, 73.39]
- `admm_compressor`: [90.71, 90.37, 67.55, 88.42, 78.56, 62.04]
- `aegis_v1`: [90.71]
- `aegis_v2`: [90.71, 79.36, 78.56]
- `aegis_v4`: [78.56]
- `armor_pruner`: [67.55]
- `astro_quantizer`: [90.71, 90.71, 79.36, 77.87, 67.55, 87.5, 78.56, 74.77]
- `cachenotes_compressor`: [90.71, 90.6, 74.77, 67.55, 89.22, 78.56]
- `come`: [67.89]
- `compressed_attention`: [90.71]
- `cosformer`: [90.71, 90.71, 90.25, 90.71, 71.22, 58.72]
- `cur_compressor`: [90.71, 90.71, 79.36, 79.36, 78.56, 78.56]
- `cur_linear`: [67.89]
- `cure`: [67.89]
- `decay_linear_attention`: [90.71, 79.36, 78.56]
- `direct_svd`: [79.36, 78.56]
- `dobi`: [79.36, 67.89]
- `dynamic_entropy`: [90.71, 79.36, 67.89, 78.56]
- `enlighten_compressor`: [90.71, 89.22, 79.36, 61.12, 78.56, 51.03]
- `factorised_matrix_state`: [90.71, 79.36, 78.56]
- `faq_quantizer`: [90.71, 90.48, 79.36, 78.56, 78.56, 80.05]
- `flash3`: [88.0]
- `flash_attention3`: [90.71, 78.56, 79.36]
- `flash_attention3_fp8`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36, 90.25, 78.78, 90.71, 64.56, 78.44, 79.13, 50.92, 71.22]
- `full_attention`: [90.71, 79.36, 78.56]
- `fusion`: [90.71, 78.56, 79.36, 90.71, 90.25, 90.25]
- `galerkin_attention`: [90.71, 90.71]
- `gated_delta_net`: [90.71, 78.56, 79.36, 90.71, 90.25]
- `gated_deltanet_block`: [90.71, 78.56, 79.36, 90.71, 86.5, 90.71, 86.5]
- `gated_linear_attention`: [90.48, 75.0]
- `gqa`: [78.56, 79.36, 78.56, 79.36, 78.56, 79.36, 78.78]
- `granite_hybrid`: [57.91, 75.0]
- `griffin_ssd`: [79.36, 78.56]
- `hamming_attention`: [90.71, 79.13, 79.36, 71.22, 78.56]
- `hashed_linear`: [60.78, 67.89, 75.11]
- `hgrn2_gated_deltanet`: [90.71, 78.56, 79.36]
- `hgrn2_layer`: [90.71, 78.56, 79.36]
- `hgrn2_outer`: [90.94, 75.46]
- `hla_attention`: [90.94, 57.91, 75.46, 90.94, 57.91, 75.46]
- `huffman_coder`: [57.91, 67.89, 75.46]
- `hybrid_svd_layer`: [90.94, 57.91, 67.89]
- `hybridkv`: [75.11]
- `hymba_transx`: [78.56, 79.36]
- `infini`: [78.56, 79.36, 78.56, 79.36, 78.78, 78.78]
- `jamba`: [75.46]
- `jamba_info`: [78.56, 79.36]
- `kda_attention`: [90.71, 76.83, 79.36, 78.56]
- `kronecker_compressor`: [90.48, 90.94, 60.78, 60.32, 75.11, 70.07]
- `kronecker_linear`: [67.89]
- `leka_v1`: [90.71, 78.56, 79.36]
- `leka_v3`: [90.71, 90.71, 90.25, 90.71]
- `liftr`: [75.0]
- `lillama_compressor`: [90.48, 87.5, 56.88, 75.0, 85.44]
- `linformer_attention`: [90.71, 79.36]
- `lola_attention`: [90.71]
- `lsc_compressor`: [90.48, 90.48, 60.78, 62.04, 75.11, 72.02]
- `lsh_attention`: [90.71, 78.56, 90.71, 78.56]
- `mamba`: [90.71, 79.36, 78.56]
- `mamba_inject`: [90.94, 57.91, 75.46]
- `mamba_shedder`: [75.11]
- `mamba_svd_flash`: [90.71, 79.36, 78.56]
- `mgie_svd`: [75.11]
- `minimal_aurora`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36, 75.5, 76.8, 75.5, 76.8, 75.5, 76.8, 74.8, 75.5]
- `minimal_nova`: [90.71, 78.56, 79.36, 90.71, 78.56]
- `minimal_prism`: [90.71, 78.56, 79.36, 90.71, 78.56]
- `minimal_svd`: [90.71, 78.56, 79.36, 90.71, 78.56]
- `minimal_svd_v2`: [90.71, 79.36, 78.56]
- `mixattention_moska`: [78.56, 78.56, 78.78, 87.27]
- `mla_attention`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36]
- `mlorq_compressor`: [90.48, 90.48, 60.78, 62.04, 75.11, 72.02]
- `moba`: [78.56, 79.36, 78.56, 79.36, 78.78, 64.56, 78.44]
- `modegpt`: [75.11]
- `moe_deltakv_paged`: [90.71, 78.56, 79.36]
- `nasc_attention`: [78.56, 79.36, 78.56, 79.36, 85.32]
- `nli_softmax`: [79.36, 78.56]
- `nmf_compressor`: [90.71, 90.71, 79.36, 79.36, 78.56, 78.56]
- `nsa`: [78.56, 79.36, 78.56, 79.36]
- `nystrom_attention`: [90.71, 78.56, 79.36, 90.71]
- `oats`: [79.36]
- `optimized_hybrid_attention`: [67.89]
- `orion`: [90.71, 78.56, 90.71, 78.56]
- `padre_attention`: [90.71, 78.56, 79.36]
- `performer_attention`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36, 77.5]
- `phantom`: [90.71, 78.56, 79.36, 90.71, 78.56]
- `proxsparse`: [90.48]
- `proxsparse_pruner`: [90.48, 60.78, 75.11, 90.48, 60.78, 75.11]
- `pruning_utils`: [89.42, 76.8, 77.15]
- `ptp`: [78.56, 79.36, 78.56, 79.36]
- `qcqa_attention`: [90.48, 63.76, 60.78, 50.92, 75.11, 51.03]
- `rala_attention`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36, 90.25, 78.78, 90.71, 64.56, 78.44]
- `regla_attention`: [90.71, 79.82, 79.36, 57.45, 78.56, 64.79]
- `replace`: [90.48, 60.78, 75.11]
- `replaceme_compressor`: [90.48, 82.0, 60.78, 56.77, 75.11, 80.62]
- `roast_hashed_linear`: [57.91, 75.46]
- `rotational_slicegpt_int4`: [75.11]
- `routing_attention`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36, 90.25, 78.78, 90.71, 64.56, 78.44, 90.71, 79.36]
- `rpc`: [78.56, 79.36, 78.56, 79.36, 78.78]
- `saes_svd`: [89.42, 76.8, 77.15]
- `saes_svd_v2`: [90.48, 60.78, 75.11]
- `salr`: [75.11]
- `sdrm`: [75.46]
- `simple_ssm`: [90.71, 79.36, 78.56]
- `slicegpt`: [90.48, 75.11]
- `slicegpt_pca`: [75.11]
- `slim_compressor`: [90.48, 87.96, 60.78, 52.41, 75.11, 50.92]
- `smart_weighted_ensemble`: [75.11]
- `soa_fused_module`: [90.94, 87.84, 75.46]
- `soa_module`: [79.36, 78.56]
- `soka`: [75.11]
- `span_expanded_attention`: [90.71, 90.71, 79.36, 79.24, 78.56, 81.65]
- `sparsegpt`: [60.78]
- `spectraformer_attention`: [90.48, 60.78, 75.11, 90.48, 60.78, 75.11]
- `spectral_mamba`: [75.46]
- `spectron`: [79.36, 78.56]
- `squeezing_heads_distill`: [60.78, 75.11]
- `squeezing_heads_distiller`: [60.78]
- `svd_attention`: [90.71, 79.36, 78.56]
- `svd_linear_attention`: [90.71, 78.56, 79.36, 90.71, 78.56, 79.36, 88.5, 75.5, 76.8, 89.5, 75.5, 76.8, 85.5, 72.5, 73.2]
- `svd_linear_decay`: [57.91]
- `svd_llm_v2_attention`: [90.71, 78.56, 79.36, 90.71, 78.56]
- `svd_mamba`: [78.56, 79.36, 78.56, 79.36, 75.5, 76.8, 78.78, 76.5, 77.8]
- `svd_proj_attn`: [90.48, 60.78]
- `svd_projection_attention`: [90.48, 75.11]
- `svd_sequence_attention`: [90.94, 57.91, 75.46, 90.94, 57.91, 75.46]
- `template_attention`: [60.78, 75.11]
- `tlinformer_attention`: [57.91, 75.46, 57.91, 75.46]
- `token_compressed_attention`: [90.71, 78.56, 79.36]
- `tomoe`: [60.78, 75.11]
- `tpa_attention`: [90.71, 75.0, 79.36, 75.46, 59.75]
- `unified_rope`: [78.56, 79.36]
- `uniql_compressor`: [90.48, 90.14, 60.78, 60.89, 75.11, 82.45]
- `upi_mamba`: [90.94, 57.91, 75.46]
- `upi_stuffed_mamba`: [90.94, 57.91, 75.46]
- `vig_attention`: [90.94, 88.3, 57.91, 61.35, 75.46, 61.24]
- `weighted_ensemble`: [75.11]
- `wgqa`: [75.46]
- `xattention`: [90.94]
- `zerokv_neo`: [90.94, 87.84]
- `zeros_attention`: [90.94, 75.46]
