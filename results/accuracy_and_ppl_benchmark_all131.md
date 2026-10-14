# Attention Modules Benchmark: Zero-Shot Accuracy & Perplexity (All 131 Modules)

- **Hardware Split**: RTX 4070 (modules 1-66) + GTX 1060 (modules 67-131)
- **RoBERTa SST-2 Baseline Accuracy**: 95.00%
- **GPT-2 WikiText-2 Baseline PPL**: 27.50
- **Total Modules Evaluated**: 131

| # | Module | RoBERTa SST-2 Acc (%) | GPT-2 WikiText-2 PPL | Eval Time (s) | Status |
|---|---|---|---|---|---|
| - | **BASELINE** | **95.00%** | **27.50** | - | Reference |
| 1 | `adaptive_span_attention` | 95.00% | 11458.20 | 2.18s | OK |
| 2 | `adaptive_sparse_attention` | 92.00% | 27.54 | 1.56s | OK |
| 3 | `adaptive_svd_attention` | 52.00% | 1333.74 | 8.36s | OK |
| 4 | `adasplash_attention` | 94.00% | 75.70 | 1.98s | OK |
| 5 | `aegis_hybrid` | 52.00% | N/A | 8.22s | ERR: GPT2 error: expected mat1 |
| 6 | `aegis_spectron_fp8` | 52.00% | 2666.09 | 13.38s | OK |
| 7 | `aegis_v1` | 52.00% | 1891.05 | 9.79s | OK |
| 8 | `aegis_v2` | 52.00% | 1965.27 | 8.73s | OK |
| 9 | `aegis_v3` | 52.00% | 2015.07 | 8.79s | OK |
| 10 | `aegis_v4` | 52.00% | 2082.57 | 9.72s | OK |
| 11 | `aegis_v5` | 52.00% | N/A | 8.20s | ERR: GPT2 error: expected mat1 |
| 12 | `aegis_v6` | 52.00% | 1787.84 | 9.06s | OK |
| 13 | `alibi` | 92.00% | 45221.59 | 2.06s | OK |
| 14 | `asvd_attention` | 79.00% | 141.96 | 8.48s | OK |
| 15 | `bam_umoe` | 52.00% | 749.82 | 5.15s | OK |
| 16 | `block_transformer_hiba` | 93.00% | 112.24 | 2.32s | OK |
| 17 | `chela_slay` | 89.00% | 47.03 | 4.83s | OK |
| 18 | `chela_slay_attention` | 91.00% | 45.51 | 2.04s | OK |
| 19 | `cluster_attention` | 94.00% | 32.34 | 2.24s | OK |
| 20 | `compressed_attention` | 52.00% | 1355.87 | 4.89s | OK |
| 21 | `compressed_mha_attention` | 52.00% | 1384.04 | 5.24s | OK |
| 22 | `cosformer` | 95.00% | 569.27 | 2.73s | OK |
| 23 | `cosformer_attention` | 95.00% | 478.08 | 1.95s | OK |
| 24 | `cross_attention` | 93.00% | 27.50 | 1.57s | OK |
| 25 | `decay_linear_attention` | 66.00% | 18157.22 | 2.21s | OK |
| 26 | `decay_linear_attn` | 66.00% | 18157.22 | 1.97s | OK |
| 27 | `dynamic_conv_attention` | 52.00% | 3096182.99 | 1.82s | OK |
| 28 | `eat` | 90.00% | 31.43 | 1.87s | OK |
| 29 | `eat_attention` | 92.00% | 29.83 | 2.20s | OK |
| 30 | `expert_choice_moe` | 92.00% | 32.75 | 1.58s | OK |
| 31 | `factorised_matrix_state` | 70.00% | 25843.83 | 1.73s | OK |
| 32 | `flash_attention3` | 92.00% | 31.31 | 1.96s | OK |
| 33 | `flash_attention3_fp8` | 94.00% | 31.25 | 1.51s | OK |
| 34 | `full` | 95.00% | 27.50 | 1.17s | OK |
| 35 | `full_attention` | 93.00% | 33.68 | 2.15s | OK |
| 36 | `fused_svd_kernel` | 52.00% | 1297.06 | 8.45s | OK |
| 37 | `fusion_attn` | 94.00% | 113.70 | 1.44s | OK |
| 38 | `galerkin_attention` | 54.00% | 33.39 | 1.49s | OK |
| 39 | `gated_delta_net` | 52.00% | 1436.25 | 2.90s | OK |
| 40 | `gated_deltanet_block` | 52.00% | 1425.05 | 3.52s | OK |
| 41 | `gated_linear_attention` | 66.00% | 18157.22 | 1.71s | OK |
| 42 | `gated_linear_attn` | 66.00% | 18157.22 | 3.29s | OK |
| 43 | `gla_attention` | 55.00% | 10446.07 | 2.91s | OK |
| 44 | `goldfinch_ahn` | 95.00% | 8863.33 | 1.61s | OK |
| 45 | `gqa_attention` | 94.00% | 32.16 | 1.83s | OK |
| 46 | `granite_hybrid` | 52.00% | 3949.75 | 3.92s | OK |
| 47 | `griffin_ssd` | 91.00% | 127.52 | 1.47s | OK |
| 48 | `halo_regla` | 91.00% | 4164.81 | 1.49s | OK |
| 49 | `hamming_attention` | 60.00% | 2851.32 | 1.44s | OK |
| 50 | `hgrn2_gated_deltanet` | 92.00% | 4581.57 | 3.85s | OK |
| 51 | `hgrn2_gated_layer` | 92.00% | 4597.19 | 2.31s | OK |
| 52 | `hgrn2_gated_outer` | 52.00% | 4354.63 | 3.02s | OK |
| 53 | `hgrn2_outer` | 52.00% | 4427.27 | 2.51s | OK |
| 54 | `hla_attention` | 50.00% | 4669.51 | 2.01s | OK |
| 55 | `hybrid_attention` | 52.00% | 104.04 | 2.08s | OK |
| 56 | `hybrid_local_kernel` | 52.00% | 2086.72 | 8.62s | OK |
| 57 | `hymba_transx` | 94.00% | 92.17 | 1.75s | OK |
| 58 | `jamba` | 93.00% | 31.75 | 2.40s | OK |
| 59 | `jamba_info` | 55.00% | 2807.01 | 2.16s | OK |
| 60 | `kda_attention` | 51.00% | 404015.41 | 2.02s | OK |
| 61 | `landmark_local_attention` | 52.00% | 1159.19 | 8.93s | OK |
| 62 | `linformer_attention` | 52.00% | 1491.44 | 9.03s | OK |
| 63 | `lola_attention` | 52.00% | 1311.49 | 8.53s | OK |
| 64 | `lora_attention` | 94.00% | 30.71 | 1.51s | OK |
| 65 | `lsh_attention` | 52.00% | 1109.58 | 8.23s | OK |
| 66 | `mamba_block` | 52.00% | 4556.13 | 5.86s | OK |
| 67 | `mamba_svd_flash` | 95.00% | 32.47 | 7.06s | OK |
| 68 | `minimal_nova` | 52.00% | 1272.12 | 17.43s | OK |
| 69 | `minimal_svd` | 52.00% | 1370.12 | 23.01s | OK |
| 70 | `mixattention_moska` | 91.00% | 31.85 | 2.64s | OK |
| 71 | `mla_attention` | 52.00% | 667.74 | 15.30s | OK |
| 72 | `moba_attention` | 94.00% | 44.74 | 3.58s | OK |
| 73 | `moe_deltakv_paged` | 92.00% | N/A | 3.26s | ERR: GPT2 error: expected mat1 |
| 74 | `mqa` | 52.00% | 986.86 | 2.73s | OK |
| 75 | `naive_svd` | 52.00% | 1262.22 | 23.36s | OK |
| 76 | `nasc_attention` | 92.00% | 321.64 | 2.88s | OK |
| 77 | `native_hybrid_attention` | 95.00% | 5791.01 | 3.24s | OK |
| 78 | `nexus` | 56.00% | N/A | 2.92s | ERR: GPT2 error: mat1 and mat2 |
| 79 | `nsa_attention` | 95.00% | 33.31 | 2.70s | OK |
| 80 | `nystrom_attention` | 94.00% | 1209.13 | 2.93s | OK |
| 81 | `optimized_hybrid_attention` | 95.00% | 7040.08 | 2.84s | OK |
| 82 | `orion` | 52.00% | 1894.81 | 25.31s | OK |
| 83 | `padre_attention` | 92.00% | 358.82 | 2.63s | OK |
| 84 | `performer_attention` | 74.00% | 14221.36 | 2.67s | OK |
| 85 | `phantom` | 52.00% | 1232.98 | 25.36s | OK |
| 86 | `positive_kernel_linear_attention` | 52.00% | 834.27 | 23.93s | OK |
| 87 | `primal_attention` | 94.00% | 29.92 | 22.71s | OK |
| 88 | `progressive_token_pruning` | 94.00% | 132.52 | 2.58s | OK |
| 89 | `projection_factorized_attn` | 52.00% | 3595.72 | 13.03s | OK |
| 90 | `qcqa_attention` | 52.00% | 33.83 | 3.13s | OK |
| 91 | `rala_attention` | 75.00% | 22199.22 | 4.12s | OK |
| 92 | `reap_moe` | 95.00% | 33.25 | 3.68s | OK |
| 93 | `refined_landmark_attention` | 52.00% | 1800.99 | 24.22s | OK |
| 94 | `regla_attention` | 73.00% | 941.67 | 2.43s | OK |
| 95 | `residual_svd_attention` | 52.00% | 1277.10 | 17.52s | OK |
| 96 | `routing_attention` | 52.00% | 1408.10 | 24.14s | OK |
| 97 | `rpc_attention` | 52.00% | 2250.15 | 23.73s | OK |
| 98 | `salt_attention` | 52.00% | 747.84 | 59.67s | OK |
| 99 | `samba_m3` | 85.00% | 111.16 | 3.78s | OK |
| 100 | `samba_moe` | 57.00% | 622.41 | 5.97s | OK |
| 101 | `selective_ssm` | 52.00% | 5314.13 | 5.93s | OK |
| 102 | `simple_ssm` | 52.00% | 5612.84 | 6.37s | OK |
| 103 | `sink_attention` | 94.00% | 840.82 | 3.75s | OK |
| 104 | `soa_attn` | 52.00% | 591.59 | 20.98s | OK |
| 105 | `spectraformer_attention` | 49.00% | 879844.09 | 2.98s | OK |
| 106 | `spectron` | 52.00% | 3595.72 | 23.65s | OK |
| 107 | `spectron_fp8` | 52.00% | 2579.80 | 35.93s | OK |
| 108 | `split_head_hybrid` | 52.00% | N/A | 24.34s | ERR: GPT2 error: expected mat1 |
| 109 | `svd_compressed` | 52.00% | 3567.74 | 23.24s | OK |
| 110 | `svd_linear` | 52.00% | 3567.74 | 13.79s | OK |
| 111 | `svd_linear_attention` | 52.00% | 3567.74 | 14.76s | OK |
| 112 | `svd_linear_attn` | 52.00% | 3595.72 | 13.48s | OK |
| 113 | `svd_linear_decay` | 52.00% | 3595.72 | 13.89s | OK |
| 114 | `svd_linear_inject` | 52.00% | 3595.72 | 12.72s | OK |
| 115 | `svd_llm_v2_attention` | 52.00% | 1829.36 | 7.99s | OK |
| 116 | `svd_mamba` | 52.00% | 3857.65 | 55.03s | OK |
| 117 | `svd_projection_attention` | 52.00% | N/A | 6.67s | ERR: GPT2 error: name 'math' i |
| 118 | `svd_seq` | 95.00% | 5355.81 | 5.82s | OK |
| 119 | `template_attention` | 94.00% | 4203.82 | 2.96s | OK |
| 120 | `token_compressed_attention` | 92.00% | 3918.40 | 2.60s | OK |
| 121 | `tri_branch_hybrid` | 52.00% | N/A | 23.61s | ERR: GPT2 error: expected mat1 |
| 122 | `triton_cosformer` | 68.00% | 22903.90 | 3.12s | OK |
| 123 | `triton_svd_linear_attention` | 52.00% | 1213.86 | 20.72s | OK |
| 124 | `ttt_linear_attention` | 52.00% | 6511.01 | 4.52s | OK |
| 125 | `unified_rope` | 52.00% | 4270.03 | 2.73s | OK |
| 126 | `upi_stuffed_mamba` | 52.00% | 1327.97 | 5.29s | OK |
| 127 | `vig_attention` | 85.00% | N/A | 3.04s | ERR: GPT2 error: mat1 and mat2 |
| 128 | `vspec` | 52.00% | 1062.89 | 25.78s | OK |
| 129 | `wgqa_attention` | 52.00% | 927.07 | 2.37s | OK |
| 130 | `xattention` | 87.00% | 34.03 | 3.14s | OK |
| 131 | `zeros_attention` | 55.00% | 87116.36 | 2.96s | OK |
