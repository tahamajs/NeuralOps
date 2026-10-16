# Full Sweep Benchmark Report (Task 2)

## Executive Summary

- **Total Attention Modules Evaluated**: 130
- **Attention Modules Passed**: 130
- **Attention Modules Failed**: 0
- **Total Compression Engines Evaluated**: 72
- **Compression Engines Passed**: 70
- **Compression Engines Skipped (Deprecated)**: 2
- **Total Modules Evaluated Across Framework**: 202
- **Overall Pass Rate**: 99.0%

## Attention Modules Sweep (130 Modules)

| Module Key | Status | Forward (ms) | Backward (ms) | Total Step (ms) | Memory (MB) | Loss | Accuracy (%) | Grad Norm |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `adaptive_span_attention` | **PASS** | 46.94 | 66.28 | 113.22 | 2508.8 | 0.7473 | 49.1 | 6.2997 |
| `adaptive_sparse_attention` | **PASS** | 50.81 | 59.34 | 110.15 | 2817.0 | 0.7381 | 49.1 | 5.6956 |
| `adaptive_svd_attention` | **PASS** | 35.88 | 59.72 | 95.60 | 2880.7 | 0.7521 | 49.1 | 6.9856 |
| `adasplash_attention` | **PASS** | 61.85 | 88.59 | 150.44 | 3229.2 | 0.7384 | 49.1 | 5.0473 |
| `aegis_hybrid` | **PASS** | 56.07 | 102.64 | 158.71 | 4993.4 | 0.7590 | 49.1 | 7.1341 |
| `aegis_spectron_fp8` | **PASS** | 38.57 | 111.86 | 150.43 | 3299.1 | 0.7124 | 49.1 | 4.319 |
| `aegis_v1` | **PASS** | 75.22 | 157.67 | 232.89 | 4574.4 | 0.7665 | 49.1 | 7.4352 |
| `aegis_v2` | **PASS** | 73.65 | 157.65 | 231.30 | 4574.4 | 0.7661 | 49.1 | 7.4508 |
| `aegis_v3` | **PASS** | 73.69 | 157.61 | 231.30 | 4574.4 | 0.7714 | 49.1 | 7.738 |
| `aegis_v4` | **PASS** | 73.67 | 157.67 | 231.34 | 4574.4 | 0.7767 | 49.1 | 7.5817 |
| `aegis_v5` | **PASS** | 58.67 | 115.49 | 174.16 | 3116.0 | 0.7377 | 49.1 | 6.0889 |
| `aegis_v6` | **PASS** | 17.63 | 44.87 | 62.50 | 1755.0 | 0.7462 | 49.1 | 6.7996 |
| `alibi` | **PASS** | 31.49 | 52.30 | 83.79 | 2523.4 | 0.7691 | 49.1 | 7.4889 |
| `asvd_attention` | **PASS** | 33.39 | 51.41 | 84.80 | 2481.4 | 0.7531 | 49.1 | 6.6077 |
| `bam_umoe` | **PASS** | 40.09 | 65.95 | 106.04 | 3033.9 | 0.7086 | 49.1 | 2.9521 |
| `block_transformer_hiba` | **PASS** | 39.65 | 91.42 | 131.07 | 3188.5 | 0.7441 | 49.1 | 5.6094 |
| `chela_slay` | **PASS** | 10.43 | 31.47 | 41.90 | 1088.2 | 0.6891 | 49.1 | 1.9658 |
| `chela_slay_attention` | **PASS** | 54.06 | 154.48 | 208.54 | 4213.7 | 0.7032 | 49.1 | 2.5686 |
| `cluster_attention` | **PASS** | 869.98 | 694.05 | 1564.03 | 0.0 | 0.7174 | 49.1 | 4.1857 |
| `compressed_attention` | **PASS** | 19.48 | 30.37 | 49.85 | 1626.5 | 0.7635 | 49.1 | 7.3407 |
| `compressed_mha_attention` | **PASS** | 36.55 | 54.85 | 91.40 | 2392.7 | 0.7370 | 49.1 | 5.7445 |
| `cosformer` | **PASS** | 44.93 | 67.99 | 112.92 | 3987.7 | 0.7012 | 49.1 | 2.6229 |
| `cosformer_attention` | **PASS** | 45.32 | 67.88 | 113.20 | 3987.7 | 0.7030 | 49.1 | 2.813 |
| `cross_attention` | **PASS** | 19.28 | 30.97 | 50.25 | 1795.7 | 0.7696 | 49.1 | 7.5198 |
| `decay_linear_attention` | **PASS** | 24.09 | 53.25 | 77.34 | 2386.6 | 0.7665 | 49.1 | 7.5867 |
| `decay_linear_attn` | **PASS** | 23.24 | 53.19 | 76.43 | 2386.6 | 0.7656 | 49.1 | 7.4233 |
| `dynamic_conv_attention` | **PASS** | 15.49 | 77.33 | 92.82 | 1980.8 | 0.7337 | 49.1 | 5.3793 |
| `eat` | **PASS** | 39.01 | 79.04 | 118.05 | 3294.7 | 0.7009 | 49.1 | 2.2767 |
| `eat_attention` | **PASS** | 38.97 | 78.98 | 117.95 | 3294.7 | 0.6961 | 49.1 | 2.1355 |
| `expert_choice_moe` | **PASS** | 21.77 | 37.16 | 58.93 | 2251.4 | 0.6988 | 49.1 | 2.2093 |
| `factorised_matrix_state` | **PASS** | 22.29 | 41.77 | 64.06 | 2298.4 | 0.7666 | 49.1 | 7.4263 |
| `flash_attention3` | **PASS** | 19.43 | 30.85 | 50.28 | 1723.2 | 0.7019 | 49.1 | 2.8434 |
| `flash_attention3_fp8` | **PASS** | 19.50 | 30.82 | 50.32 | 1723.2 | 0.7035 | 49.1 | 2.8321 |
| `full` | **PASS** | 15.91 | 26.53 | 42.44 | 1682.9 | 0.7443 | 49.1 | 6.8824 |
| `full_attention` | **PASS** | 34.99 | 56.69 | 91.68 | 2457.4 | 0.7025 | 49.1 | 2.7802 |
| `fused_svd_kernel` | **PASS** | 17.84 | 28.62 | 46.46 | 1746.2 | 0.7397 | 49.1 | 6.0476 |
| `fusion_attn` | **PASS** | 40.84 | 70.94 | 111.78 | 3250.7 | 0.7458 | 49.1 | 6.3558 |
| `galerkin_attention` | **PASS** | 17.74 | 30.43 | 48.17 | 1731.2 | 0.7002 | 49.1 | 2.2194 |
| `gated_delta_net` | **PASS** | 105.54 | 233.42 | 338.96 | 4777.2 | 0.7444 | 49.1 | 6.1487 |
| `gated_deltanet_block` | **PASS** | 107.24 | 234.59 | 341.83 | 4777.2 | 0.7394 | 49.1 | 6.0398 |
| `gated_linear_attention` | **PASS** | 23.13 | 53.32 | 76.45 | 2386.6 | 0.7668 | 49.1 | 7.4569 |
| `gated_linear_attn` | **PASS** | 23.33 | 53.38 | 76.71 | 2386.6 | 0.7680 | 49.1 | 7.498 |
| `goldfinch_ahn` | **PASS** | 23.98 | 50.93 | 74.91 | 2607.2 | 0.7718 | 49.1 | 7.4734 |
| `gqa_attention` | **PASS** | 34.02 | 57.11 | 91.13 | 2553.5 | 0.7471 | 49.1 | 6.2421 |
| `granite_hybrid` | **PASS** | 18.18 | 104.64 | 122.82 | 2389.6 | 0.6902 | 49.1 | 1.7027 |
| `griffin_ssd` | **PASS** | 39.79 | 78.34 | 118.13 | 3478.2 | 0.7198 | 49.1 | 4.8015 |
| `halo_regla` | **PASS** | 42.87 | 67.46 | 110.33 | 2491.3 | 0.7052 | 49.1 | 2.305 |
| `hamming_attention` | **PASS** | 39.40 | 68.18 | 107.58 | 3225.4 | 0.6971 | 49.1 | 2.587 |
| `hgrn2_gated_deltanet` | **PASS** | 101.14 | 244.93 | 346.07 | 2997.9 | 0.7075 | 49.1 | 4.6233 |
| `hgrn2_gated_layer` | **PASS** | 34.55 | 56.55 | 91.10 | 3501.4 | 0.7633 | 49.1 | 7.6207 |
| `hgrn2_gated_outer` | **PASS** | 22.02 | 59.55 | 81.57 | 3020.4 | 0.7439 | 49.1 | 5.9959 |
| `hgrn2_outer` | **PASS** | 22.05 | 59.45 | 81.50 | 3020.4 | 0.7415 | 49.1 | 6.1338 |
| `hla_attention` | **PASS** | 23.69 | 42.53 | 66.22 | 2457.2 | 0.7669 | 49.1 | 7.6072 |
| `hybrid_attention` | **PASS** | 26.49 | 66.96 | 93.45 | 2604.5 | 0.7249 | 49.1 | 5.3217 |
| `hybrid_local_kernel` | **PASS** | 73.70 | 157.75 | 231.45 | 4574.7 | 0.7682 | 49.1 | 7.6254 |
| `hymba_transx` | **PASS** | 57.86 | 111.94 | 169.80 | 4213.0 | 0.7506 | 49.1 | 6.4329 |
| `inject_leka_v1` | **PASS** | 37.61 | 76.95 | 114.56 | 2777.8 | 0.6948 | 49.1 | 1.1341 |
| `inject_leka_v2` | **PASS** | 38.82 | 50.27 | 89.09 | 2774.6 | 0.7319 | 49.1 | 5.7043 |
| `inject_leka_v3` | **PASS** | 56.42 | 91.44 | 147.86 | 4639.9 | 0.7474 | 49.1 | 6.2808 |
| `inject_native_hybrid_attention` | **PASS** | 31.49 | 53.11 | 84.60 | 2762.0 | 0.6989 | 49.1 | 2.4079 |
| `jamba` | **PASS** | 35.52 | 80.05 | 115.57 | 2649.7 | 0.7425 | 49.1 | 6.3919 |
| `jamba_info` | **PASS** | 32.71 | 84.90 | 117.61 | 2903.7 | 0.7147 | 48.5 | 1.8265 |
| `kda_attention` | **PASS** | 30.25 | 53.97 | 84.22 | 3263.8 | 0.7676 | 49.1 | 7.5623 |
| `linformer_attention` | **PASS** | 21.73 | 37.36 | 59.09 | 1986.1 | 0.7627 | 49.1 | 7.5372 |
| `lola_attention` | **PASS** | 43.16 | 82.62 | 125.78 | 3551.4 | 0.7633 | 49.1 | 7.2548 |
| `lora_attention` | **PASS** | 33.35 | 56.79 | 90.14 | 2489.5 | 0.7483 | 49.1 | 6.259 |
| `lsh_attention` | **PASS** | 34.61 | 62.06 | 96.67 | 2753.0 | 0.7623 | 49.1 | 7.1249 |
| `mamba_block` | **PASS** | 41.00 | 168.33 | 209.33 | 4388.7 | 0.7655 | 49.1 | 7.5199 |
| `mamba_svd_flash` | **PASS** | 60.05 | 2514.43 | 2574.48 | 3380.6 | 0.7270 | 49.1 | 4.7144 |
| `minimal_nova` | **PASS** | 16.60 | 29.47 | 46.07 | 1722.7 | 0.7358 | 49.1 | 5.7324 |
| `minimal_svd` | **PASS** | 16.59 | 29.35 | 45.94 | 1722.7 | 0.7400 | 49.1 | 5.8522 |
| `mixattention_moska` | **PASS** | 38.50 | 68.23 | 106.73 | 3073.3 | 0.7386 | 49.1 | 6.2252 |
| `mla_attention` | **PASS** | 35.48 | 61.51 | 96.99 | 2968.7 | 0.7335 | 49.1 | 6.0522 |
| `moba_attention` | **PASS** | 35.94 | 65.31 | 101.25 | 3299.4 | 0.7432 | 49.1 | 6.2449 |
| `moe_deltakv_paged` | **PASS** | 52.37 | 68.86 | 121.23 | 2716.5 | 0.7394 | 49.2 | 4.8828 |
| `mqa` | **PASS** | 33.69 | 61.85 | 95.54 | 2817.0 | 0.7703 | 49.1 | 7.6515 |
| `naive_svd` | **PASS** | 15.35 | 26.17 | 41.52 | 1578.4 | 0.7330 | 49.1 | 5.5953 |
| `nasc_attention` | **PASS** | 23.09 | 50.61 | 73.70 | 2028.7 | 0.6908 | 49.1 | 2.5265 |
| `nexus` | **PASS** | 60.58 | 100.10 | 160.68 | 4244.7 | 0.7707 | 49.1 | 7.6037 |
| `nsa_attention` | **PASS** | 47.14 | 54.63 | 101.77 | 2668.2 | 0.7628 | 49.1 | 7.5918 |
| `nystrom_attention` | **PASS** | 30.88 | 53.89 | 84.77 | 2410.0 | 0.7662 | 49.1 | 7.5501 |
| `optimized_hybrid_attention` | **PASS** | 35.49 | 67.28 | 102.77 | 3195.8 | 0.7676 | 49.1 | 7.5022 |
| `orion` | **PASS** | 51.29 | 101.34 | 152.63 | 3632.8 | 0.7632 | 49.1 | 7.2375 |
| `padre_attention` | **PASS** | 49.75 | 88.35 | 138.10 | 3383.2 | 0.7252 | 50.9 | 4.3109 |
| `performer_attention` | **PASS** | 21.01 | 39.82 | 60.83 | 1981.2 | 0.7753 | 49.1 | 7.5495 |
| `phantom` | **PASS** | 64.66 | 108.01 | 172.67 | 4394.3 | 0.7369 | 49.1 | 5.3802 |
| `primal_attention` | **PASS** | 41.15 | 55.10 | 96.25 | 2674.9 | 0.7429 | 49.1 | 6.2767 |
| `progressive_token_pruning` | **PASS** | 31.11 | 52.46 | 83.57 | 2465.2 | 0.7228 | 48.9 | 3.6779 |
| `projection_factorized_attn` | **PASS** | 29.22 | 47.33 | 76.55 | 2872.7 | 0.7655 | 49.1 | 7.9136 |
| `qcqa_attention` | **PASS** | 19.64 | 34.41 | 54.05 | 1955.2 | 0.7443 | 49.1 | 6.3869 |
| `rala_attention` | **PASS** | 55.29 | 377.92 | 433.21 | 4421.7 | 0.7670 | 49.1 | 7.5783 |
| `rala_compressor` | **PASS** | 51.90 | 151.13 | 203.03 | 2233.4 | 0.6918 | 49.1 | 1.3774 |
| `reap_moe` | **PASS** | 23.96 | 40.91 | 64.87 | 2650.4 | 0.6945 | 50.9 | 0.3705 |
| `regla_attention` | **PASS** | 63.20 | 201.26 | 264.46 | 4441.6 | 0.7662 | 49.1 | 7.4908 |
| `residual_svd_attention` | **PASS** | 38.15 | 67.69 | 105.84 | 3000.7 | 0.7650 | 49.1 | 7.8061 |
| `routing_attention` | **PASS** | 37.67 | 52.49 | 90.16 | 2191.0 | 0.7654 | 49.1 | 7.4797 |
| `rpc_attention` | **PASS** | 38.89 | 69.12 | 108.01 | 3009.2 | 0.7593 | 49.1 | 6.5434 |
| `salt_attention` | **PASS** | 34.95 | 55.17 | 90.12 | 3114.4 | 0.7119 | 49.1 | 3.9746 |
| `samba_m3` | **PASS** | 29.59 | 233.24 | 262.83 | 2605.9 | 0.7021 | 49.1 | 1.5283 |
| `samba_moe` | **PASS** | 20.86 | 56.98 | 77.84 | 1764.6 | 0.7085 | 49.1 | 1.9745 |
| `selective_ssm` | **PASS** | 51.85 | 169.22 | 221.07 | 3894.2 | 0.7470 | 49.1 | 7.5047 |
| `simple_ssm` | **PASS** | 51.31 | 112.97 | 164.28 | 3894.2 | 0.7392 | 49.1 | 7.4864 |
| `sink_attention` | **PASS** | 19.63 | 34.00 | 53.63 | 1793.2 | 0.7003 | 49.1 | 2.0977 |
| `soa_attn` | **PASS** | 17.63 | 30.03 | 47.66 | 1786.7 | 0.7502 | 49.1 | 5.1794 |
| `spectraformer_attention` | **PASS** | 26.76 | 45.09 | 71.85 | 1987.5 | 0.7057 | 50.7 | 0.9877 |
| `spectron` | **PASS** | 36.08 | 62.30 | 98.38 | 3001.2 | 0.7111 | 49.1 | 3.5351 |
| `spectron_fp8` | **PASS** | 38.52 | 111.91 | 150.43 | 3299.4 | 0.7146 | 49.1 | 4.0421 |
| `split_head_hybrid` | **PASS** | 58.72 | 115.64 | 174.36 | 3116.1 | 0.7368 | 49.1 | 6.2206 |
| `svd_compressed` | **PASS** | 34.68 | 53.61 | 88.29 | 3000.7 | 0.7660 | 49.1 | 7.4207 |
| `svd_linear` | **PASS** | 15.93 | 26.43 | 42.36 | 1697.8 | 0.7089 | 49.1 | 3.3569 |
| `svd_linear_attention` | **PASS** | 28.86 | 47.33 | 76.19 | 2872.7 | 0.7688 | 49.1 | 7.9045 |
| `svd_linear_attn` | **PASS** | 28.95 | 47.35 | 76.30 | 2872.7 | 0.7695 | 49.1 | 8.0678 |
| `svd_linear_decay` | **PASS** | 28.92 | 47.34 | 76.26 | 2872.7 | 0.7729 | 49.1 | 7.9708 |
| `svd_linear_inject` | **PASS** | 28.91 | 47.31 | 76.22 | 2872.7 | 0.7657 | 49.1 | 7.9842 |
| `svd_llm_v2_attention` | **PASS** | 38.22 | 59.79 | 98.01 | 3064.4 | 0.7023 | 50.8 | 1.4469 |
| `svd_mamba` | **PASS** | 32.16 | 158.83 | 190.99 | 4396.3 | 0.6861 | 49.1 | 1.7434 |
| `svd_projection_attention` | **PASS** | 23.57 | 30.03 | 53.60 | 1730.7 | 0.7614 | 49.1 | 7.2021 |
| `svd_seq` | **PASS** | 31.13 | 31.26 | 62.39 | 1849.7 | 0.7670 | 49.1 | 7.5305 |
| `template_attention` | **PASS** | 39.58 | 72.02 | 111.60 | 3257.5 | 0.7454 | 49.1 | 6.1198 |
| `token_compressed_attention` | **PASS** | 30.12 | 49.86 | 79.98 | 2442.7 | 0.7529 | 49.1 | 6.2425 |
| `tri_branch_hybrid` | **PASS** | 56.13 | 102.71 | 158.84 | 4993.9 | 0.7580 | 49.1 | 7.1003 |
| `triton_cosformer` | **PASS** | 16.42 | 49.08 | 65.50 | 1609.2 | 0.7663 | 49.1 | 7.9473 |
| `triton_svd_linear_attention` | **PASS** | 15.25 | 40.79 | 56.04 | 1576.8 | 0.7640 | 49.1 | 8.0092 |
| `unified_rope` | **PASS** | 21.32 | 36.23 | 57.55 | 1893.2 | 0.7472 | 49.1 | 6.2891 |
| `upi_stuffed_mamba` | **PASS** | 17.88 | 46.47 | 64.35 | 2862.8 | 0.7080 | 49.1 | 4.3914 |
| `vig_attention` | **PASS** | 26.55 | 60.16 | 86.71 | 2505.3 | 0.7200 | 49.1 | 4.3226 |
| `vspec` | **PASS** | 41.67 | 68.22 | 109.89 | 2773.8 | 0.7684 | 49.1 | 7.2626 |
| `wgqa_attention` | **PASS** | 17.49 | 31.07 | 48.56 | 1763.0 | 0.7687 | 49.1 | 7.626 |
| `xattention` | **PASS** | 35.55 | 56.27 | 91.82 | 2689.9 | 0.7698 | 49.1 | 7.5359 |
| `zeros_attention` | **PASS** | 19.28 | 48.48 | 67.76 | 2225.9 | 0.7417 | 49.1 | 5.8135 |

## Compression Engines Sweep (72 Modules)

| Module Key | Status | Forward (ms) | Backward (ms) | Total Step (ms) | Memory (MB) | Loss | Compression Ratio | Grad Norm | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `a3_compressor` | **PASS** | 21.20 | 34.14 | 55.34 | 1788.8 | 0.6965 | 1.002 | 1.9289 | Compressor did not reduce parameters; all layers likely fell back. |
| `aa_svd_compressor` | **PASS** | 16.52 | 26.86 | 43.38 | 1712.2 | 0.6924 | 1.005 | 1.5984 | Compressor did not reduce parameters; all layers likely fell back. |
| `aatc_xkv_compressor` | **FAIL** | 0.00 | 0.00 | 0.00 | 0.0 | 0.0000 | 1.000 | 0.0 | Module 'aatc_xkv_compressor' is in DEPRECATED_MODULES. |
| `adaptive_ranks` | **PASS** | 16.13 | 25.66 | 41.79 | 1522.5 | 0.6933 | 1.000 | 0.729 | Compressor did not reduce parameters; all layers likely fell back. |
| `adasvd_compressor` | **PASS** | 18.45 | 30.22 | 48.67 | 1690.3 | 0.6986 | 1.002 | 1.4553 | Compressor did not reduce parameters; all layers likely fell back. |
| `admm_compressor` | **PASS** | 15.87 | 26.55 | 42.42 | 1682.5 | 0.6951 | 1.000 | 1.7942 | Compressor did not reduce parameters; all layers likely fell back. |
| `admm_pruner` | **PASS** | 15.88 | 26.49 | 42.37 | 1682.5 | 0.6924 | 1.000 | 2.0694 | Compressor did not reduce parameters; all layers likely fell back. |
| `armor_pruner` | **PASS** | 22.17 | 32.42 | 54.59 | 1773.2 | 0.6884 | 0.981 | 0.673 |  |
| `ast_pruner` | **PASS** | 16.20 | 27.51 | 43.71 | 1684.5 | 0.6918 | 1.000 | 1.2341 | Compressor did not reduce parameters; all layers likely fell back. |
| `astro_quantizer` | **PASS** | 16.19 | 26.82 | 43.01 | 1682.5 | 0.6910 | 1.000 | 1.3796 | Compressor did not reduce parameters; all layers likely fell back. |
| `bonsai_pruner` | **PASS** | 16.22 | 27.14 | 43.36 | 1684.5 | 0.6996 | 1.000 | 2.0991 | Compressor did not reduce parameters; all layers likely fell back. |
| `cachenotes_compressor` | **PASS** | 15.85 | 25.52 | 41.37 | 1522.5 | 0.6930 | 1.000 | 0.9211 | Compressor did not reduce parameters; all layers likely fell back. |
| `come_compressor` | **PASS** | 20.89 | 31.90 | 52.79 | 1755.0 | 0.6934 | 0.991 | 1.0865 |  |
| `compot_compressor` | **PASS** | 19.53 | 31.22 | 50.75 | 1748.3 | 0.7032 | 1.013 | 1.6963 | Compressor did not reduce parameters; all layers likely fell back. |
| `coopq_quantizer` | **PASS** | 17.28 | 26.64 | 43.92 | 1685.9 | 0.6941 | 1.000 | 2.145 | Compressor did not reduce parameters; all layers likely fell back. |
| `cr_compressor` | **PASS** | 15.99 | 26.56 | 42.55 | 1712.5 | 0.6957 | 1.040 | 0.7101 | Compressor did not reduce parameters; all layers likely fell back. |
| `cur_compressor` | **PASS** | 16.09 | 26.81 | 42.90 | 1744.5 | 0.6919 | 1.038 | 0.5862 | Compressor did not reduce parameters; all layers likely fell back. |
| `cur_moe_compressor` | **PASS** | 15.95 | 26.60 | 42.55 | 1713.1 | 0.6914 | 1.026 | 0.7518 | Compressor did not reduce parameters; all layers likely fell back. |
| `delta_kv` | **PASS** | 15.80 | 25.55 | 41.35 | 1522.5 | 0.6955 | 1.000 | 0.9307 |  |
| `direct_svd` | **PASS** | 19.71 | 30.89 | 50.60 | 1759.4 | 0.6967 | 1.015 | 1.2416 | Compressor did not reduce parameters; all layers likely fell back. |
| `dobi_svd` | **PASS** | 18.37 | 30.03 | 48.40 | 1665.2 | 0.6957 | 1.026 | 0.5137 | Compressor did not reduce parameters; all layers likely fell back. |
| `drone_compressor` | **PASS** | 16.37 | 26.66 | 43.03 | 1697.8 | 0.6964 | 1.012 | 1.4071 | Compressor did not reduce parameters; all layers likely fell back. |
| `dynamic_rank` | **PASS** | 15.94 | 26.70 | 42.64 | 1712.5 | 0.6963 | 1.039 | 1.2003 | Compressor did not reduce parameters; all layers likely fell back. |
| `enlighten_compressor` | **PASS** | 15.87 | 26.54 | 42.41 | 1682.5 | 0.6987 | 1.000 | 2.051 | Compressor did not reduce parameters; all layers likely fell back. |
| `entropy_budget_allocation` | **PASS** | 17.55 | 27.23 | 44.78 | 1720.5 | 0.7004 | 1.040 | 0.9208 | Compressor did not reduce parameters; all layers likely fell back. |
| `erc_svd` | **PASS** | 19.24 | 30.13 | 49.37 | 1693.4 | 0.6915 | 1.056 | 0.4491 | Compressor did not reduce parameters; all layers likely fell back. |
| `faq_quantizer` | **PASS** | 21.59 | 31.71 | 53.30 | 1723.7 | 0.6976 | 0.991 | 0.5739 |  |
| `flap` | **PASS** | 16.13 | 26.94 | 43.07 | 1682.5 | 0.6942 | 1.000 | 0.9222 | Compressor did not reduce parameters; all layers likely fell back. |
| `flat_llm` | **PASS** | 19.79 | 29.82 | 49.61 | 1519.2 | 0.6955 | 1.017 | 0.4313 | Compressor did not reduce parameters; all layers likely fell back. |
| `frandi_compressor` | **PASS** | 18.38 | 30.35 | 48.73 | 1712.5 | 0.6992 | 1.040 | 1.1808 | Compressor did not reduce parameters; all layers likely fell back. |
| `fwsvd_compressor` | **PASS** | 16.38 | 26.70 | 43.08 | 1697.8 | 0.6945 | 1.012 | 1.4364 | Compressor did not reduce parameters; all layers likely fell back. |
| `gfws` | **PASS** | 17.68 | 26.94 | 44.62 | 1522.5 | 0.6972 | 1.000 | 0.576 | Compressor did not reduce parameters; all layers likely fell back. |
| `grecodec_compressor` | **PASS** | 18.17 | 30.27 | 48.44 | 1712.5 | 0.6946 | 1.040 | 1.2153 | Compressor did not reduce parameters; all layers likely fell back. |
| `hashed_net_compressor` | **PASS** | 23.50 | 36.55 | 60.05 | 1717.1 | 0.6976 | 1.050 | 1.0161 | Compressor did not reduce parameters; all layers likely fell back. |
| `huffman_coder` | **PASS** | 15.92 | 26.59 | 42.51 | 1682.5 | 0.6932 | 1.000 | 1.5839 | Compressor did not reduce parameters; all layers likely fell back. |
| `hybrid_svd_layer` | **PASS** | 17.33 | 27.27 | 44.60 | 1781.5 | 0.7008 | 1.000 | 1.6498 | Compressor did not reduce parameters; all layers likely fell back. |
| `hybridkv_compressor` | **FAIL** | 0.00 | 0.00 | 0.00 | 0.0 | 0.0000 | 1.000 | 0.0 | Module 'hybridkv_compressor' is in DEPRECATED_MODULES. |
| `kmeans_quantize` | **PASS** | 15.87 | 30.49 | 46.36 | 1735.3 | 0.6947 | 1.060 | 1.3904 | Compressor did not reduce parameters; all layers likely fell back. |
| `kronecker_compressor` | **PASS** | 20.92 | 33.18 | 54.10 | 1945.0 | 0.6976 | 1.029 | 1.1991 | Compressor did not reduce parameters; all layers likely fell back. |
| `kronecker_tt_lora` | **PASS** | 28.05 | 37.11 | 65.16 | 1928.0 | 0.6999 | 0.985 | 0.9214 |  |
| `kvtc_compressor` | **PASS** | 16.57 | 25.57 | 42.14 | 1521.6 | 0.6957 | 1.060 | 1.1081 | Compressor did not reduce parameters; all layers likely fell back. |
| `liftr` | **PASS** | 5.48 | 10.12 | 15.60 | 746.3 | 0.6941 | 1.000 | 2.1202 | Compressor did not reduce parameters; all layers likely fell back. |
| `lsc_compressor` | **PASS** | 19.29 | 28.38 | 47.67 | 1722.5 | 0.6939 | 1.000 | 0.4144 | Compressor did not reduce parameters; all layers likely fell back. |
| `mamba_shedder` | **PASS** | 18.30 | 27.22 | 45.52 | 1520.2 | 0.6946 | 1.012 | 0.7234 | Compressor did not reduce parameters; all layers likely fell back. |
| `mgie_budget` | **PASS** | 19.33 | 29.95 | 49.28 | 1720.5 | 0.6959 | 1.040 | 1.1838 | Compressor did not reduce parameters; all layers likely fell back. |
| `mlorq_compressor` | **PASS** | 22.97 | 33.63 | 56.60 | 1714.3 | 0.6973 | 1.060 | 0.7074 | Compressor did not reduce parameters; all layers likely fell back. |
| `modegpt_compressor` | **PASS** | 16.04 | 25.40 | 41.44 | 1515.1 | 0.6989 | 1.040 | 0.8029 | Compressor did not reduce parameters; all layers likely fell back. |
| `nmf_compressor` | **PASS** | 20.71 | 27.96 | 48.67 | 1717.2 | 0.6937 | 1.040 | 0.506 | Compressor did not reduce parameters; all layers likely fell back. |
| `oats_compressor` | **PASS** | 15.80 | 25.37 | 41.17 | 1522.5 | 0.6964 | 1.000 | 0.4331 | Compressor did not reduce parameters; all layers likely fell back. |
| `optimal_subspace_projection` | **PASS** | 18.20 | 29.00 | 47.20 | 1522.5 | 0.6934 | 1.000 | 0.6919 | Compressor did not reduce parameters; all layers likely fell back. |
| `proxsparse` | **PASS** | 18.62 | 30.60 | 49.22 | 1682.5 | 0.6958 | 1.000 | 1.1794 | Compressor did not reduce parameters; all layers likely fell back. |
| `residual_svd_compressor` | **PASS** | 24.39 | 35.69 | 60.08 | 1922.7 | 0.7024 | 1.034 | 1.24 | Compressor did not reduce parameters; all layers likely fell back. |
| `roast_hashed_linear` | **PASS** | 15.89 | 26.78 | 42.67 | 1682.8 | 0.6957 | 1.059 | 0.6055 | Compressor did not reduce parameters; all layers likely fell back. |
| `rotational_slicegpt_int4` | **PASS** | 21.67 | 31.14 | 52.81 | 1726.8 | 0.6915 | 1.060 | 0.5656 | Compressor did not reduce parameters; all layers likely fell back. |
| `rpca_compressor` | **PASS** | 16.98 | 26.58 | 43.56 | 1720.5 | 0.6969 | 1.040 | 1.2417 | Compressor did not reduce parameters; all layers likely fell back. |
| `saes_svd_v2` | **PASS** | 17.00 | 26.62 | 43.62 | 1720.5 | 0.7000 | 1.040 | 1.1466 | Compressor did not reduce parameters; all layers likely fell back. |
| `salr` | **PASS** | 16.01 | 26.58 | 42.59 | 1712.5 | 0.6979 | 1.040 | 1.0174 | Compressor did not reduce parameters; all layers likely fell back. |
| `sdrm_compressor` | **PASS** | 17.07 | 27.06 | 44.13 | 1716.4 | 0.6988 | 1.010 | 2.0785 | Compressor did not reduce parameters; all layers likely fell back. |
| `slicegpt` | **PASS** | 19.96 | 30.32 | 50.28 | 1719.8 | 0.6994 | 1.017 | 1.0483 | Compressor did not reduce parameters; all layers likely fell back. |
| `slicegpt_pca` | **PASS** | 19.58 | 30.59 | 50.17 | 1727.3 | 0.6977 | 1.003 | 1.0472 | Compressor did not reduce parameters; all layers likely fell back. |
| `slim_compressor` | **PASS** | 15.86 | 26.58 | 42.44 | 1682.5 | 0.6923 | 1.000 | 1.0326 | Compressor did not reduce parameters; all layers likely fell back. |
| `soka_compressor` | **PASS** | 19.13 | 30.40 | 49.53 | 1928.5 | 0.6996 | 1.040 | 1.2023 | Compressor did not reduce parameters; all layers likely fell back. |
| `sparsegpt` | **PASS** | 15.98 | 26.66 | 42.64 | 1682.5 | 0.6912 | 1.000 | 1.1782 | Compressor did not reduce parameters; all layers likely fell back. |
| `subspace_projection` | **PASS** | 18.29 | 30.55 | 48.84 | 1712.5 | 0.7011 | 1.040 | 1.2734 | Compressor did not reduce parameters; all layers likely fell back. |
| `svd_linear` | **PASS** | 15.94 | 26.45 | 42.39 | 1697.3 | 0.6958 | 1.023 | 2.1042 | Compressor did not reduce parameters; all layers likely fell back. |
| `svd_llm_compressor` | **PASS** | 22.21 | 35.09 | 57.30 | 1866.0 | 0.7009 | 1.008 | 1.5959 | Compressor did not reduce parameters; all layers likely fell back. |
| `swift_svd` | **PASS** | 19.48 | 30.64 | 50.12 | 1743.3 | 0.6959 | 1.024 | 0.8826 | Compressor did not reduce parameters; all layers likely fell back. |
| `swift_svd_act` | **PASS** | 19.36 | 30.15 | 49.51 | 1720.5 | 0.6931 | 1.040 | 0.7968 | Compressor did not reduce parameters; all layers likely fell back. |
| `tomoe_compressor` | **PASS** | 30.03 | 45.78 | 75.81 | 2544.4 | 0.6952 | 0.999 | 0.3961 | Compressor did not reduce parameters; all layers likely fell back. |
| `tt_compressor` | **PASS** | 16.90 | 25.37 | 42.27 | 1523.2 | 0.6949 | 1.039 | 0.5033 | Compressor did not reduce parameters; all layers likely fell back. |
| `uniql_compressor` | **PASS** | 19.51 | 28.09 | 47.60 | 1721.3 | 0.7012 | 1.060 | 0.8534 | Compressor did not reduce parameters; all layers likely fell back. |
| `zerokv_neo_compressor` | **PASS** | 15.87 | 26.58 | 42.45 | 1682.5 | 0.6940 | 1.000 | 1.2414 | Compressor did not reduce parameters; all layers likely fell back. |