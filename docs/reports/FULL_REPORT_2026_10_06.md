# Full Merged Benchmark & Forensic Audit Report

**Generated:** 2026-10-06 20:29:00 | **Branch:** ablation (merged origin/main)

This comprehensive report merges historical (old) and newly pulled/completed (new) experimental benchmarks across knowledge distillation (KD) perplexity recovery on WikiText-2 (GPT-2), downstream classification fine-tuning across GLUE / SST-2 / MRPC, architectural compatibility screening over 236 modules, and targeted forensic ablations.

## Section A: Knowledge Distillation (KD) PPL Recovery Sweep (WikiText-2 on GPT-2)

Teacher Baseline PPL: **66.07** (Evaluated on WikiText-2 test set, 15,000 tokens, seq_len=128, stride=128).

- **BEAT**: Post-KD PPL < 66.07 (Student surpasses teacher accuracy)
- **MATCH**: Post-KD PPL within ±2 of 66.07 (64.07 - 68.07)
- **FAIL**: Post-KD PPL > 68.07 or divergence

| Module | Pre-KD PPL | Post-KD Stage 1 | Post-KD Stage 2 | KD Gain | Self Gain | Total Gain | Latency (s/step) | Peak VRAM (MB) | Status | Source |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `expert_choice_moe` | 66.26 | 58.23 | 48.41 | 8.03 | 9.82 | 17.85 | 0.649 | 4233.9 | **BEAT** | new |
| `fusion_attn` | 66.07 | 58.02 | 48.47 | 8.05 | 9.55 | 17.60 | 0.413 | 4044.5 | **BEAT** | new |
| `flash3_fp8` | 66.07 | 58.14 | 48.47 | 7.93 | 9.67 | 17.60 | 0.365 | 4043.8 | **BEAT** | new |
| `block_transformer_hiba` | 66.07 | 58.17 | 48.54 | 7.90 | 9.63 | 17.53 | 0.404 | 4047.6 | **BEAT** | new |
| `fasa` | 66.07 | 58.08 | 48.55 | 7.99 | 9.53 | 17.52 | 0.371 | 4043.2 | **BEAT** | new |
| `galerkin` | 66.07 | 57.99 | 48.57 | 8.08 | 9.42 | 17.50 | 0.376 | 4047.6 | **BEAT** | new |
| `flash3` | 66.07 | 58.50 | 48.64 | 7.57 | 9.86 | 17.43 | 0.364 | 4047.6 | **BEAT** | new |
| `huffman` | 66.07 | 59.24 | 48.93 | 6.83 | 10.31 | 17.14 | 0.216 | 4021.4 | **BEAT** | new |
| `direct_svd` | 66.07 | 59.40 | 49.14 | 6.67 | 10.26 | 16.93 | 0.216 | 4023.8 | **BEAT** | new |
| `granite_hybrid` | 66.07 | 59.56 | 49.18 | 6.51 | 10.38 | 16.89 | 0.216 | 4022.0 | **BEAT** | new |
| `hashed_net` | 66.07 | 59.30 | 49.23 | 6.77 | 10.07 | 16.84 | 0.217 | 4022.9 | **BEAT** | new |
| `eat` | 66.07 | 59.61 | 49.29 | 6.46 | 10.32 | 16.78 | 0.216 | 4022.0 | **BEAT** | new |
| `faq` | 67.09 | 59.76 | 49.48 | 7.33 | 10.28 | 17.61 | 0.589 | 4416.2 | **BEAT** | new |
| `asvd` | 180.62 | 61.31 | 49.57 | 119.31 | 11.74 | 131.05 | 1.670 | 4137.4 | **BEAT** | new |
| `hybridkv` | 66.07 | 61.13 | 50.28 | 4.94 | 10.85 | 15.79 | 0.215 | 3034.7 | **BEAT** | new |
| `adasplash_inject` | 69.15 | 59.53 | 52.17 | 9.62 | 7.36 | 16.98 | 1.088 | 4347.5 | **BEAT** | new |
| `cluster` | 66.06 | 59.74 | 52.18 | 6.32 | 7.56 | 13.88 | 0.418 | 3724.7 | **BEAT** | new |
| `adaptive_span` | 65.97 | 59.84 | 52.20 | 6.13 | 7.64 | 13.77 | 0.460 | 3751.8 | **BEAT** | new |
| `adasplash` | 69.15 | 59.55 | 52.23 | 9.60 | 7.32 | 16.92 | 1.083 | 4348.6 | **BEAT** | new |
| `cluster_inject` | 66.06 | 59.80 | 52.24 | 6.26 | 7.56 | 13.82 | 0.414 | 3724.7 | **BEAT** | new |
| `delta_kv` | 143.32 | 63.67 | 52.47 | 79.65 | 11.20 | 90.85 | 0.216 | 4023.4 | **BEAT** | new |
| `aegis_spectron_fp8_fn` | 66.07 | 61.54 | 53.41 | 4.53 | 8.13 | 12.66 | 0.197 | 3519.9 | **BEAT** | new |
| `aegis_hybrid_fn` | 66.07 | 61.56 | 53.41 | 4.51 | 8.15 | 12.66 | 0.206 | 3519.9 | **BEAT** | new |
| `aegis_v4_inject` | 66.07 | 61.53 | 53.45 | 4.54 | 8.08 | 12.62 | 0.196 | 3519.9 | **BEAT** | new |
| `coopq` | 140.73 | 66.09 | 54.57 | 74.64 | 11.52 | 86.16 | 0.492 | 4014.9 | **BEAT** | new |
| `aasvd` | 157.68 | 71.23 | 54.70 | 86.45 | 16.53 | 102.98 | 0.225 | 3448.4 | **BEAT** | new |
| `astro` | 69.73 | 63.05 | 54.83 | 6.68 | 8.22 | 14.90 | 0.196 | 3526.0 | **BEAT** | new |
| `cachenotes` | 72.90 | 63.75 | 55.01 | 9.15 | 8.74 | 17.89 | 0.196 | 3524.8 | **BEAT** | new |
| `a3` | 279.30 | 71.62 | 58.79 | 207.68 | 12.83 | 220.51 | 0.215 | 3428.3 | **BEAT** | new |
| `admm` | 99.53 | 70.86 | 61.60 | 28.67 | 9.26 | 37.93 | 0.196 | 3523.9 | **BEAT** | new |
| `compressed_mha` | 869.18 | 84.75 | 62.98 | 784.43 | 21.77 | 806.20 | 0.412 | 3658.8 | **BEAT** | new |
| `compressed_attention` | 870.06 | 85.06 | 63.40 | 785.00 | 21.66 | 806.66 | 0.340 | 3657.4 | **BEAT** | new |
| `adaptive_span_inject` | 65.97 | - | 64.11 | - | - | - | 0.655 | 2597.8 | **BEAT** | new |
| `adaptive_sparse_inject` | 65.93 | 65.94 | 65.94 | -0.01 | 0.00 | -0.01 | 2.725 | 2438.5 | **BEAT** | new |
| `adaptive_sparse` | 65.93 | 65.94 | 65.94 | -0.01 | 0.00 | -0.01 | 0.566 | 3751.4 | **BEAT** | new |
| `come` | 148.64 | 86.44 | 72.51 | 62.20 | 13.93 | 76.13 | 0.275 | 3444.2 | FAIL | new |
| `enlighten` | 405.57 | 91.04 | 72.54 | 314.53 | 18.50 | 333.03 | 0.217 | 4020.0 | FAIL | new |
| `armor` | 153.34 | 97.38 | 83.65 | 55.96 | 13.73 | 69.69 | 0.314 | 3903.0 | FAIL | new |
| `bam_umoe` | 910.76 | 129.25 | 95.80 | 781.51 | 33.45 | 814.96 | 0.549 | 3583.2 | FAIL | new |
| `admm_pruner` | 592.62 | 118.55 | 96.86 | 474.07 | 21.69 | 495.76 | 0.201 | 3524.3 | FAIL | new |
| `aegis_v1` | 1361.57 | 126.07 | 99.82 | 1235.50 | 26.25 | 1261.75 | 0.788 | 5199.2 | FAIL | new |
| `aegis_v5` | 1174.96 | 160.37 | 121.53 | 1014.59 | 38.84 | 1053.43 | 0.837 | 4518.4 | FAIL | new |
| `aegis_spectron_fp8` | 2004.71 | 210.50 | 146.25 | 1794.21 | 64.25 | 1858.46 | 2.791 | 4720.8 | FAIL | new |
| `flap` | 1208.38 | 236.81 | 160.38 | 971.57 | 76.43 | 1048.00 | 0.216 | 4019.0 | FAIL | new |
| `hamming` | 1754.51 | 213.04 | 161.24 | 1541.47 | 51.80 | 1593.27 | 0.468 | 4045.0 | FAIL | new |
| `adasvd` | 1233.94 | 212.20 | 170.32 | 1021.74 | 41.88 | 1063.62 | 0.220 | 3462.1 | FAIL | new |
| `factorised_matrix_state` | 16332.04 | 294.96 | 215.59 | 16037.08 | 79.37 | 16116.45 | 0.425 | 4340.2 | FAIL | new |
| `decay_linear_attention_inject` | 9583.01 | 298.85 | 219.22 | 9284.16 | 79.63 | 9363.79 | 0.447 | 4344.7 | FAIL | new |
| `decay_linear_attn` | 9583.01 | 297.85 | 219.39 | 9285.16 | 78.46 | 9363.62 | 0.452 | 4349.3 | FAIL | new |
| `compressed_attention_layer_energy` | 1844.20 | 362.17 | 231.40 | 1482.03 | 130.77 | 1612.80 | 0.338 | 3364.4 | FAIL | new |
| `fused_svd_kernel` | 1518.69 | 367.11 | 235.65 | 1151.58 | 131.46 | 1283.04 | 0.322 | 3375.8 | FAIL | new |
| `adaptive_svd` | 1518.20 | 410.73 | 316.35 | 1107.47 | 94.38 | 1201.85 | 0.473 | 3413.7 | FAIL | new |
| `bonsai` | 4140.24 | 519.29 | 385.11 | 3620.95 | 134.18 | 3755.13 | 0.197 | 3520.9 | FAIL | new |
| `cosformer` | 53449.56 | 601.68 | 409.05 | 52847.88 | 192.63 | 53040.51 | 1.032 | 4045.7 | FAIL | new |
| `gated_delta_net` | 2351.64 | 725.28 | 454.88 | 1626.36 | 270.40 | 1896.76 | 0.565 | 4653.9 | FAIL | new |
| `flat_llm` | 6614.54 | 748.61 | 494.56 | 5865.93 | 254.05 | 6119.98 | 0.242 | 3351.1 | FAIL | new |
| `gqa` | 2974.42 | 797.45 | 501.51 | 2176.97 | 295.94 | 2472.91 | 0.460 | 3758.0 | FAIL | new |
| `ast` | 8513.91 | 945.85 | 642.46 | 7568.06 | 303.39 | 7871.45 | 0.197 | 3518.8 | FAIL | new |
| `hgrn2_layer` | 45300.34 | 1113.51 | 644.74 | 44186.83 | 468.77 | 44655.60 | 0.400 | 4217.1 | FAIL | new |
| `gated_deltanet_block` | 1746.04 | 1125.07 | 667.93 | 620.97 | 457.14 | 1078.11 | 0.576 | 4550.4 | FAIL | new |
| `aegis_v6` | 868.62 | 868.62 | 868.62 | 0.00 | 0.00 | 0.00 | 1.388 | 6712.2 | FAIL | new |
| `hybrid_svd` | 4711.58 | 1608.16 | 954.97 | 3103.42 | 653.19 | 3756.61 | 0.266 | 2276.9 | FAIL | new |
| `cur_moe` | 5215.99 | 1790.59 | 1137.45 | 3425.40 | 653.14 | 4078.54 | 0.259 | 2833.8 | FAIL | new |
| `gfws` | 4118.21 | 1784.69 | 1250.90 | 2333.52 | 533.79 | 2867.31 | 0.219 | 4021.4 | FAIL | new |
| `gated_linear_attention_inject` | 42781.03 | 2106.45 | 1318.52 | 40674.58 | 787.93 | 41462.51 | 0.499 | 4555.3 | FAIL | new |
| `gated_linear_attn` | 42781.03 | 2112.24 | 1334.65 | 40668.79 | 777.59 | 41446.38 | 0.506 | 4557.6 | FAIL | new |
| `hgrn2_outer` | 18524.53 | 2146.16 | 1361.42 | 16378.37 | 784.74 | 17163.11 | 0.781 | 4707.1 | FAIL | new |
| `hla` | 3757.90 | 2759.35 | 1616.57 | 998.55 | 1142.78 | 2141.33 | 0.592 | 4046.9 | FAIL | new |
| `entropy_rank_allocator` | 5349.86 | 2593.68 | 1651.64 | 2756.18 | 942.04 | 3698.22 | 0.260 | 2070.0 | FAIL | new |
| `grecodec` | 4570.47 | 2558.28 | 1653.20 | 2012.19 | 905.08 | 2917.27 | 0.261 | 2070.3 | FAIL | new |
| `dynamic_rank` | 5349.86 | 2604.06 | 1653.94 | 2745.80 | 950.12 | 3695.92 | 0.261 | 2070.0 | FAIL | new |
| `entropy_budget` | 4571.52 | 2546.11 | 1661.61 | 2025.41 | 884.50 | 2909.91 | 0.268 | 2068.4 | FAIL | new |
| `chela_slay` | 3399.21 | 2472.18 | 1664.06 | 927.03 | 808.12 | 1735.15 | 0.884 | 4535.0 | FAIL | new |
| `frandi` | 4571.02 | 2559.43 | 1668.73 | 2011.59 | 890.70 | 2902.29 | 0.258 | 2069.7 | FAIL | new |
| `compot` | 6627.60 | 2408.06 | 1674.04 | 4219.54 | 734.02 | 4953.56 | 0.229 | 3126.1 | FAIL | new |
| `hgrn2_gated_deltanet` | 236472.08 | 2893.96 | 1909.00 | 233578.12 | 984.96 | 234563.08 | 0.913 | 5828.8 | FAIL | new |
| `cr` | 12965.85 | 3167.63 | 1996.38 | 9798.22 | 1171.25 | 10969.47 | 0.259 | 2067.3 | FAIL | new |
| `dobi` | 3448.91 | 2819.45 | 2013.45 | 629.46 | 806.00 | 1435.46 | 0.269 | 1938.9 | FAIL | new |
| `cur` | 3460.92 | 3017.61 | 2607.06 | 443.31 | 410.55 | 853.86 | 0.302 | 2252.2 | FAIL | new |
| `erc` | 3553.69 | 4113.58 | 2722.27 | -559.89 | 1391.31 | 831.42 | 0.267 | 1934.3 | FAIL | new |
| `aegis_hybrid` | 1273.71 | 6509.35 | 6507.68 | -5235.64 | 1.67 | -5233.97 | 0.722 | 4131.7 | FAIL | new |


## Section B: Downstream Classification Benchmarks

Zero-shot & post-recovery fine-tuning accuracy across DistilBERT, RoBERTa, and GPT-2 (SST-2 / MRPC).

| Module | Architecture Family | Replacement Type | DistilBERT Acc (%) | RoBERTa Acc (%) | GPT-2 Acc (%) | Mean Acc (%) | Rank | Source |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `liftr` | General | Full Replacement | - | - | - | 0.00% | 1 | old |
| `lillama` | General | Full Replacement | - | - | - | 0.00% | 2 | old |
| `mamba_shedder` | General | Full Replacement | - | - | - | 0.00% | 3 | old |
| `mgie_svd` | General | Full Replacement | - | - | - | 0.00% | 4 | old |
| `mlorq` | General | Full Replacement | - | - | - | 0.00% | 5 | old |
| `aasvd` | General | Full Replacement | - | - | - | 0.00% | 6 | old |
| `adaptive_compressor` | General | Full Replacement | - | - | - | 0.00% | 7 | old |
| `adaptive_svd_attention` | General | Full Replacement | - | - | - | 0.00% | 8 | old |
| `adaptive_svd_attn` | General | Full Replacement | - | - | - | 0.00% | 9 | old |
| `aegis_hybrid` | General | Full Replacement | - | - | - | 0.00% | 10 | old |
| `aegis_v1` | General | Full Replacement | - | - | - | 0.00% | 11 | old |
| `aegis_v1_module` | General | Full Replacement | - | - | - | 0.00% | 12 | old |
| `aegis_v2` | General | Full Replacement | - | - | - | 0.00% | 13 | old |
| `aegis_v3` | General | Full Replacement | - | - | - | 0.00% | 14 | old |
| `aegis_v3_module` | General | Full Replacement | - | - | - | 0.00% | 15 | old |
| `aegis_v4` | General | Full Replacement | - | - | - | 0.00% | 16 | old |
| `aegis_v4_module` | General | Full Replacement | - | - | - | 0.00% | 17 | old |
| `aegis_v5` | General | Full Replacement | - | - | - | 0.00% | 18 | old |
| `aegis_v5_module` | General | Full Replacement | - | - | - | 0.00% | 19 | old |
| `asvd` | General | Full Replacement | - | - | - | 0.00% | 20 | old |
| `block_transformer_hiba` | General | Full Replacement | - | - | - | 0.00% | 21 | old |
| `compressed_attention` | General | Full Replacement | - | - | - | 0.00% | 22 | old |
| `compressed_attention_layer` | General | Full Replacement | - | - | - | 0.00% | 23 | old |
| `cosformer` | General | Full Replacement | - | - | - | 0.00% | 24 | old |
| `cur_moe` | General | Full Replacement | - | - | - | 0.00% | 25 | old |
| `decay_linear_attention` | General | Full Replacement | - | - | - | 0.00% | 26 | old |
| `delta_kv` | General | Full Replacement | - | - | - | 0.00% | 27 | old |
| `factorised_matrix_state` | General | Full Replacement | - | - | - | 0.00% | 28 | old |
| `fms` | General | Full Replacement | - | - | - | 0.00% | 29 | old |
| `hymba_transx` | General | Full Replacement | - | - | - | 0.00% | 30 | old |
| `infini` | General | Full Replacement | - | - | - | 0.00% | 31 | old |
| `infini_attention` | General | Full Replacement | - | - | - | 0.00% | 32 | old |
| `jamba_info` | General | Full Replacement | - | - | - | 0.00% | 33 | old |
| `kronecker_tt_lora` | General | LoRA Adapter | - | - | - | 0.00% | 34 | old |
| `kvtc` | General | Full Replacement | - | - | - | 0.00% | 35 | old |
| `leka_v2` | General | Full Replacement | - | - | - | 0.00% | 36 | old |
| `leka_v3` | General | Full Replacement | - | - | - | 0.00% | 37 | old |
| `lola_attention` | General | Full Replacement | - | - | - | 0.00% | 38 | old |
| `lola_attn` | General | Full Replacement | - | - | - | 0.00% | 39 | old |
| `lora_attention` | General | LoRA Adapter | - | - | - | 0.00% | 40 | old |
| `lora_attn` | General | LoRA Adapter | - | - | - | 0.00% | 41 | old |
| `lsh` | General | Full Replacement | - | - | - | 0.00% | 42 | old |
| `lsh_attention` | General | Full Replacement | - | - | - | 0.00% | 43 | old |
| `minimal_aurora` | General | Full Replacement | - | - | - | 0.00% | 44 | old |
| `minimal_nova` | General | Full Replacement | - | - | - | 0.00% | 45 | old |
| `minimal_prism` | General | Full Replacement | - | - | - | 0.00% | 46 | old |
| `minimal_svd` | General | Full Replacement | - | - | - | 0.00% | 47 | old |
| `minimal_svd_v2` | General | Full Replacement | - | - | - | 0.00% | 48 | old |
| `mixattention_moska` | General | Full Replacement | - | - | - | 0.00% | 49 | old |
| `moba` | General | Full Replacement | - | - | - | 0.00% | 50 | old |
| `modegpt` | General | Full Replacement | - | - | - | 0.00% | 51 | old |
| `moe_deltakv_paged` | General | Full Replacement | - | - | - | 0.00% | 52 | old |
| `nasc` | General | Full Replacement | - | - | - | 0.00% | 53 | old |
| `nasc_attention` | General | Full Replacement | - | - | - | 0.00% | 54 | old |
| `nli_softmax` | General | Full Replacement | - | - | - | 0.00% | 55 | old |
| `nmf` | General | Full Replacement | - | - | - | 0.00% | 56 | old |
| `oats` | General | Full Replacement | - | - | - | 0.00% | 57 | old |
| `optimal_subspace_projection` | General | Full Replacement | - | - | - | 0.00% | 58 | old |
| `padre` | General | Full Replacement | - | - | - | 0.00% | 59 | old |
| `phantom` | General | Full Replacement | - | - | - | 0.00% | 60 | old |
| `primal` | General | Full Replacement | - | - | - | 0.00% | 61 | old |
| `progressive_token_pruning` | General | Full Replacement | - | - | - | 0.00% | 62 | old |
| `projection_factorized_attn` | General | Full Replacement | - | - | - | 0.00% | 63 | old |
| `proxsparse` | General | Full Replacement | - | - | - | 0.00% | 64 | old |
| `proxsparse_pruner` | General | Full Replacement | - | - | - | 0.00% | 65 | old |
| `qcqa` | General | Full Replacement | - | - | - | 0.00% | 66 | old |
| `replaceme` | General | Full Replacement | - | - | - | 0.00% | 67 | old |
| `res_svd` | General | Full Replacement | - | - | - | 0.00% | 68 | old |
| `roast` | General | Full Replacement | - | - | - | 0.00% | 69 | old |
| `rotational_slicegpt_int4` | General | Full Replacement | - | - | - | 0.00% | 70 | old |
| `rpc` | General | Full Replacement | - | - | - | 0.00% | 71 | old |
| `rpca` | General | Full Replacement | - | - | - | 0.00% | 72 | old |
| `saes_svd_v2` | General | Full Replacement | - | - | - | 0.00% | 73 | old |
| `salr` | General | Full Replacement | - | - | - | 0.00% | 74 | old |
| `salt` | General | Full Replacement | - | - | - | 0.00% | 75 | old |
| `salt_attention` | General | Full Replacement | - | - | - | 0.00% | 76 | old |
| `sdrm` | General | Full Replacement | - | - | - | 0.00% | 77 | old |
| `slicegpt` | General | Full Replacement | - | - | - | 0.00% | 78 | old |
| `slicegpt_pca` | General | Full Replacement | - | - | - | 0.00% | 79 | old |
| `slim` | General | Full Replacement | - | - | - | 0.00% | 80 | old |
| `smart_weighted_ensemble` | General | Full Replacement | - | - | - | 0.00% | 81 | old |
| `soa_attn` | General | Full Replacement | - | - | - | 0.00% | 82 | old |
| `soa_fused_module` | General | Full Replacement | - | - | - | 0.00% | 83 | old |
| `soa_module` | General | Full Replacement | - | - | - | 0.00% | 84 | old |
| `soafusedattention` | General | Full Replacement | - | - | - | 0.00% | 85 | old |
| `soka` | General | Full Replacement | - | - | - | 0.00% | 86 | old |
| `span_expanded` | General | Full Replacement | - | - | - | 0.00% | 87 | old |
| `sparsegpt` | General | Full Replacement | - | - | - | 0.00% | 88 | old |
| `spectra_v2` | General | Full Replacement | - | - | - | 0.00% | 89 | old |
| `spectraformer` | General | Full Replacement | - | - | - | 0.00% | 90 | old |
| `spectraformer_attention` | General | Full Replacement | - | - | - | 0.00% | 91 | old |
| `squeezing_heads` | General | Full Replacement | - | - | - | 0.00% | 92 | old |
| `squeezing_heads_distill` | General | Full Replacement | - | - | - | 0.00% | 93 | old |
| `squeezing_heads_distiller` | General | Full Replacement | - | - | - | 0.00% | 94 | old |
| `subspace_projection` | General | Full Replacement | - | - | - | 0.00% | 95 | old |
| `svd_compressed` | General | Full Replacement | - | - | - | 0.00% | 96 | old |
| `svd_linear` | General | Full Replacement | - | - | - | 0.00% | 97 | old |
| `svd_linear_attn` | General | Full Replacement | - | - | - | 0.00% | 98 | old |
| `svd_llm_v2` | General | Full Replacement | - | - | - | 0.00% | 99 | old |
| `svd_proj` | General | Full Replacement | - | - | - | 0.00% | 100 | old |
| `svd_proj_attn` | General | Full Replacement | - | - | - | 0.00% | 101 | old |
| `svd_projection` | General | Full Replacement | - | - | - | 0.00% | 102 | old |
| `swift_svd` | General | Full Replacement | - | - | - | 0.00% | 103 | old |
| `swift_svd_act` | General | Full Replacement | - | - | - | 0.00% | 104 | old |
| `template_attention` | General | Full Replacement | - | - | - | 0.00% | 105 | old |
| `tomoe` | General | Full Replacement | - | - | - | 0.00% | 106 | old |
| `tomoe_compressor` | General | Full Replacement | - | - | - | 0.00% | 107 | old |
| `tt` | General | Full Replacement | - | - | - | 0.00% | 108 | old |
| `uniql` | General | Full Replacement | - | - | - | 0.00% | 109 | old |
| `weighted_ensemble` | General | Full Replacement | - | - | - | 0.00% | 110 | old |
| `zerokv_neo` | General | Full Replacement | - | - | - | 0.00% | 111 | old |
| `adaptive_sparse` | General | Full Replacement | - | - | - | 0.00% | 112 | old |
| `adaptive_sparse_attention` | General | Full Replacement | - | - | - | 0.00% | 113 | old |
| `aegis_spectron_fp8` | General | Full Replacement | - | - | - | 0.00% | 114 | old |
| `aegis_v6` | General | Full Replacement | - | - | - | 0.00% | 115 | old |
| `cluster` | General | Full Replacement | - | - | - | 0.00% | 116 | old |
| `compressed_mha` | General | Full Replacement | - | - | - | 0.00% | 117 | old |
| `dobi` | General | Full Replacement | - | - | - | 0.00% | 118 | old |
| `dynamic_entropy` | General | Full Replacement | - | - | - | 0.00% | 119 | old |
| `full` | General | Full Replacement | - | - | - | 0.00% | 120 | old |
| `fused_svd_kernel` | General | Full Replacement | - | - | - | 0.00% | 121 | old |
| `galerkin` | General | Full Replacement | - | - | - | 0.00% | 122 | old |
| `galerkin_attention` | General | Full Replacement | - | - | - | 0.00% | 123 | old |
| `gated_delta_net` | General | Full Replacement | - | - | - | 0.00% | 124 | old |
| `gated_deltanet_block` | General | Full Replacement | - | - | - | 0.00% | 125 | old |
| `gated_linear_attention` | General | Full Replacement | - | - | - | 0.00% | 126 | old |
| `goldfinch_ahn` | General | Full Replacement | - | - | - | 0.00% | 127 | old |
| `granite_hybrid` | General | Full Replacement | - | - | - | 0.00% | 128 | old |
| `griffin_ssd` | General | Full Replacement | - | - | - | 0.00% | 129 | old |
| `hamming` | General | Full Replacement | - | - | - | 0.00% | 130 | old |
| `hgrn2` | General | Full Replacement | - | - | - | 0.00% | 131 | old |
| `hgrn2_gated_deltanet` | General | Full Replacement | - | - | - | 0.00% | 132 | old |
| `hgrn2_layer` | General | Full Replacement | - | - | - | 0.00% | 133 | old |
| `hgrn2_outer` | General | Full Replacement | - | - | - | 0.00% | 134 | old |
| `hla` | General | Full Replacement | - | - | - | 0.00% | 135 | old |
| `hla_attention` | General | Full Replacement | - | - | - | 0.00% | 136 | old |
| `huffman_coder` | General | Full Replacement | - | - | - | 0.00% | 137 | old |
| `hybrid_svd_layer` | General | Full Replacement | - | - | - | 0.00% | 138 | old |
| `jamba` | General | Full Replacement | - | - | - | 0.00% | 139 | old |
| `kda` | General | Full Replacement | - | - | - | 0.00% | 140 | old |
| `kronecker_linear` | General | Full Replacement | - | - | - | 0.00% | 141 | old |
| `leka_v1` | General | Full Replacement | - | - | - | 0.00% | 142 | old |
| `linformer` | General | Full Replacement | - | - | - | 0.00% | 143 | old |
| `mamba` | General | Full Replacement | - | - | - | 0.00% | 144 | old |
| `mamba_inject` | General | LoRA Adapter | - | - | - | 0.00% | 145 | old |
| `mamba_svd_flash` | General | Full Replacement | - | - | - | 0.00% | 146 | old |
| `mla` | General | Full Replacement | - | - | - | 0.00% | 147 | old |
| `nexus` | General | Full Replacement | - | - | - | 0.00% | 148 | old |
| `nha` | General | Full Replacement | - | - | - | 0.00% | 149 | old |
| `nha_compressor` | General | Full Replacement | - | - | - | 0.00% | 150 | old |
| `nsa` | General | Full Replacement | - | - | - | 0.00% | 151 | old |
| `nsa_attention` | General | Full Replacement | - | - | - | 0.00% | 152 | old |
| `nystrom` | General | Full Replacement | - | - | - | 0.00% | 153 | old |
| `oha` | General | Full Replacement | - | - | - | 0.00% | 154 | old |
| `optimized_hybrid_attention` | General | Full Replacement | - | - | - | 0.00% | 155 | old |
| `orion` | General | Full Replacement | - | - | - | 0.00% | 156 | old |
| `orion_module` | General | Full Replacement | - | - | - | 0.00% | 157 | old |
| `performer` | General | Full Replacement | - | - | - | 0.00% | 158 | old |
| `rala` | General | Full Replacement | - | - | - | 0.00% | 159 | old |
| `rala_compressor` | General | Full Replacement | - | - | - | 0.00% | 160 | old |
| `reap_moe` | General | Full Replacement | - | - | - | 0.00% | 161 | old |
| `regla` | General | Full Replacement | - | - | - | 0.00% | 162 | old |
| `roast_hashed_linear` | General | Full Replacement | - | - | - | 0.00% | 163 | old |
| `routing` | General | Full Replacement | - | - | - | 0.00% | 164 | old |
| `saes_svd_compressor` | General | Full Replacement | - | - | - | 0.00% | 165 | old |
| `samba_m3` | General | Full Replacement | - | - | - | 0.00% | 166 | old |
| `samba_moe` | General | Full Replacement | - | - | - | 0.00% | 167 | old |
| `sambam3` | General | Full Replacement | - | - | - | 0.00% | 168 | old |
| `sambamoe` | General | Full Replacement | - | - | - | 0.00% | 169 | old |
| `simple_ssm` | General | Full Replacement | - | - | - | 0.00% | 170 | old |
| `spectral_mamba` | General | Full Replacement | - | - | - | 0.00% | 171 | old |
| `spectron` | General | Full Replacement | - | - | - | 0.00% | 172 | old |
| `svd-mamba` | General | Full Replacement | - | - | - | 0.00% | 173 | old |
| `svd` | General | Full Replacement | - | - | - | 0.00% | 174 | old |
| `svd_linear_decay` | General | Full Replacement | - | - | - | 0.00% | 175 | old |
| `svd_mamba` | General | Full Replacement | - | - | - | 0.00% | 176 | old |
| `svd_sequence` | General | Full Replacement | - | - | - | 0.00% | 177 | old |
| `svd_sequence_attention` | General | Full Replacement | - | - | - | 0.00% | 178 | old |
| `tlinformer` | General | Full Replacement | - | - | - | 0.00% | 179 | old |
| `tlinformer_attention` | General | Full Replacement | - | - | - | 0.00% | 180 | old |
| `token_compressed` | General | Full Replacement | - | - | - | 0.00% | 181 | old |
| `tpa` | General | Full Replacement | - | - | - | 0.00% | 182 | old |
| `tt_linear` | General | Full Replacement | - | - | - | 0.00% | 183 | old |
| `upi_mamba` | General | Full Replacement | - | - | - | 0.00% | 184 | old |
| `upi_stuffed_mamba` | General | Full Replacement | - | - | - | 0.00% | 185 | old |
| `vig` | General | Full Replacement | - | - | - | 0.00% | 186 | old |
| `vspec` | General | Full Replacement | - | - | - | 0.00% | 187 | old |
| `wgqa` | General | Full Replacement | - | - | - | 0.00% | 188 | old |
| `xattention` | General | Full Replacement | - | - | - | 0.00% | 189 | old |
| `zeros` | General | Full Replacement | - | - | - | 0.00% | 190 | old |


## Section C: Module Architecture Screening (236 Tracked Modules)

| Module | Compatibility Status | Passed Families | DistilBERT | RoBERTa | GPT-2 | Family Group | Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| `a3` | FULLY_COMPATIBLE | 3/3 | PASS (90.6%) | PASS (92.66%) | PASS (87.16%) | SVD / Low-Rank | old+new |
| `aasvd` | FULLY_COMPATIBLE | 3/3 | PASS (84.06%) | PASS (80.85%) | PASS (75.11%) | SVD / Low-Rank | old+new |
| `adaptive_compressor` | FULLY_COMPATIBLE | 3/3 | PASS (88.07%) | PASS (86.35%) | PASS (81.88%) | SVD / Low-Rank | old+new |
| `adaptive_ranks` | FULLY_COMPATIBLE | 3/3 | PASS (82.11%) | PASS (79.47%) | PASS (79.82%) | SVD / Low-Rank | old+new |
| `adaptive_span` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.5%) | PASS (85.67%) | SVD / Low-Rank | old+new |
| `adaptive_sparse` | PARTIALLY_COMPATIBLE | 1/3 | NOT_RUN (0.0%) | PASS (94.38%) | NOT_RUN (0.0%) | Pruning / Sparsity | old+new |
| `adaptive_sparse_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.5%) | PASS (85.67%) | Pruning / Sparsity | old+new |
| `adaptive_sparse_attention_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (90.6%) | PASS (87.84%) | Pruning / Sparsity | old+new |
| `adaptive_svd_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.5%) | PASS (85.67%) | SVD / Low-Rank | old+new |
| `adaptive_svd_attn` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.5%) | PASS (85.67%) | SVD / Low-Rank | old+new |
| `adasplash` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.5%) | PASS (85.67%) | SVD / Low-Rank | old+new |
| `adasvd` | FULLY_COMPATIBLE | 3/3 | PASS (90.37%) | PASS (92.32%) | PASS (83.83%) | SVD / Low-Rank | old+new |
| `admm` | FULLY_COMPATIBLE | 3/3 | PASS (90.14%) | PASS (93.35%) | PASS (87.73%) | SVD / Low-Rank | old+new |
| `admm_pruner` | PARTIALLY_COMPATIBLE | 2/3 | PASS (89.79%) | PASS (90.25%) | TIMEOUT (0.0%) | Pruning / Sparsity | old+new |
| `aegis_hybrid` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `aegis_spectron_fp8` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | Quantization / Compression | old+new |
| `aegis_v1` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `aegis_v1_module` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `aegis_v2` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `aegis_v3` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `aegis_v3_module` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `aegis_v4` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `aegis_v4_module` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `aegis_v5` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `aegis_v5_module` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `aegis_v6` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `armor` | FULLY_COMPATIBLE | 3/3 | PASS (89.91%) | PASS (92.66%) | PASS (87.61%) | SVD / Low-Rank | old+new |
| `astro` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (92.89%) | PASS (88.65%) | SVD / Low-Rank | old+new |
| `asvd` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `bam_umoe` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | MoE | old+new |
| `block_transformer_hiba` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `cachenotes` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (92.66%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `chela_slay` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `cluster` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `come` | PARTIALLY_COMPATIBLE | 2/3 | PASS (86.35%) | PASS (72.48%) | CRASH (0.0%) | SVD / Low-Rank | old+new |
| `compressed_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `compressed_attention_layer` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `compressed_mha` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `cosformer` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | Linear Attention | old+new |
| `cur` | INCOMPATIBLE | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | SVD / Low-Rank | old+new |
| `cur_linear` | INCOMPATIBLE | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | Linear Attention | old+new |
| `cur_moe` | INCOMPATIBLE | 0/3 | PASS_NO_COMPRESSION (90.94%) | PASS_NO_COMPRESSION (93.46%) | PASS_NO_COMPRESSION (89.11%) | MoE | old+new |
| `cure` | INCOMPATIBLE | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | SVD / Low-Rank | old+new |
| `decay_linear_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | Linear Attention | old+new |
| `delta_kv` | PARTIALLY_COMPATIBLE | 1/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS (87.39%) | SVD / Low-Rank | old+new |
| `direct_svd` | FULLY_COMPATIBLE | 3/3 | PASS (88.99%) | PASS (92.43%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `dobi` | INCOMPATIBLE | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | SVD / Low-Rank | old+new |
| `dobi_svd` | PARTIALLY_COMPATIBLE | 2/3 | PASS (90.48%) | PASS (90.6%) | PASS_LOW (49.54%) | SVD / Low-Rank | old+new |
| `dynamic_entropy` | INCOMPATIBLE | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | SVD / Low-Rank | old+new |
| `dynamic_entropy_svd` | PARTIALLY_COMPATIBLE | 2/3 | PASS (90.94%) | PASS (90.6%) | PASS_LOW (49.54%) | SVD / Low-Rank | old+new |
| `eat` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `enlighten` | FULLY_COMPATIBLE | 3/3 | PASS (90.14%) | PASS (92.66%) | PASS (85.78%) | SVD / Low-Rank | old+new |
| `entropy_budget` | INCOMPATIBLE | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | SVD / Low-Rank | old+new |
| `erc` | INCOMPATIBLE | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | SVD / Low-Rank | old+new |
| `expert_choice_moe` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | MoE | old+new |
| `factorised_matrix_state` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `faq` | PARTIALLY_COMPATIBLE | 1/3 | PASS (90.83%) | TIMEOUT (0.0%) | CRASH (0.0%) | SVD / Low-Rank | old+new |
| `fasa` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `flap` | PARTIALLY_COMPATIBLE | 1/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS (88.76%) | SVD / Low-Rank | old+new |
| `flash3` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `flash3_fp8` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | Quantization / Compression | old+new |
| `flat_llm` | PARTIALLY_COMPATIBLE | 2/3 | PASS_NO_COMPRESSION (90.71%) | PASS (93.69%) | PASS (87.5%) | SVD / Low-Rank | old+new |
| `fms` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `frandi` | INCOMPATIBLE | 0/3 | PASS_NO_COMPRESSION (90.94%) | PASS_NO_COMPRESSION (93.46%) | PASS_NO_COMPRESSION (87.84%) | SVD / Low-Rank | old+new |
| `full` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `fused_svd_kernel` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `fusion_attn` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `galerkin` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | Linear Attention | old+new |
| `galerkin_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | Linear Attention | old+new |
| `gated_delta_net` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (94.15%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `gated_deltanet_block` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SSM / Linear RNN | old+new |
| `gated_linear_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | Linear Attention | old+new |
| `goldfinch_ahn` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `gqa` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `granite_hybrid` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `grecodec` | INCOMPATIBLE | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | SVD / Low-Rank | old+new |
| `griffin_ssd` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `halo_regla` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `hamming` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SVD / Low-Rank | old+new |
| `hashed_linear` | INCOMPATIBLE | 0/3 | CRASH_NO_OUTPUT (0.0%) | PASS_NO_COMPRESSION (94.38%) | CRASH (0.0%) | Linear Attention | old+new |
| `hashednets` | INCOMPATIBLE | 0/3 | PASS_NO_COMPRESSION (90.71%) | PASS_NO_COMPRESSION (94.38%) | PASS_NO_COMPRESSION (89.33%) | SVD / Low-Rank | old+new |
| `hgrn2_gated_deltanet` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (94.15%) | PASS (87.84%) | SSM / Linear RNN | old+new |
| `hgrn2_layer` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SSM / Linear RNN | old+new |
| `hgrn2_outer` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (89.33%) | SSM / Linear RNN | old+new |
| `hla` | FULLY_COMPATIBLE | 3/3 | PASS (90.71%) | PASS (94.38%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `hla_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `huffman` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `huffman_coder` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `hybrid_svd` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `hybrid_svd_layer` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `hybridkv` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `hymba_transx` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `infini` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `infini_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `jamba` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `jamba_info` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `kda` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `kmeans` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `kronecker` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `kronecker_linear` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | Linear Attention | old+new |
| `kronecker_tt_lora` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `kvtc` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `leka_v1` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `leka_v2` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `leka_v3` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `liftr` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.0%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `lillama` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.0%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `linformer` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `lola_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `lola_attn` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `lora_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `lora_attn` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.0%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `lsc` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `lsh` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `lsh_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `mamba` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SSM / Linear RNN | old+new |
| `mamba_inject` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SSM / Linear RNN | old+new |
| `mamba_shedder` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SSM / Linear RNN | old+new |
| `mamba_svd_flash` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (60.78%) | SSM / Linear RNN | old+new |
| `mgie_budget` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (88.76%) | PASS (81.31%) | SVD / Low-Rank | old+new |
| `mgie_svd` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `minimal_aurora` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `minimal_nova` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `minimal_prism` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `minimal_svd` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `minimal_svd_v2` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `mixattention_moska` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `mla` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `mlorq` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `moba` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `modegpt` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `moe_deltakv_paged` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | MoE | old+new |
| `nasc` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `nasc_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `nexus` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `nha` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `nha_compressor` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `nli_softmax` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `nmf` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `nsa` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `nsa_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `nystrom` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `oats` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `oha` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `optimal_subspace_projection` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `optimized_hybrid_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `orion` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `orion_module` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `padre` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `performer` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | Linear Attention | old+new |
| `phantom` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `primal` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `progressive_token_pruning` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | Pruning / Sparsity | old+new |
| `projection_factorized_attn` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `proxsparse` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | Pruning / Sparsity | old+new |
| `proxsparse_pruner` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | Pruning / Sparsity | old+new |
| `qcqa` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `rala` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `rala_compressor` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `rala_rerun` | INCOMPATIBLE | 0/3 | NOT_RUN (0.0%) | PASS_LOW (51.03%) | NOT_RUN (0.0%) | SVD / Low-Rank | old+new |
| `reap_moe` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | MoE | old+new |
| `regla` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `replace` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `replaceme` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `res_svd` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (60.09%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `roast` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `roast_hashed_linear` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | Linear Attention | old+new |
| `rope` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `rotational_slicegpt_int4` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | Quantization / Compression | old+new |
| `routing` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `routing_rerun` | INCOMPATIBLE | 0/3 | NOT_RUN (0.0%) | PASS_LOW (50.92%) | NOT_RUN (0.0%) | SVD / Low-Rank | old+new |
| `rpc` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (60.09%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `rpca` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `saes_svd_compressor` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `saes_svd_v2` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `salr` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `salt` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `salt_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `sambam3` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `sambamoe` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | MoE | old+new |
| `sdrm` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `simple_ssm` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SSM / Linear RNN | old+new |
| `slicegpt` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | Quantization / Compression | old+new |
| `slicegpt_pca` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | Quantization / Compression | old+new |
| `slim` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `smart_weighted_ensemble` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `soa_attn` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `soa_fused_module` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `soa_module` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `soafusedattention` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `soka` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `span_expanded` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `sparsegpt` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | Pruning / Sparsity | old+new |
| `spectra_v2` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `spectraformer` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `spectraformer_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `spectral_mamba` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SSM / Linear RNN | old+new |
| `spectron` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `squeezing_heads` | PARTIALLY_COMPATIBLE | 1/3 | PASS (90.94%) | PASS_LOW (50.92%) | PASS_LOW (49.54%) | SVD / Low-Rank | old+new |
| `squeezing_heads_distill` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `squeezing_heads_distiller` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `subspace_projection` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `svd-mamba` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (60.78%) | SSM / Linear RNN | old+new |
| `svd_compressed` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `svd_linear` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | Linear Attention | old+new |
| `svd_linear_attn` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | Linear Attention | old+new |
| `svd_linear_decay` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | Linear Attention | old+new |
| `svd_llm_v2` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `svd_mamba` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (60.78%) | SSM / Linear RNN | old+new |
| `svd_proj` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `svd_proj_attn` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `svd_projection` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `svd_sequence` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `svd_sequence_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `swift_svd` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `swift_svd_act` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `template_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `tlinformer` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `tlinformer_attention` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `token_compressed` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `tomoe` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | MoE | old+new |
| `tomoe_compressor` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | MoE | old+new |
| `tpa` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.8%) | SVD / Low-Rank | old+new |
| `tt` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `tt_linear` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | Linear Attention | old+new |
| `uniql` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `upi_mamba` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SSM / Linear RNN | old+new |
| `upi_stuffed_mamba` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SSM / Linear RNN | old+new |
| `vig` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |
| `vspec` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `weighted_ensemble` | FULLY_COMPATIBLE | 3/3 | PASS (90.48%) | PASS (75.11%) | PASS (60.78%) | SVD / Low-Rank | old+new |
| `wgqa` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `wgqa_rerun` | PARTIALLY_COMPATIBLE | 1/3 | PASS (90.37%) | PASS_LOW (50.92%) | PASS_LOW (49.31%) | SVD / Low-Rank | old+new |
| `xattention` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `zerokv_neo` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (87.84%) | SVD / Low-Rank | old+new |
| `zeros` | FULLY_COMPATIBLE | 3/3 | PASS (90.94%) | PASS (75.46%) | PASS (57.91%) | SVD / Low-Rank | old+new |


## Section D: Known Bugs, Code Issues & Forensic Root Causes

### LoRA unscaled multiplier bug (adaptive_svd, asvd)
- **Root Cause:** ΔW = B·A was computed directly without multiplying by (α / r).
- **Impact:** Caused severe gradient explosion (~2x scaling overshoot), destabilizing recovery steps.
- **Status:** FIXED (lora_scaling parameter and alpha/r multiplier integrated across AdaSplashLinear and ASVD).

### Recurrence state explosion in aegis_spectron_fp8
- **Root Cause:** Decay parameter decay_param initialized to 4.5951 (sigmoid ≈ 0.99). Over 128 tokens, unconstrained recurrence compounded state magnitude by ~100x.
- **Impact:** Pre-KD perplexity exploded to 37,728.77.
- **Status:** FIXED on ablation branch (decay_param init to 0.0, linear_B/C scaled by 1/√window, LayerNorm post-state recurrence added). Reduced PPL from 37,728 to 1,942.

### SDPA dead-code in _local_window_attention
- **Root Cause:** The fast-path branch checked `if token_valid is None:`. On WikiText-2, attention_mask is passed as all-ones tensor, so token_valid was never None.
- **Impact:** SDPA was bypassed 100% of the time, dropping execution into slow looped causal attention (~2s/step latency).
- **Status:** FIXED (condition relaxed to `if token_valid is None or bool(token_valid.all()):`). Latency dropped to sub-second when active.

### Local window radius=32 kills long-range context
- **Root Cause:** A hard radius of 32 tokens truncated the receptive field, discarding 75% of context in 128-token GPT-2 sequences.
- **Impact:** Pre-KD PPL remained elevated (>1,300) even with balanced gating.
- **Status:** MITIGATED via `AEGIS_DISABLE_LOCAL=1` unconstrained causal fallback; dropped Pre-KD PPL to 885.81.

### Router bias (4,0,0) forces 94% local attention
- **Root Cause:** Initialization bias strongly prioritized the local sliding window branch (α ≈ 0.94), starving linear and global branches.
- **Impact:** Prevented the student model from leveraging pretrained global attention projections.
- **Status:** MITIGATED via balanced router bias `(1.0, 1.0, 1.0)` and env override `AEGIS_ROUTER_BIAS`.

### adasplash_attention.py line 277 indentation error
- **Root Cause:** Accidental syntax indentation error inside `__init__` block.
- **Impact:** SyntaxError prevented module import on Python 3.10+ environments.
- **Status:** FIXED (clean indentation restored and verified via import test).

### Triton kernel dtype mismatch
- **Root Cause:** Custom Triton kernels only accept fp16/bf16 tensors; float32 input tensors trigger type mismatch.
- **Impact:** Kernel execution failure on standard fp32 training loops.
- **Status:** NOT FIXED — currently falls back to PyTorch native execution.

### Triton low_rank kernel execution error (tl.dot shape mismatch)
- **Root Cause:** Block size dimensions in tl.dot did not match dynamically padded sequence ranks.
- **Impact:** Kernel crashes on odd rank dimensions.
- **Status:** NOT FIXED — caught in try/except fallback block.

### LR override bug (1.25e-04 instead of 5e-06 in some runs)
- **Root Cause:** In certain recovery configs, student learning rate defaulted to high warmup rate (1.25e-4) instead of conservative fine-tuning rate (5e-6).
- **Impact:** Catastrophic forgetting observed in sensitive hybrid modules (e.g., aegis_hybrid diverged to 6,507 PPL).
- **Status:** NEEDS VERIFICATION across execution runner scripts.

### batch_size default=2 but docstring says 8 in run_kd_all.py
- **Root Cause:** Mismatched documentation vs CLI argparse default.
- **Impact:** Minor operational confusion regarding throughput and step scaling.
- **Status:** NOT FIXED — cosmetic/config discrepancy.

### timeout=1800s too tight for slow modules
- **Root Cause:** Certain heavy RNN/SSM modules (e.g., hla, jamba) require >1800s to finish full KD passes.
- **Impact:** Modules prematurely terminated with TIMEOUT status.
- **Status:** NOT FIXED — recommend increasing to 3600s for recurrent modules.

### Per-module subprocess reloading (no model cache)
- **Root Cause:** Each module run spawns an isolated Python interpreter process and re-downloads/re-loads GPT-2 base weights.
- **Impact:** Wastes 5-8s per benchmark iteration.
- **Status:** NOT FIXED — intentional for memory isolation, but causes overhead.

### FP8 emulation on Windows (no native torch.float8 cast path)
- **Root Cause:** Windows CUDA PyTorch builds lack full native fp8 casting intrinsics, requiring torch.clamp/float emulation.
- **Impact:** FP8 simulation accounts for ~25% of forward step latency on Windows RTX 4070 Ti.
- **Status:** NOT FIXED — hardware/OS software stack limitation.

## Section E: Cross-Cutting Findings & Architecture Viability

1. **Top Performers on GPT-2:** Modules utilizing local attention with adaptive span or low-rank projection (such as `adaptive_span`, `adasplash`, `cluster`, `expert_choice_moe`, and `flash3`) achieve post-KD perplexities between 48.41 and 52.23, outperforming the full teacher model (66.07) by ~21-27%.
2. **Incompatible Architectures:** Pure linear recurrent models and state-space models (`aegis_hybrid`, `chela_slay`, `compot`, `hgrn2_gated_deltanet`) exhibit severe inductive bias mismatch with GPT-2 pretrained causal softmax attention. Initial pre-KD perplexities exceed 1,000 to 200,000, requiring extensive recovery steps and risking divergence.
3. **Theoretical PPL Floor:** For low-rank SVD linear attention, the theoretical floor without KD is ~11.5 PPL on WikiText-2 (large sequence benchmarks) and ~48-52 PPL under 200-step distillation. Self-training consistently adds 10-30% additional PPL reduction beyond initial KD transfer.
4. **Module Selection Guidance:**
   - **Retain:** `adasplash`, `adaptive_span`, `flash3_fp8`, `expert_choice_moe`, `direct_svd`, `galerkin`.
   - **Deprecate/Isolate:** `aegis_hybrid` (unstable dynamics), `hgrn2_outer` (gradient instabilities), `cure`/`cur` (no compression benefit).

## Section F: Open Questions & Next Experiments

1. **Aegis-Spectron-FP8 Clean Recovery:** Now that recurrence explosion is mitigated (pre-KD PPL down to 885.81), run a full 200-step distillation with `AEGIS_DISABLE_LOCAL=1` and `router_bias=(1,1,1)` to verify whether post-KD PPL can reach <66.
2. **Triton Low-Rank Kernel Fix:** Align block dimension tiling to resolve tl.dot shape mismatch, eliminating PyTorch fallback overhead.
3. **Comprehensive LR Verification:** Standardize student KD learning rate across all test runner scripts to strictly prevent 1.25e-4 overshooting.
