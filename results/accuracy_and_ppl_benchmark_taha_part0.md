# Accuracy & Perplexity Benchmark (`taha` - `NVIDIA GeForce RTX 4070 Laptop GPU`)

- **Baseline RoBERTa SST-2 Accuracy**: **95.00%**
- **Baseline GPT-2 WikiText-2 PPL**: **27.50**

| # | Attention Module | SST-2 Accuracy (%) | WikiText-2 PPL | Latency (s) | Status / Error |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `adaptive_span_attention` | **95.0%** | **11458.2** | 2.18 | OK |
| 2 | `adaptive_sparse_attention` | **92.0%** | **27.54** | 1.56 | OK |
| 3 | `adaptive_svd_attention` | **52.0%** | **1333.74** | 8.36 | OK |
| 4 | `adasplash_attention` | **94.0%** | **75.7** | 1.98 | OK |
| 5 | `aegis_hybrid` | **52.0%** | N/A | 8.22 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 6 | `aegis_spectron_fp8` | **52.0%** | **2666.09** | 13.38 | OK |
| 7 | `aegis_v1` | **52.0%** | **1891.05** | 9.79 | OK |
| 8 | `aegis_v2` | **52.0%** | **1965.27** | 8.73 | OK |
| 9 | `aegis_v3` | **52.0%** | **2015.07** | 8.79 | OK |
| 10 | `aegis_v4` | **52.0%** | **2082.57** | 9.72 | OK |
| 11 | `aegis_v5` | **52.0%** | N/A | 8.2 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 12 | `aegis_v6` | **52.0%** | **1787.84** | 9.06 | OK |
| 13 | `alibi` | **92.0%** | **45221.59** | 2.06 | OK |
| 14 | `asvd_attention` | **79.0%** | **141.96** | 8.48 | OK |
| 15 | `bam_umoe` | **52.0%** | **749.82** | 5.15 | OK |
| 16 | `block_transformer_hiba` | **93.0%** | **112.24** | 2.32 | OK |
| 17 | `chela_slay` | **89.0%** | **47.03** | 4.83 | OK |
| 18 | `chela_slay_attention` | **91.0%** | **45.51** | 2.04 | OK |
| 19 | `cluster_attention` | **94.0%** | **32.34** | 2.24 | OK |
| 20 | `compressed_attention` | **52.0%** | **1355.87** | 4.89 | OK |
| 21 | `compressed_mha_attention` | **52.0%** | **1384.04** | 5.24 | OK |
| 22 | `cosformer` | **95.0%** | **569.27** | 2.73 | OK |
| 23 | `cosformer_attention` | **95.0%** | **478.08** | 1.95 | OK |
| 24 | `cross_attention` | **93.0%** | **27.5** | 1.57 | OK |
| 25 | `decay_linear_attention` | **66.0%** | **18157.22** | 2.21 | OK |
| 26 | `decay_linear_attn` | **66.0%** | **18157.22** | 1.97 | OK |
| 27 | `dynamic_conv_attention` | **52.0%** | **3096182.99** | 1.82 | OK |
| 28 | `eat` | **90.0%** | **31.43** | 1.87 | OK |
| 29 | `eat_attention` | **92.0%** | **29.83** | 2.2 | OK |
| 30 | `expert_choice_moe` | **92.0%** | **32.75** | 1.58 | OK |
| 31 | `factorised_matrix_state` | **70.0%** | **25843.83** | 1.73 | OK |
| 32 | `flash_attention3` | **92.0%** | **31.31** | 1.96 | OK |
| 33 | `flash_attention3_fp8` | **94.0%** | **31.25** | 1.51 | OK |
| 34 | `full` | **95.0%** | **27.5** | 1.17 | OK |
| 35 | `full_attention` | **93.0%** | **33.68** | 2.15 | OK |
| 36 | `fused_svd_kernel` | **52.0%** | **1297.06** | 8.45 | OK |
| 37 | `fusion_attn` | **94.0%** | **113.7** | 1.44 | OK |
| 38 | `galerkin_attention` | **54.0%** | **33.39** | 1.49 | OK |
| 39 | `gated_delta_net` | **52.0%** | **1436.25** | 2.9 | OK |
| 40 | `gated_deltanet_block` | **52.0%** | **1425.05** | 3.52 | OK |
| 41 | `gated_linear_attention` | **66.0%** | **18157.22** | 1.71 | OK |
| 42 | `gated_linear_attn` | **66.0%** | **18157.22** | 3.29 | OK |
| 43 | `gla_attention` | **55.0%** | **10446.07** | 2.91 | OK |
| 44 | `goldfinch_ahn` | **95.0%** | **8863.33** | 1.61 | OK |
| 45 | `gqa_attention` | **94.0%** | **32.16** | 1.83 | OK |
| 46 | `granite_hybrid` | **52.0%** | **3949.75** | 3.92 | OK |
| 47 | `griffin_ssd` | **91.0%** | **127.52** | 1.47 | OK |
| 48 | `halo_regla` | **91.0%** | **4164.81** | 1.49 | OK |
| 49 | `hamming_attention` | **60.0%** | **2851.32** | 1.44 | OK |
| 50 | `hgrn2_gated_deltanet` | **92.0%** | **4581.57** | 3.85 | OK |
| 51 | `hgrn2_gated_layer` | **92.0%** | **4597.19** | 2.31 | OK |
| 52 | `hgrn2_gated_outer` | **52.0%** | **4354.63** | 3.02 | OK |
| 53 | `hgrn2_outer` | **52.0%** | **4427.27** | 2.51 | OK |
| 54 | `hla_attention` | **50.0%** | **4669.51** | 2.01 | OK |
| 55 | `hybrid_attention` | **52.0%** | **104.04** | 2.08 | OK |
| 56 | `hybrid_local_kernel` | **52.0%** | **2086.72** | 8.62 | OK |
| 57 | `hymba_transx` | **94.0%** | **92.17** | 1.75 | OK |
| 58 | `jamba` | **93.0%** | **31.75** | 2.4 | OK |
| 59 | `jamba_info` | **55.0%** | **2807.01** | 2.16 | OK |
| 60 | `kda_attention` | **51.0%** | **404015.41** | 2.02 | OK |
| 61 | `landmark_local_attention` | **52.0%** | **1159.19** | 8.93 | OK |
| 62 | `linformer_attention` | **52.0%** | **1491.44** | 9.03 | OK |
| 63 | `lola_attention` | **52.0%** | **1311.49** | 8.53 | OK |
| 64 | `lora_attention` | **94.0%** | **30.71** | 1.51 | OK |
| 65 | `lsh_attention` | **52.0%** | **1109.58** | 8.23 | OK |
| 66 | `mamba_block` | **52.0%** | **4556.13** | 5.86 | OK |
