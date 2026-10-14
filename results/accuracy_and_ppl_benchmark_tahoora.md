# Accuracy & Perplexity Benchmark (`tahoora` - `NVIDIA GeForce GTX 1060`)

- **Baseline RoBERTa SST-2 Accuracy**: **95.00%**
- **Baseline GPT-2 WikiText-2 PPL**: **27.56**

| # | Attention Module | SST-2 Accuracy (%) | WikiText-2 PPL | Latency (s) | Status / Error |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `adaptive_span_attention` | **94.0%** | **10005.97** | 4.14 | OK |
| 2 | `adaptive_sparse_attention` | **94.0%** | **27.51** | 3.07 | OK |
| 3 | `adaptive_svd_attention` | **52.0%** | **1327.97** | 24.92 | OK |
| 4 | `adasplash_attention` | **94.0%** | **70.93** | 2.8 | OK |
| 5 | `aegis_hybrid` | **52.0%** | N/A | 23.12 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 6 | `aegis_spectron_fp8` | **52.0%** | **2461.67** | 36.04 | OK |
| 7 | `aegis_v1` | **52.0%** | **1970.29** | 24.24 | OK |
| 8 | `aegis_v2` | **52.0%** | **1752.41** | 23.92 | OK |
| 9 | `aegis_v3` | **52.0%** | **2064.85** | 25.29 | OK |
| 10 | `aegis_v4` | **52.0%** | **2017.02** | 27.39 | OK |
| 11 | `aegis_v5` | **52.0%** | N/A | 23.44 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 12 | `aegis_v6` | **52.0%** | **1678.71** | 23.65 | OK |
| 13 | `alibi` | **90.0%** | **30820.57** | 3.76 | OK |
| 14 | `asvd_attention` | **77.0%** | **145.54** | 25.15 | OK |
| 15 | `bam_umoe` | **52.0%** | **756.65** | 13.98 | OK |
| 16 | `block_transformer_hiba` | **93.0%** | **109.43** | 3.21 | OK |
| 17 | `chela_slay` | **92.0%** | **43.7** | 3.7 | OK |
| 18 | `chela_slay_attention` | **94.0%** | **43.96** | 3.45 | OK |
| 19 | `cluster_attention` | **94.0%** | **30.33** | 2.97 | OK |
| 20 | `compressed_attention` | **52.0%** | **1375.49** | 11.98 | OK |
| 21 | `compressed_mha_attention` | **52.0%** | **1386.27** | 12.25 | OK |
| 22 | `cosformer` | **95.0%** | **436.21** | 3.04 | OK |
| 23 | `cosformer_attention` | **94.0%** | **708.04** | 3.29 | OK |
| 24 | `cross_attention` | **95.0%** | **27.56** | 3.52 | OK |
| 25 | `decay_linear_attention` | **66.0%** | **18118.48** | 3.84 | OK |
| 26 | `decay_linear_attn` | **66.0%** | **18118.48** | 4.38 | OK |
| 27 | `dynamic_conv_attention` | **52.0%** | **3168440.28** | 3.02 | OK |
| 28 | `eat` | **94.0%** | **31.05** | 2.74 | OK |
| 29 | `eat_attention` | **94.0%** | **31.29** | 3.09 | OK |
| 30 | `expert_choice_moe` | **90.0%** | **30.45** | 4.17 | OK |
| 31 | `factorised_matrix_state` | **70.0%** | **25751.55** | 3.16 | OK |
| 32 | `flash_attention3` | **95.0%** | **30.93** | 3.07 | OK |
| 33 | `flash_attention3_fp8` | **94.0%** | **31.23** | 3.14 | OK |
| 34 | `full` | **95.0%** | **27.56** | 2.14 | OK |
| 35 | `full_attention` | **94.0%** | **31.29** | 3.36 | OK |
| 36 | `fused_svd_kernel` | **52.0%** | **1262.22** | 23.29 | OK |
| 37 | `fusion_attn` | **94.0%** | **95.08** | 3.43 | OK |
| 38 | `galerkin_attention` | **54.0%** | **32.1** | 3.57 | OK |
| 39 | `gated_delta_net` | **52.0%** | **1435.88** | 6.31 | OK |
| 40 | `gated_deltanet_block` | **52.0%** | **1430.28** | 5.57 | OK |
| 41 | `gated_linear_attention` | **66.0%** | **18118.48** | 3.64 | OK |
| 42 | `gated_linear_attn` | **66.0%** | **18118.48** | 3.28 | OK |
| 43 | `gla_attention` | **56.0%** | **10989.41** | 3.03 | OK |
| 44 | `goldfinch_ahn` | **94.0%** | **7263.56** | 3.6 | OK |
| 45 | `gqa_attention` | **93.0%** | **33.51** | 2.8 | OK |
| 46 | `granite_hybrid` | N/A | **4171.11** | 5.46 | RoBERTa error: CUDA out of memory. Tried to allocate 516.00 MiB. GPU 0 has a total capacity of 5.92 GiB of which 273.62 MiB is free. Including non-PyTorch memory, this process has 5.65 GiB memory in use. Of the allocated memory 5.30 GiB is allocated by PyTorch, and 240.43 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables);  |
| 47 | `griffin_ssd` | **92.0%** | **127.44** | 3.87 | OK |
| 48 | `halo_regla` | **93.0%** | **4138.65** | 3.24 | OK |
| 49 | `hamming_attention` | **73.0%** | **3457.97** | 2.67 | OK |
| 50 | `hgrn2_gated_deltanet` | **92.0%** | **4581.07** | 8.08 | OK |
| 51 | `hgrn2_gated_layer` | **92.0%** | **4581.07** | 3.98 | OK |
| 52 | `hgrn2_gated_outer` | **52.0%** | **4510.05** | 5.2 | OK |
| 53 | `hgrn2_outer` | **52.0%** | **4270.03** | 4.87 | OK |
| 54 | `hla_attention` | **50.0%** | **4653.21** | 3.15 | OK |
| 55 | `hybrid_attention` | **46.0%** | **94.34** | 3.57 | OK |
| 56 | `hybrid_local_kernel` | **52.0%** | **2105.57** | 23.94 | OK |
| 57 | `hymba_transx` | **92.0%** | **147.83** | 3.64 | OK |
| 58 | `jamba` | **94.0%** | **31.41** | 4.12 | OK |
| 59 | `jamba_info` | **56.0%** | **1978.0** | 4.06 | OK |
| 60 | `kda_attention` | **51.0%** | **405981.35** | 4.92 | OK |
| 61 | `landmark_local_attention` | **52.0%** | **1242.65** | 24.3 | OK |
| 62 | `linformer_attention` | **52.0%** | **2056.8** | 23.0 | OK |
| 63 | `lola_attention` | **52.0%** | **1292.15** | 11.83 | OK |
| 64 | `lora_attention` | **95.0%** | **29.22** | 2.77 | OK |
| 65 | `lsh_attention` | **52.0%** | **1054.62** | 24.1 | OK |
| 66 | `mamba_block` | **52.0%** | **4270.03** | 15.41 | OK |
| 67 | `mamba_svd_flash` | **94.0%** | **31.05** | 6.12 | OK |
| 68 | `minimal_nova` | **52.0%** | **1242.65** | 17.55 | OK |
| 69 | `minimal_svd` | **52.0%** | **1322.79** | 23.57 | OK |
| 70 | `mixattention_moska` | **90.0%** | **30.45** | 3.07 | OK |
| 71 | `mla_attention` | **52.0%** | **686.26** | 15.14 | OK |
| 72 | `moba_attention` | **94.0%** | **52.82** | 2.67 | OK |
| 73 | `moe_deltakv_paged` | **93.0%** | N/A | 3.79 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 74 | `mqa` | **52.0%** | **998.5** | 2.52 | OK |
| 75 | `naive_svd` | **52.0%** | **1307.38** | 23.42 | OK |
| 76 | `nasc_attention` | **93.0%** | **350.51** | 3.03 | OK |
| 77 | `native_hybrid_attention` | **96.0%** | **4011.32** | 3.28 | OK |
| 78 | `nexus` | **55.0%** | N/A | 2.61 | GPT2 error: mat1 and mat2 must have the same dtype, but got Float and Half |
| 79 | `nsa_attention` | **95.0%** | **33.97** | 3.07 | OK |
| 80 | `nystrom_attention` | **95.0%** | **1277.1** | 3.07 | OK |
| 81 | `optimized_hybrid_attention` | **94.0%** | **6068.93** | 2.87 | OK |
| 82 | `orion` | **52.0%** | **1917.15** | 25.33 | OK |
| 83 | `padre_attention` | **91.0%** | **330.56** | 3.38 | OK |
| 84 | `performer_attention` | **70.0%** | **14221.36** | 3.09 | OK |
| 85 | `phantom` | **52.0%** | **1232.98** | 25.46 | OK |
| 86 | `positive_kernel_linear_attention` | **52.0%** | **834.27** | 23.81 | OK |
| 87 | `primal_attention` | **95.0%** | **31.6** | 21.97 | OK |
| 88 | `progressive_token_pruning` | **94.0%** | **121.13** | 3.02 | OK |
| 89 | `projection_factorized_attn` | **52.0%** | **3595.72** | 13.44 | OK |
| 90 | `qcqa_attention` | **52.0%** | **34.77** | 2.81 | OK |
| 91 | `rala_attention` | **75.0%** | **22199.22** | 4.25 | OK |
| 92 | `reap_moe` | **94.0%** | **30.39** | 3.78 | OK |
| 93 | `refined_landmark_attention` | **52.0%** | **1408.1** | 23.59 | OK |
| 94 | `regla_attention` | **76.0%** | **811.77** | 3.52 | OK |
| 95 | `residual_svd_attention` | **52.0%** | **1282.09** | 18.0 | OK |
| 96 | `routing_attention` | **52.0%** | **1504.79** | 23.85 | OK |
| 97 | `rpc_attention` | **52.0%** | **2189.45** | 23.4 | OK |
| 98 | `salt_attention` | **52.0%** | **747.84** | 59.27 | OK |
| 99 | `samba_m3` | **87.0%** | **126.45** | 3.91 | OK |
| 100 | `samba_moe` | **47.0%** | **662.55** | 5.97 | OK |
| 101 | `selective_ssm` | **52.0%** | **4726.49** | 5.33 | OK |
| 102 | `simple_ssm` | **52.0%** | **4953.32** | 7.13 | OK |
| 103 | `sink_attention` | **95.0%** | **850.73** | 2.85 | OK |
| 104 | `soa_attn` | **52.0%** | **627.29** | 20.16 | OK |
| 105 | `spectraformer_attention` | **49.0%** | **879844.09** | 3.58 | OK |
| 106 | `spectron` | **52.0%** | **3595.72** | 23.45 | OK |
| 107 | `spectron_fp8` | **52.0%** | **2579.8** | 36.41 | OK |
| 108 | `split_head_hybrid` | **52.0%** | N/A | 23.55 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 109 | `svd_compressed` | **52.0%** | **3567.74** | 24.11 | OK |
| 110 | `svd_linear` | **52.0%** | **3567.74** | 14.19 | OK |
| 111 | `svd_linear_attention` | **52.0%** | **3567.74** | 13.12 | OK |
| 112 | `svd_linear_attn` | **52.0%** | **3595.72** | 12.79 | OK |
| 113 | `svd_linear_decay` | **52.0%** | **3595.72** | 13.13 | OK |
| 114 | `svd_linear_inject` | **52.0%** | **3595.72** | 12.79 | OK |
| 115 | `svd_llm_v2_attention` | **52.0%** | **1829.36** | 7.73 | OK |
| 116 | `svd_mamba` | **52.0%** | **3857.65** | 56.32 | OK |
| 117 | `svd_projection_attention` | **52.0%** | N/A | 6.94 | GPT2 error: name 'math' is not defined |
| 118 | `svd_seq` | **95.0%** | **5355.81** | 6.65 | OK |
| 119 | `template_attention` | **94.0%** | **4203.82** | 2.76 | OK |
| 120 | `token_compressed_attention` | **92.0%** | **3918.4** | 2.36 | OK |
| 121 | `tri_branch_hybrid` | **52.0%** | N/A | 23.84 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 122 | `triton_cosformer` | **68.0%** | **22903.9** | 3.08 | OK |
| 123 | `triton_svd_linear_attention` | **52.0%** | **1213.86** | 20.39 | OK |
| 124 | `ttt_linear_attention` | **52.0%** | **6511.01** | 4.6 | OK |
| 125 | `unified_rope` | **52.0%** | **4270.03** | 2.97 | OK |
| 126 | `upi_stuffed_mamba` | **52.0%** | **1327.97** | 5.08 | OK |
| 127 | `vig_attention` | **85.0%** | N/A | 2.73 | GPT2 error: mat1 and mat2 must have the same dtype, but got Float and Half |
| 128 | `vspec` | **52.0%** | **1062.89** | 26.04 | OK |
| 129 | `wgqa_attention` | **52.0%** | **927.07** | 3.33 | OK |
| 130 | `xattention` | **87.0%** | **34.03** | 3.3 | OK |
| 131 | `zeros_attention` | **55.0%** | **87116.36** | 3.48 | OK |
