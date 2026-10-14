# Accuracy & Perplexity Benchmark (`taha` - `NVIDIA GeForce RTX 4070 Laptop GPU`)

- **Baseline RoBERTa SST-2 Accuracy**: **95.00%**
- **Baseline GPT-2 WikiText-2 PPL**: **27.50**

| # | Attention Module | SST-2 Accuracy (%) | WikiText-2 PPL | Latency (s) | Status / Error |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `adaptive_span_attention` | **94.0%** | **11070.82** | 3.12 | OK |
| 2 | `adaptive_sparse_attention` | **92.0%** | **27.54** | 3.75 | OK |
| 3 | `adaptive_svd_attention` | **52.0%** | **1300.04** | 16.45 | OK |
| 4 | `adasplash_attention` | **94.0%** | **67.92** | 1.74 | OK |
| 5 | `aegis_hybrid` | **52.0%** | N/A | 8.54 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 6 | `aegis_spectron_fp8` | **52.0%** | **2328.62** | 12.79 | OK |
| 7 | `aegis_v1` | **52.0%** | **2035.24** | 8.83 | OK |
| 8 | `aegis_v2` | **52.0%** | **2036.91** | 8.17 | OK |
| 9 | `aegis_v3` | **52.0%** | **2026.4** | 8.88 | OK |
| 10 | `aegis_v4` | **52.0%** | **2016.72** | 8.58 | OK |
| 11 | `aegis_v5` | **52.0%** | N/A | 8.2 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 12 | `aegis_v6` | **52.0%** | **1701.72** | 8.55 | OK |
| 13 | `alibi` | **91.0%** | **37074.94** | 1.53 | OK |
| 14 | `asvd_attention` | **79.0%** | **126.6** | 8.68 | OK |
| 15 | `bam_umoe` | **52.0%** | **756.6** | 4.93 | OK |
| 16 | `block_transformer_hiba` | **92.0%** | **140.13** | 2.4 | OK |
| 17 | `chela_slay` | **89.0%** | **46.91** | 1.74 | OK |
| 18 | `chela_slay_attention` | **89.0%** | **45.16** | 1.57 | OK |
| 19 | `cluster_attention` | **94.0%** | **30.3** | 1.47 | OK |
| 20 | `compressed_attention` | **52.0%** | **1359.33** | 4.1 | OK |
| 21 | `compressed_mha_attention` | **52.0%** | **1373.73** | 4.11 | OK |
| 22 | `cosformer` | **95.0%** | **443.94** | 1.62 | OK |
| 23 | `cosformer_attention` | **95.0%** | **541.86** | 1.87 | OK |
| 24 | `cross_attention` | **93.0%** | **27.5** | 1.42 | OK |
| 25 | `decay_linear_attention` | **66.0%** | **18157.22** | 1.92 | OK |
| 26 | `decay_linear_attn` | **66.0%** | **18157.22** | 2.62 | OK |
| 27 | `dynamic_conv_attention` | **52.0%** | **2250704.43** | 1.62 | OK |
| 28 | `eat` | **91.0%** | **31.34** | 1.41 | OK |
| 29 | `eat_attention` | **89.0%** | **29.17** | 1.73 | OK |
| 30 | `expert_choice_moe` | **89.0%** | **32.67** | 1.47 | OK |
| 31 | `factorised_matrix_state` | **70.0%** | **25843.83** | 1.43 | OK |
| 32 | `flash_attention3` | **92.0%** | **31.21** | 1.69 | OK |
| 33 | `flash_attention3_fp8` | **92.0%** | **31.37** | 1.42 | OK |
| 34 | `full` | **95.0%** | **27.5** | 1.11 | OK |
| 35 | `full_attention` | **94.0%** | **32.05** | 1.44 | OK |
| 36 | `fused_svd_kernel` | **52.0%** | **1322.2** | 8.06 | OK |
| 37 | `fusion_attn` | **95.0%** | **71.77** | 1.39 | OK |
| 38 | `galerkin_attention` | **54.0%** | **31.81** | 1.73 | OK |
| 39 | `gated_delta_net` | **52.0%** | **1432.28** | 2.63 | OK |
| 40 | `gated_deltanet_block` | **52.0%** | **1428.54** | 2.69 | OK |
| 41 | `gated_linear_attention` | **66.0%** | **18157.22** | 1.94 | OK |
| 42 | `gated_linear_attn` | **66.0%** | **18157.22** | 1.89 | OK |
| 43 | `gla_attention` | **56.0%** | **10475.47** | 1.43 | OK |
| 44 | `goldfinch_ahn` | **95.0%** | **10491.3** | 1.65 | OK |
| 45 | `gqa_attention` | **95.0%** | **30.66** | 1.53 | OK |
| 46 | `granite_hybrid` | N/A | **5035.83** | 3.91 | RoBERTa error: CUDA out of memory. Tried to allocate 516.00 MiB. GPU 0 has a total capacity of 7.62 GiB of which 86.81 MiB is free. Including non-PyTorch memory, this process has 2.57 GiB memory in use. Process 725928 has 4.94 GiB memory in use. Of the allocated memory 1.73 GiB is allocated by PyTorch, and 250.75 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf);  |
| 47 | `griffin_ssd` | **91.0%** | **127.52** | 3.3 | OK |
| 48 | `halo_regla` | **92.0%** | **4150.02** | 3.17 | OK |
| 49 | `hamming_attention` | **62.0%** | **4135.06** | 3.62 | OK |
| 50 | `hgrn2_gated_deltanet` | **92.0%** | **4581.09** | 7.93 | OK |
| 51 | `hgrn2_gated_layer` | **92.0%** | **4597.07** | 4.56 | OK |
| 52 | `hgrn2_gated_outer` | N/A | **4372.25** | 3.22 | RoBERTa error: CUDA out of memory. Tried to allocate 300.00 MiB. GPU 0 has a total capacity of 7.62 GiB of which 142.81 MiB is free. Including non-PyTorch memory, this process has 2.52 GiB memory in use. Process 725928 has 4.94 GiB memory in use. Of the allocated memory 1.75 GiB is allocated by PyTorch, and 176.46 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf);  |
| 53 | `hgrn2_outer` | N/A | **4409.86** | 2.86 | RoBERTa error: CUDA out of memory. Tried to allocate 300.00 MiB. GPU 0 has a total capacity of 7.62 GiB of which 54.81 MiB is free. Including non-PyTorch memory, this process has 2.61 GiB memory in use. Process 725928 has 4.94 GiB memory in use. Of the allocated memory 1.75 GiB is allocated by PyTorch, and 264.04 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf);  |
| 54 | `hla_attention` | **50.0%** | **4669.51** | 3.0 | OK |
| 55 | `hybrid_attention` | **52.0%** | **100.82** | 3.34 | OK |
| 56 | `hybrid_local_kernel` | **52.0%** | **2069.46** | 21.83 | OK |
| 57 | `hymba_transx` | **93.0%** | **140.64** | 5.34 | OK |
| 58 | `jamba` | **94.0%** | **29.9** | 2.41 | OK |
| 59 | `jamba_info` | **55.0%** | **2098.35** | 1.88 | OK |
| 60 | `kda_attention` | **51.0%** | **nan** | 2.39 | OK |
| 61 | `landmark_local_attention` | **52.0%** | **1196.81** | 8.59 | OK |
| 62 | `linformer_attention` | **52.0%** | **1712.01** | 9.23 | OK |
| 63 | `lola_attention` | **52.0%** | **1291.54** | 10.01 | OK |
| 64 | `lora_attention` | **94.0%** | **30.71** | 1.9 | OK |
| 65 | `lsh_attention` | **52.0%** | **1072.44** | 9.02 | OK |
| 66 | `mamba_block` | **52.0%** | **3959.32** | 5.79 | OK |
| 67 | `mamba_svd_flash` | **95.0%** | **30.35** | 3.84 | OK |
| 68 | `minimal_nova` | **52.0%** | **1295.42** | 6.66 | OK |
| 69 | `minimal_svd` | **52.0%** | **1258.71** | 8.62 | OK |
| 70 | `mixattention_moska` | **90.0%** | **31.3** | 1.92 | OK |
| 71 | `mla_attention` | **52.0%** | **698.78** | 5.66 | OK |
| 72 | `moba_attention` | **95.0%** | **49.14** | 1.56 | OK |
| 73 | `moe_deltakv_paged` | **91.0%** | N/A | 2.81 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 74 | `mqa` | **52.0%** | **992.49** | 1.47 | OK |
| 75 | `naive_svd` | **52.0%** | **1257.15** | 8.31 | OK |
| 76 | `nasc_attention` | **92.0%** | **339.55** | 3.06 | OK |
| 77 | `native_hybrid_attention` | **94.0%** | **5095.43** | 1.93 | OK |
| 78 | `nexus` | **55.0%** | N/A | 1.35 | GPT2 error: mat1 and mat2 must have the same dtype, but got Float and Half |
| 79 | `nsa_attention` | **95.0%** | **36.54** | 1.89 | OK |
| 80 | `nystrom_attention` | **93.0%** | **1171.66** | 1.87 | OK |
| 81 | `optimized_hybrid_attention` | **94.0%** | **6425.2** | 3.08 | OK |
| 82 | `orion` | **52.0%** | **1898.29** | 9.43 | OK |
| 83 | `padre_attention` | **93.0%** | **344.29** | 1.84 | OK |
| 84 | `performer_attention` | **75.0%** | **16604.77** | 1.59 | OK |
| 85 | `phantom` | **52.0%** | **1232.49** | 9.44 | OK |
| 86 | `positive_kernel_linear_attention` | **52.0%** | **834.03** | 10.55 | OK |
| 87 | `primal_attention` | **94.0%** | **29.6** | 16.17 | OK |
| 88 | `progressive_token_pruning` | **94.0%** | **121.75** | 1.54 | OK |
| 89 | `projection_factorized_attn` | **52.0%** | **3588.41** | 6.21 | OK |
| 90 | `qcqa_attention` | **52.0%** | **32.78** | 1.88 | OK |
| 91 | `rala_attention` | **75.0%** | **22067.47** | 2.06 | OK |
| 92 | `reap_moe` | **93.0%** | **32.68** | 2.8 | OK |
| 93 | `refined_landmark_attention` | **52.0%** | **2122.52** | 8.97 | OK |
| 94 | `regla_attention` | **77.0%** | **833.54** | 1.65 | OK |
| 95 | `residual_svd_attention` | **52.0%** | **1286.72** | 9.75 | OK |
| 96 | `routing_attention` | **52.0%** | **1444.23** | 9.41 | OK |
| 97 | `rpc_attention` | **52.0%** | **2184.07** | 10.2 | OK |
| 98 | `salt_attention` | **52.0%** | **782.72** | 27.89 | OK |
| 99 | `samba_m3` | **83.0%** | **106.77** | 2.08 | OK |
| 100 | `samba_moe` | **90.0%** | **685.83** | 3.16 | OK |
| 101 | `selective_ssm` | **52.0%** | **5432.31** | 2.47 | OK |
| 102 | `simple_ssm` | **52.0%** | **5406.81** | 2.48 | OK |
| 103 | `sink_attention` | **96.0%** | **807.45** | 1.94 | OK |
| 104 | `soa_attn` | **52.0%** | **568.8** | 7.22 | OK |
| 105 | `spectraformer_attention` | **49.0%** | **880608.83** | 8.09 | OK |
| 106 | `spectron` | **52.0%** | **3609.21** | 8.25 | OK |
| 107 | `spectron_fp8` | **52.0%** | **2403.1** | 14.1 | OK |
| 108 | `split_head_hybrid` | **52.0%** | N/A | 8.5 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 109 | `svd_compressed` | **52.0%** | **3588.72** | 8.91 | OK |
| 110 | `svd_linear` | **52.0%** | **3565.18** | 7.62 | OK |
| 111 | `svd_linear_attention` | **52.0%** | **3610.76** | 5.84 | OK |
| 112 | `svd_linear_attn` | **52.0%** | **3592.59** | 4.67 | OK |
| 113 | `svd_linear_decay` | **52.0%** | **3606.8** | 5.03 | OK |
| 114 | `svd_linear_inject` | **52.0%** | **3603.2** | 5.03 | OK |
| 115 | `svd_llm_v2_attention` | **52.0%** | **1832.08** | 3.49 | OK |
| 116 | `svd_mamba` | **52.0%** | **3846.3** | 22.94 | OK |
| 117 | `svd_projection_attention` | **52.0%** | N/A | 7.17 | GPT2 error: name 'math' is not defined |
| 118 | `svd_seq` | **96.0%** | **7910.01** | 3.1 | OK |
| 119 | `template_attention` | **95.0%** | **4152.16** | 2.42 | OK |
| 120 | `token_compressed_attention` | **94.0%** | **3940.38** | 3.03 | OK |
| 121 | `tri_branch_hybrid` | **52.0%** | N/A | 8.67 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 122 | `triton_cosformer` | **68.0%** | **22770.5** | 2.99 | OK |
| 123 | `triton_svd_linear_attention` | **52.0%** | **1213.26** | 11.62 | OK |
| 124 | `ttt_linear_attention` | **52.0%** | **6529.27** | 12.17 | OK |
| 125 | `unified_rope` | **92.0%** | **4218.74** | 2.05 | OK |
| 126 | `upi_stuffed_mamba` | N/A | **1328.82** | 2.02 | RoBERTa error: CUDA out of memory. Tried to allocate 258.00 MiB. GPU 0 has a total capacity of 7.62 GiB of which 240.81 MiB is free. Including non-PyTorch memory, this process has 3.66 GiB memory in use. Process 733153 has 3.70 GiB memory in use. Of the allocated memory 2.66 GiB is allocated by PyTorch, and 410.98 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf);  |
| 127 | `vig_attention` | **86.0%** | N/A | 2.49 | GPT2 error: mat1 and mat2 must have the same dtype, but got Float and Half |
| 128 | `vspec` | **52.0%** | **1084.77** | 16.09 | OK |
| 129 | `wgqa_attention` | **52.0%** | **929.8** | 3.27 | OK |
| 130 | `xattention` | **87.0%** | **29.78** | 1.85 | OK |
| 131 | `zeros_attention` | **55.0%** | **82506.78** | 2.12 | OK |
