# Accuracy & Perplexity Benchmark (`tahoora` - `NVIDIA GeForce GTX 1060`)

- **Baseline RoBERTa SST-2 Accuracy**: **95.00%**
- **Baseline GPT-2 WikiText-2 PPL**: **27.56**

| # | Attention Module | SST-2 Accuracy (%) | WikiText-2 PPL | Latency (s) | Status / Error |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `mamba_svd_flash` | **95.0%** | **32.47** | 7.06 | OK |
| 2 | `minimal_nova` | **52.0%** | **1272.12** | 17.43 | OK |
| 3 | `minimal_svd` | **52.0%** | **1370.12** | 23.01 | OK |
| 4 | `mixattention_moska` | **91.0%** | **31.85** | 2.64 | OK |
| 5 | `mla_attention` | **52.0%** | **667.74** | 15.3 | OK |
| 6 | `moba_attention` | **94.0%** | **44.74** | 3.58 | OK |
| 7 | `moe_deltakv_paged` | **92.0%** | N/A | 3.26 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 8 | `mqa` | **52.0%** | **986.86** | 2.73 | OK |
| 9 | `naive_svd` | **52.0%** | **1262.22** | 23.36 | OK |
| 10 | `nasc_attention` | **92.0%** | **321.64** | 2.88 | OK |
| 11 | `native_hybrid_attention` | **95.0%** | **5791.01** | 3.24 | OK |
| 12 | `nexus` | **56.0%** | N/A | 2.92 | GPT2 error: mat1 and mat2 must have the same dtype, but got Float and Half |
| 13 | `nsa_attention` | **95.0%** | **33.31** | 2.7 | OK |
| 14 | `nystrom_attention` | **94.0%** | **1209.13** | 2.93 | OK |
| 15 | `optimized_hybrid_attention` | **95.0%** | **7040.08** | 2.84 | OK |
| 16 | `orion` | **52.0%** | **1894.81** | 25.31 | OK |
| 17 | `padre_attention` | **92.0%** | **358.82** | 2.63 | OK |
| 18 | `performer_attention` | **74.0%** | **14221.36** | 2.67 | OK |
| 19 | `phantom` | **52.0%** | **1232.98** | 25.36 | OK |
| 20 | `positive_kernel_linear_attention` | **52.0%** | **834.27** | 23.93 | OK |
| 21 | `primal_attention` | **94.0%** | **29.92** | 22.71 | OK |
| 22 | `progressive_token_pruning` | **94.0%** | **132.52** | 2.58 | OK |
| 23 | `projection_factorized_attn` | **52.0%** | **3595.72** | 13.03 | OK |
| 24 | `qcqa_attention` | **52.0%** | **33.83** | 3.13 | OK |
| 25 | `rala_attention` | **75.0%** | **22199.22** | 4.12 | OK |
| 26 | `reap_moe` | **95.0%** | **33.25** | 3.68 | OK |
| 27 | `refined_landmark_attention` | **52.0%** | **1800.99** | 24.22 | OK |
| 28 | `regla_attention` | **73.0%** | **941.67** | 2.43 | OK |
| 29 | `residual_svd_attention` | **52.0%** | **1277.1** | 17.52 | OK |
| 30 | `routing_attention` | **52.0%** | **1408.1** | 24.14 | OK |
| 31 | `rpc_attention` | **52.0%** | **2250.15** | 23.73 | OK |
| 32 | `salt_attention` | **52.0%** | **747.84** | 59.67 | OK |
| 33 | `samba_m3` | **85.0%** | **111.16** | 3.78 | OK |
| 34 | `samba_moe` | **57.0%** | **622.41** | 5.97 | OK |
| 35 | `selective_ssm` | **52.0%** | **5314.13** | 5.93 | OK |
| 36 | `simple_ssm` | **52.0%** | **5612.84** | 6.37 | OK |
| 37 | `sink_attention` | **94.0%** | **840.82** | 3.75 | OK |
| 38 | `soa_attn` | **52.0%** | **591.59** | 20.98 | OK |
| 39 | `spectraformer_attention` | **49.0%** | **879844.09** | 2.98 | OK |
| 40 | `spectron` | **52.0%** | **3595.72** | 23.65 | OK |
| 41 | `spectron_fp8` | **52.0%** | **2579.8** | 35.93 | OK |
| 42 | `split_head_hybrid` | **52.0%** | N/A | 24.34 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 43 | `svd_compressed` | **52.0%** | **3567.74** | 23.24 | OK |
| 44 | `svd_linear` | **52.0%** | **3567.74** | 13.79 | OK |
| 45 | `svd_linear_attention` | **52.0%** | **3567.74** | 14.76 | OK |
| 46 | `svd_linear_attn` | **52.0%** | **3595.72** | 13.48 | OK |
| 47 | `svd_linear_decay` | **52.0%** | **3595.72** | 13.89 | OK |
| 48 | `svd_linear_inject` | **52.0%** | **3595.72** | 12.72 | OK |
| 49 | `svd_llm_v2_attention` | **52.0%** | **1829.36** | 7.99 | OK |
| 50 | `svd_mamba` | **52.0%** | **3857.65** | 55.03 | OK |
| 51 | `svd_projection_attention` | **52.0%** | N/A | 6.67 | GPT2 error: name 'math' is not defined |
| 52 | `svd_seq` | **95.0%** | **5355.81** | 5.82 | OK |
| 53 | `template_attention` | **94.0%** | **4203.82** | 2.96 | OK |
| 54 | `token_compressed_attention` | **92.0%** | **3918.4** | 2.6 | OK |
| 55 | `tri_branch_hybrid` | **52.0%** | N/A | 23.61 | GPT2 error: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half |
| 56 | `triton_cosformer` | **68.0%** | **22903.9** | 3.12 | OK |
| 57 | `triton_svd_linear_attention` | **52.0%** | **1213.86** | 20.72 | OK |
| 58 | `ttt_linear_attention` | **52.0%** | **6511.01** | 4.52 | OK |
| 59 | `unified_rope` | **52.0%** | **4270.03** | 2.73 | OK |
| 60 | `upi_stuffed_mamba` | **52.0%** | **1327.97** | 5.29 | OK |
| 61 | `vig_attention` | **85.0%** | N/A | 3.04 | GPT2 error: mat1 and mat2 must have the same dtype, but got Float and Half |
| 62 | `vspec` | **52.0%** | **1062.89** | 25.78 | OK |
| 63 | `wgqa_attention` | **52.0%** | **927.07** | 2.37 | OK |
| 64 | `xattention` | **87.0%** | **34.03** | 3.14 | OK |
| 65 | `zeros_attention` | **55.0%** | **87116.36** | 2.96 | OK |
