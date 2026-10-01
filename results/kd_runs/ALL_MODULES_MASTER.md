# All Modules KD Results

**Generated:** 2026-10-01T14:12:53.498823  
**Total modules tested:** 52  
**Baseline (GPT-2 Teacher):** PPL = `31.54`

## Results Table

| Rank | Module | Pre PPL | Post PPL | Recovery | Ratio | Status |
|------|--------|---------|----------|----------|-------|--------|
| 1 | `lora_attention` | 31.54 | **31.52** | 0.0% | 1.00x | PASS (Excellent) |
| 2 | `adaptive_sparse_attention` | 31.57 | **31.57** | 0.0% | 1.00x | PASS (Excellent) |
| 3 | `galerkin` | 31.54 | **31.65** | 0.0% | 1.00x | PASS (Excellent) |
| 4 | `gqa_attention` | 31.54 | **31.65** | 0.0% | 1.00x | PASS (Excellent) |
| 5 | `fusion_attn` | 31.54 | **31.67** | 0.0% | 1.00x | PASS (Excellent) |
| 6 | `expert_choice_moe` | 31.54 | **31.71** | 0.0% | 1.01x | PASS (Excellent) |
| 7 | `squeezing_heads_distill` | 31.54 | **31.73** | 0.0% | 1.01x | PASS (Excellent) |
| 8 | `flash_attention3` | 31.54 | **31.74** | 0.0% | 1.01x | PASS (Excellent) |
| 9 | `granite_hybrid` | 31.54 | **31.74** | 0.0% | 1.01x | PASS (Excellent) |
| 10 | `block_transformer_hiba` | 31.54 | **31.76** | 0.0% | 1.01x | PASS (Excellent) |
| 11 | `adaptive_span_attention` | 31.54 | **31.77** | 0.0% | 1.01x | PASS (Excellent) |
| 12 | `adasplash_attention` | 32.28 | **31.8** | 64.86% | 1.01x | PASS (Excellent) |
| 13 | `full_attention` | 31.54 | **31.8** | 0.0% | 1.01x | PASS (Excellent) |
| 14 | `flash_attention3_fp8` | 31.54 | **31.81** | 0.0% | 1.01x | PASS (Excellent) |
| 15 | `squeezing_heads` | 31.54 | **31.83** | 0.0% | 1.01x | PASS (Excellent) |
| 16 | `primal_attention` | 31.7 | **32.01** | -193.75% | 1.01x | PASS (Excellent) |
| 17 | `mamba_svd_flash` | 31.54 | **32.03** | 0.0% | 1.02x | PASS (Excellent) |
| 18 | `eat` | 34.01 | **34.55** | 0.0% | 1.10x | PASS (Excellent) |
| 19 | `asvd_attention` | 39.38 | **34.82** | 58.16% | 1.10x | PASS (Excellent) |
| 20 | `xattention` | 47.62 | **36.1** | 71.64% | 1.14x | PASS (Excellent) |
| 21 | `griffin_ssd` | 413.3 | **37.86** | 98.34% | 1.20x | PASS (Excellent) |
| 22 | `aa_svd` | 47.22 | **40.11** | 45.34% | 1.27x | PASS (Excellent) |
| 23 | `compressed_attention` | 165.59 | **43.2** | 91.3% | 1.37x | PASS (Excellent) |
| 24 | `svd_mamba` | 74.58 | **47.11** | 63.82% | 1.49x | PASS (Excellent) |
| 25 | `kda_attention` | 2796.78 | **54.56** | 99.17% | 1.73x | PASS (Good) |
| 26 | `hamming_attention` | 648.92 | **83.59** | 91.57% | 2.65x | PASS (Good) |
| 27 | `aegis_spectron_fp8` | 407.77 | **83.79** | 86.11% | 2.66x | PASS (Good) |
| 28 | `bam_umoe` | 173.0 | **92.9** | 56.62% | 2.95x | PASS (Good) |
| 29 | `aegis_v5` | 211.86 | **98.77** | 62.72% | 3.13x | PASS (Good) |
| 30 | `factorised_matrix_state` | 3867.96 | **114.52** | 97.84% | 3.63x | PASS (Good) |
| 31 | `res_svd` | 277.4 | **115.3** | 65.93% | 3.66x | PASS (Good) |
| 32 | `svd_llm_v2_attention` | 339.46 | **120.41** | 71.14% | 3.82x | PASS (Good) |
| 33 | `decay_linear_attention` | 2492.37 | **131.84** | 95.92% | 4.18x | PASS (Good) |
| 34 | `leka_v2` | 388.19 | **140.6** | 69.42% | 4.46x | PASS (Good) |
| 35 | `aegis_v6` | 156.83 | **156.83** | 0.0% | 4.97x | PASS |
| 36 | `svd_linear_attention` | 339.37 | **173.41** | 53.91% | 5.50x | PASS |
| 37 | `compressed_mha_attention` | 601.87 | **176.91** | 74.51% | 5.61x | PASS |
| 38 | `gated_delta_net` | 784.04 | **177.42** | 80.61% | 5.63x | PASS |
| 39 | `compressed_attention_layer_energy` | 601.81 | **179.38** | 74.08% | 5.69x | PASS |
| 40 | `adaptive_svd_attention` | 339.43 | **184.52** | 50.31% | 5.85x | PASS |
| 41 | `aegis_v6_attn` | 212.2 | **212.2** | 0.0% | 6.73x | PASS |
| 42 | `triton_svd_linear_attention` | 489.13 | **226.84** | 57.32% | 7.19x | PASS |
| 43 | `orion` | 451.57 | **250.08** | 47.97% | 7.93x | PASS |
| 44 | `leka_v3` | 711.96 | **261.47** | 66.21% | 8.29x | PASS |
| 45 | `fused_svd_kernel` | 339.36 | **281.41** | 18.83% | 8.92x | PASS |
| 46 | `aegis_hybrid` | 371.97 | **431.19** | -17.4% | 13.67x | PASS |
| 47 | `gated_deltanet_block` | 1332.05 | **454.39** | 67.61% | 14.41x | PASS |
| 48 | `hgrn2_outer` | 12352.75 | **1014.88** | 92.02% | 32.18x | PASS |
| 49 | `hybrid_attention` | 31.54 | **1198.81** | 0.0% | 38.01x | PASS |
| 50 | `chela_slay` | 3368.59 | **1346.73** | 60.59% | 42.70x | PASS |
| 51 | `hla_attention` | 3229.08 | **1750.78** | 46.23% | 55.51x | PASS |
| 52 | `cosformer` | 39435.26 | **5058.89** | 87.24% | 160.40x | PASS |