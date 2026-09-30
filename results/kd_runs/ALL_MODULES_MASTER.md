# All Modules Knowledge Distillation (KD) Master Results

**Last Updated:** 2026-09-30 21:56:55  
**Teacher Baseline PPL (GPT-2 on WikiText-2):** `31.54`  

## Summary
- **Total Modules Available:** 113
- **Total Successfully Tested:** 27
- **Failed / Incompatible / OOM:** 2

## Top 10 by Post-KD Perplexity

| Rank | Module | Pre PPL | Post PPL | Ratio to Baseline (31.54) | Recovery % | Step Time | Peak VRAM |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `lora_attention` | 31.54 | **31.52** | 1.00x | 0.00% | 541.7ms | 1460.8 MB |
| 2 | `adaptive_sparse_attention` | 31.57 | **31.57** | 1.00x | 0.00% | 576.0ms | 1216.1 MB |
| 3 | `galerkin` | 31.54 | **31.65** | 1.00x | 0.00% | 392.5ms | 1452.8 MB |
| 4 | `gqa_attention` | 31.54 | **31.65** | 1.00x | 0.00% | 500.2ms | 1474.1 MB |
| 5 | `fusion_attn` | 31.54 | **31.67** | 1.00x | 0.00% | 338.6ms | 1453.6 MB |
| 6 | `squeezing_heads_distill` | 31.54 | **31.73** | 1.01x | 0.00% | 343.0ms | 1453.9 MB |
| 7 | `flash_attention3` | 31.54 | **31.74** | 1.01x | 0.00% | 491.8ms | 1443.8 MB |
| 8 | `granite_hybrid` | 31.54 | **31.74** | 1.01x | 0.00% | 253.9ms | 1420.1 MB |
| 9 | `adaptive_span_attention` | 31.54 | **31.77** | 1.01x | 0.00% | 550.8ms | 1497.0 MB |
| 10 | `squeezing_heads` | 31.54 | **31.83** | 1.01x | 0.00% | 451.0ms | 1453.9 MB |

## Full Results Table

| Rank | Module | Pre PPL | Post PPL | Recovery % | Ratio | Step Time | Peak VRAM | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `lora_attention` | 31.54 | 31.52 | 0.00% | 1.00x | 541.7ms | 1460.8 MB | PASS |
| 2 | `adaptive_sparse_attention` | 31.57 | 31.57 | 0.00% | 1.00x | 576.0ms | 1216.1 MB | PASS |
| 3 | `galerkin` | 31.54 | 31.65 | 0.00% | 1.00x | 392.5ms | 1452.8 MB | PASS |
| 4 | `gqa_attention` | 31.54 | 31.65 | 0.00% | 1.00x | 500.2ms | 1474.1 MB | PASS |
| 5 | `fusion_attn` | 31.54 | 31.67 | 0.00% | 1.00x | 338.6ms | 1453.6 MB | PASS |
| 6 | `squeezing_heads_distill` | 31.54 | 31.73 | 0.00% | 1.01x | 343.0ms | 1453.9 MB | PASS |
| 7 | `flash_attention3` | 31.54 | 31.74 | 0.00% | 1.01x | 491.8ms | 1443.8 MB | PASS |
| 8 | `granite_hybrid` | 31.54 | 31.74 | 0.00% | 1.01x | 253.9ms | 1420.1 MB | PASS |
| 9 | `adaptive_span_attention` | 31.54 | 31.77 | 0.00% | 1.01x | 550.8ms | 1497.0 MB | PASS |
| 10 | `squeezing_heads` | 31.54 | 31.83 | 0.00% | 1.01x | 451.0ms | 1453.9 MB | PASS |
| 11 | `primal_attention` | 31.70 | 32.01 | -193.75% | 1.01x | 1298.8ms | 1482.5 MB | PASS |
| 12 | `mamba_svd_flash` | 31.54 | 32.03 | 0.00% | 1.02x | 273.3ms | 1418.9 MB | PASS |
| 13 | `asvd_attention` | 39.38 | 34.82 | 58.16% | 1.10x | 488.0ms | 1161.7 MB | PASS |
| 14 | `xattention` | 47.62 | 36.10 | 71.64% | 1.14x | 558.7ms | 1460.5 MB | PASS |
| 15 | `aa_svd` | 47.22 | 40.11 | 45.34% | 1.27x | 244.7ms | 1334.2 MB | PASS |
| 16 | `compressed_attention` | 165.59 | 43.20 | 91.30% | 1.37x | 389.2ms | 1255.3 MB | PASS |
| 17 | `factorised_matrix_state` | 3867.96 | 114.52 | 97.84% | 3.63x | 419.1ms | 1754.5 MB | PASS |
| 18 | `res_svd` | 277.40 | 115.30 | 65.93% | 3.66x | 409.1ms | 1149.0 MB | PASS |
| 19 | `svd_llm_v2_attention` | 339.46 | 120.41 | 71.14% | 3.82x | 526.4ms | 1132.1 MB | PASS |
| 20 | `leka_v2` | 388.19 | 140.60 | 69.42% | 4.46x | 975.9ms | 1267.2 MB | PASS |
| 21 | `aegis_v6` | 156.83 | 156.83 | 0.00% | 4.97x | 953.8ms | 1469.6 MB | PASS |
| 22 | `svd_linear_attention` | 339.37 | 173.41 | 53.91% | 5.50x | 409.9ms | 1063.2 MB | PASS |
| 23 | `adaptive_svd_attention` | 339.43 | 184.52 | 50.31% | 5.85x | 364.4ms | 1077.2 MB | PASS |
| 24 | `kda_attention` | 2796.78 | 189.05 | 94.30% | 5.99x | 492.2ms | 1501.0 MB | PASS |
| 25 | `leka_v3` | 711.96 | 261.47 | 66.21% | 8.29x | 792.0ms | 1235.3 MB | PASS |
| 26 | `fused_svd_kernel` | 339.36 | 281.41 | 18.83% | 8.92x | 5790.3ms | 1053.0 MB | PASS |
| 27 | `hgrn2_outer` | 12352.75 | 1014.88 | 92.02% | 32.18x | 1066.3ms | 1927.3 MB | PASS |

## Failed / Incompatible Modules

| Module | Failure Reason |
|:---|:---|
| `triton_cosformer` | `not enough values to unpack (expected 3, got 0)` |
| `cosformer` | `Unknown error or OOM` |
