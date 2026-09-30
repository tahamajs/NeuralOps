# All Modules Knowledge Distillation (KD) Master Results

**Last Updated:** 2026-09-30 18:00:44  
**Teacher Baseline PPL (GPT-2 on WikiText-2):** `31.54`  

## Summary
- **Total Modules Available:** 113
- **Total Successfully Tested:** 8
- **Failed / Incompatible / OOM:** 2

## Top 10 by Post-KD Perplexity

| Rank | Module | Pre PPL | Post PPL | Ratio to Baseline (31.54) | Recovery % | Step Time | Peak VRAM |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `galerkin` | 31.54 | **31.65** | 1.00x | 0.00% | 392.5ms | 1452.8 MB |
| 2 | `xattention` | 47.62 | **36.10** | 1.14x | 71.64% | 558.7ms | 1460.5 MB |
| 3 | `compressed_attention` | 165.59 | **43.20** | 1.37x | 91.30% | 389.2ms | 1255.3 MB |
| 4 | `factorised_matrix_state` | 3867.96 | **114.52** | 3.63x | 97.84% | 419.1ms | 1754.5 MB |
| 5 | `svd_llm_v2_attention` | 339.46 | **120.41** | 3.82x | 71.14% | 526.4ms | 1132.1 MB |
| 6 | `leka_v2` | 388.19 | **140.60** | 4.46x | 69.42% | 975.9ms | 1267.2 MB |
| 7 | `svd_linear_attention` | 339.37 | **173.41** | 5.50x | 53.91% | 409.9ms | 1063.2 MB |
| 8 | `leka_v3` | 711.96 | **261.47** | 8.29x | 66.21% | 792.0ms | 1235.3 MB |

## Full Results Table

| Rank | Module | Pre PPL | Post PPL | Recovery % | Ratio | Step Time | Peak VRAM | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `galerkin` | 31.54 | 31.65 | 0.00% | 1.00x | 392.5ms | 1452.8 MB | PASS |
| 2 | `xattention` | 47.62 | 36.10 | 71.64% | 1.14x | 558.7ms | 1460.5 MB | PASS |
| 3 | `compressed_attention` | 165.59 | 43.20 | 91.30% | 1.37x | 389.2ms | 1255.3 MB | PASS |
| 4 | `factorised_matrix_state` | 3867.96 | 114.52 | 97.84% | 3.63x | 419.1ms | 1754.5 MB | PASS |
| 5 | `svd_llm_v2_attention` | 339.46 | 120.41 | 71.14% | 3.82x | 526.4ms | 1132.1 MB | PASS |
| 6 | `leka_v2` | 388.19 | 140.60 | 69.42% | 4.46x | 975.9ms | 1267.2 MB | PASS |
| 7 | `svd_linear_attention` | 339.37 | 173.41 | 53.91% | 5.50x | 409.9ms | 1063.2 MB | PASS |
| 8 | `leka_v3` | 711.96 | 261.47 | 66.21% | 8.29x | 792.0ms | 1235.3 MB | PASS |

## Failed / Incompatible Modules

| Module | Failure Reason |
|:---|:---|
| `triton_cosformer` | `not enough values to unpack (expected 3, got 0)` |
| `cosformer` | `Unknown error or OOM` |
