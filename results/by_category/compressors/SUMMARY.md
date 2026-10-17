# 📊 Comprehensive Master Registry of Evaluated Compression Engines

**Category Filter:** `compressor` | **Total Verified Records:** 101

## 1. Executive Summary & Compression Frontier Analysis

The compression registry tracks **101 structured pruning, low-rank factorization (SVD/HNC-SVD), Kronecker, Tensor-Train, and Tucker decomposition operators**. Hardware benchmarks focus on parameter reduction, FLOP savings, relative Frobenius reconstruction error, and zero-shot recovery capabilities.

### Key Pareto Frontiers & Top Performers:
- **Highest Compression Ratios**: `frandi_compressor` (2.77x), `hnc_svd` (2.00x), `swift_svd` (1.44x), `aa_svd_compressor` (1.22x), `adasvd_compressor` (1.20x), `adaptive_ranks` (1.17x)
- **Accuracy Retainers (SST-2 >= 88.0%)**: `delta_kv` (95.00%), `flap` (94.70%), `nha_compressor` (94.61%), `svd_llm_compressor` (94.50%), `cr_compressor` (94.38%), `cur_compressor` (94.38%)
- **Lowest Latency Compression Engines**: `hnc_svd` (0.740 ms), `kernel` (10.090 ms), `adaptive_span` (27.467 ms), `adaptive_svd` (30.854 ms), `enlighten_compressor` (470.000 ms)

---

## 2. Pareto Rankings (Compression Ratio vs. Accuracy Retention)

| Pareto Rank | Module Name | Family | Compression Ratio | SST-2 (%) | MNLI (%) | Triton Lat (ms) | Peak VRAM (MB) | Status |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1** | `frandi_compressor` | gpt | **2.77x** | 93.81% | -% | 24360.000 | 320.5 | VERIFIED (RTX 4070) |
| **#2** | `hnc_svd` | hnc-svd | **2.00x** | 93.40% | -% | 0.740 | - | VERIFIED (RTX 4070) |
| **#3** | `swift_svd` | matrix-compression | **1.44x** | 70.00% | -% | - | 297.8 | VERIFIED (RTX 4070) |
| **#4** | `aa_svd_compressor` | gpt | **1.22x** | 50.00% | -% | 504280.000 | - | VERIFIED (RTX 4070) |
| **#5** | `adasvd_compressor` | bert | **1.20x** | 55.00% | -% | 480.000 | 3516.2 | VERIFIED (RTX 4070) |
| **#6** | `adaptive_ranks` | matrix-compression | **1.17x** | 82.11% | -% | - | - | VERIFIED (RTX 4070) |
| **#7** | `saes_svd_compressor` | bert | **1.17x** | 93.92% | -% | 8510.000 | 302.9 | PASS (CPU/CI) |
| **#8** | `adaptive_svd` | unified-4models | **1.15x** | 92.30% | -% | 30.854 | 311.8 | PASS (CPU/CI) |
| **#9** | `compot_compressor` | bert | **1.12x** | 94.04% | -% | 8160.000 | 249.7 | VERIFIED (RTX 4070) |
| **#10** | `adaptive_compressor` | bert | **1.06x** | -% | 43.20% | 6710.000 | 249.7 | PASS (CPU/CI) |
| **#11** | `adaptive_compressor_factory` | gpt | **1.06x** | -% | -% | 259370.000 | 1093.3 | PASS (CPU/CI) |
| **#12** | `frandi_compressor_class` | bert | **1.06x** | 90.60% | 37.80% | 5540.000 | 304.1 | PASS (CPU/CI) |
| **#13** | `tt_compressor` | gpt2 | **1.05x** | 50.92% | -% | 42090.000 | 598.3 | VERIFIED (RTX 4070) |
| **#14** | `tt_compressor_factory` | gpt | **1.05x** | -% | -% | 272950.000 | 816.2 | PASS (CPU/CI) |
| **#15** | `tt_linear_compressor` | gpt | **1.05x** | -% | -% | 310690.000 | 816.2 | PASS (CPU/CI) |

---

## 3. High-Precision Master Benchmark Table

| Module Name | Family | Params | CR | SST-2 (%) | MNLI (%) | AG News (%) | WT-2 PPL | Triton (ms) | Speedup | VRAM (MB) | Throughput (tok/s) | Git SHA |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `a3_compressor` | bert | - | 1.01x | 50.00 | 43.00 | 80.50 | - | 12030.000 | - | 312.9 | 159,084.9 | `2f366e4546` |
| `aa_svd_compressor` | gpt | - | 1.22x | 50.00 | - | - | - | 504280.000 | - | - | 0.0 | `6267bfb8a8` |
| `aatc_xkv_compressor` | matrix-compression | - | 1.00x | 55.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `adaptive_compressor` | bert | - | 1.06x | - | 43.20 | 83.60 | - | 6710.000 | - | 249.7 | 251,728.5 | `2f366e4546` |
| `adaptive_compressor_factory` | gpt | - | 1.06x | - | - | - | - | 259370.000 | - | 1093.3 | 1,065.6 | `6267bfb8a8` |
| `adaptive_ranks` | matrix-compression | - | 1.17x | 82.11 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `adaptive_span` | unified-4models | - | - | 82.91 | - | - | - | 27.467 | - | - | - | `b3e2b3ad08` |
| `adaptive_svd` | unified-4models | 58,109,188 | 1.15x | 92.30 | - | - | - | 30.854 | - | 311.8 | 44,684.4 | `b3e2b3ad08` |
| `adasvd_compressor` | bert | - | 1.20x | 55.00 | - | 0.00 | - | 480.000 | - | 3516.2 | 12,956.4 | `d156d2e8d5` |
| `admm_compressor` | gpt | - | 1.00x | 50.00 | - | 82.80 | - | 14970.000 | - | 743.2 | 46,107.2 | `d156d2e8d5` |
| `admm_pruner` | matrix-compression | - | 1.00x | 90.25 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `armor_pruner` | matrix-compression | - | 0.97x | 92.66 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `ast_pruner` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `astro_quantizer` | matrix-compression | - | 1.00x | 40.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `bonsai_pruner` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `cachenotes_compressor` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `come_compressor` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `compot_compressor` | bert | - | 1.12x | 94.04 | - | 82.30 | - | 8160.000 | - | 249.7 | 111,887.1 | `2f366e4546` |
| `compot_compressor_class` | bert | - | 1.00x | 90.48 | 36.40 | - | - | 4290.000 | - | 303.3 | 114,350.1 | `2f366e4546` |
| `compot_compressor_factory` | gpt | - | 1.00x | - | - | - | - | 178370.000 | - | 816.2 | 1,438.6 | `6267bfb8a8` |
| `coopq_quantizer` | matrix-compression | - | 1.00x | 55.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `cr_compressor` | bert | - | 1.00x | 94.38 | - | - | - | 11460.000 | - | 291.9 | 101,635.9 | `2f366e4546` |
| `cur_compressor` | bert | - | 1.00x | 94.38 | - | - | - | 5040.000 | - | 291.9 | 185,291.0 | `2f366e4546` |
| `cur_moe_compressor` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `delta_kv` | matrix-compression | 66,956,548 | 1.00x | 95.00 | - | - | - | - | - | 326.5 | 72,382.5 | `25d9fe36fa` |
| `direct_svd` | matrix-compression | - | 1.03x | 92.43 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `dobi_svd` | matrix-compression | - | 1.00x | 94.38 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `dobi_svd_compressor` | gpt | - | 1.00x | - | - | - | - | 189700.000 | - | 816.2 | 1,379.8 | `6267bfb8a8` |
| `drone_compressor` | gpt | - | 1.00x | 45.00 | - | - | - | 179910.000 | - | 816.2 | 1,420.9 | `6267bfb8a8` |
| `dynamic_entropy_compressor` | gpt | - | 1.00x | - | - | - | - | 221690.000 | - | 816.2 | 1,179.3 | `6267bfb8a8` |
| `dynamic_rank` | matrix-compression | - | 1.00x | 70.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `enlighten_compressor` | bert | - | 1.00x | 35.00 | 0.00 | - | - | 470.000 | - | - | 0.0 | `2f366e4546` |
| `entropy_budget_allocation` | matrix-compression | - | 1.00x | 35.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `erc_svd` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `faq_quantizer` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `flap` | matrix-compression | 66,956,548 | 1.00x | 94.70 | - | - | - | - | - | 326.5 | 92,633.9 | `25d9fe36fa` |
| `flat_llm` | matrix-compression | - | 1.00x | 93.69 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `flat_llm_compressor` | gpt | - | 1.00x | - | - | - | - | 181140.000 | - | 816.2 | 1,396.5 | `6267bfb8a8` |
| `frandi_compressor` | gpt | - | 2.77x | 93.81 | - | - | - | 24360.000 | - | 320.5 | 30,386.1 | `2f366e4546` |
| `frandi_compressor_class` | bert | - | 1.06x | 90.60 | 37.80 | - | - | 5540.000 | - | 304.1 | 98,435.8 | `2f366e4546` |
| `frandi_compressor_factory` | gpt | - | 1.00x | - | - | - | - | 214160.000 | - | 816.2 | 1,207.6 | `6267bfb8a8` |
| `fwsvd_compressor` | matrix-compression | - | 1.00x | 45.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `gfws` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `gfws_d_compressor` | bert | - | 1.00x | 0.00 | 0.00 | - | - | 500.000 | - | - | 0.0 | `2f366e4546` |
| `grecodec_compressor` | bert | - | 1.00x | 94.38 | - | - | - | 1940.000 | - | 291.9 | 163,381.6 | `2f366e4546` |
| `grecodec_compressor_class` | bert | - | 1.00x | 90.60 | 37.80 | - | - | 2950.000 | - | 304.1 | 81,291.0 | `2f366e4546` |
| `hashed_net_compressor` | bert | - | 1.00x | 93.46 | - | - | - | 2900.000 | - | 446.0 | 129,805.6 | `2f366e4546` |
| `hnc_svd` | hnc-svd | - | 2.00x | 93.40 | - | - | - | 0.740 | - | - | - | `f8a92b3c1d` |
| `huffman_coder` | matrix-compression | - | 1.00x | 20.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `hybrid_svd_layer` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `hybridkv_compressor` | bert | - | 1.00x | 93.46 | - | - | - | 2240.000 | - | 446.0 | 102,012.3 | `2f366e4546` |
| `kernel` | triton | - | - | - | - | - | - | 10.090 | 0.01x | - | - | `0911d2c0ce` |
| `kmeans_quantize` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `kronecker_compressor` | bert | - | 1.00x | 90.60 | - | - | - | 11000.000 | - | 291.9 | 35,977.3 | `2f366e4546` |
| `kronecker_tt_lora` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `kvtc_compressor` | bert | - | 1.00x | 93.46 | 0.00 | - | - | 550.000 | - | 446.0 | 101,112.1 | `2f366e4546` |
| `liftr` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `lillama_compressor` | bert | - | 1.00x | 0.00 | - | - | - | 880.000 | - | - | 0.0 | `2f366e4546` |
| `lsc_compressor` | bert | - | 1.00x | 94.27 | - | - | - | 6950.000 | - | 351.7 | 128,139.1 | `2f366e4546` |
| `mamba_shedder` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `mgie_budget` | matrix-compression | - | 1.00x | 35.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `mgie_svd_compressor` | bert | - | 1.00x | 0.00 | 0.00 | - | - | 510.000 | - | - | 0.0 | `2f366e4546` |
| `mlorq_compressor` | gpt | - | 1.00x | 50.00 | - | - | - | 6800.000 | - | - | 0.0 | `6267bfb8a8` |
| `modegpt_compressor` | bert | - | 1.00x | 94.04 | 0.00 | - | - | 530.000 | - | 457.1 | 128,273.4 | `2f366e4546` |
| `nha_compressor` | bert | - | 1.00x | 94.61 | - | - | - | 50560.000 | - | 508.6 | 21,131.3 | `6267bfb8a8` |
| `nmf_compressor` | gpt | - | 1.00x | 94.04 | - | - | - | 840.000 | - | 457.1 | 130,799.7 | `2f366e4546` |
| `oats_compressor` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `optimal_subspace_projection` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `pca_quant_compressor` | gpt | - | 1.00x | - | - | - | - | 179420.000 | - | 816.2 | 1,435.4 | `6267bfb8a8` |
| `proxsparse` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `rala_compressor` | bert | - | 1.00x | 78.33 | 35.70 | - | - | 26410.000 | - | 398.0 | 114,160.1 | `2f366e4546` |
| `replaceme_compressor` | bert | - | 1.00x | 94.04 | - | - | - | 8180.000 | - | 457.0 | 192,063.6 | `2f366e4546` |
| `replaceme_compressor_factory` | gpt | - | 1.00x | - | - | - | - | 174510.000 | - | 816.2 | 1,444.3 | `6267bfb8a8` |
| `residual_svd_compressor` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `roast_hashed_linear` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `rotational_slicegpt_int4` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `rpca_compressor` | bert | - | 1.00x | 94.27 | 81.90 | 70.30 | - | 21440.000 | - | 302.9 | 232,095.4 | `2f366e4546` |
| `saes_svd_compressor` | bert | - | 1.17x | 93.92 | - | - | - | 8510.000 | - | 302.9 | 90,617.9 | `2f366e4546` |
| `saes_svd_v2` | matrix-compression | - | 1.00x | 60.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `salr` | matrix-compression | - | 1.00x | 55.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `salr_compressor_factory` | gpt | - | 1.00x | - | - | - | - | 178530.000 | - | 816.2 | 1,442.2 | `6267bfb8a8` |
| `sdrm_compressor` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `slicegpt` | matrix-compression | - | 1.00x | 55.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `slicegpt_compressor` | bert | - | 1.00x | 93.92 | - | - | - | 6840.000 | - | 302.9 | 102,422.4 | `2f366e4546` |
| `slicegpt_pca` | matrix-compression | 66,955,779 | 1.00x | 81.50 | - | - | - | - | - | 304.5 | 174,628.4 | `25d9fe36fa` |
| `slim_compressor` | gpt | - | 1.00x | 91.74 | - | - | - | 11290.000 | - | 456.3 | 102,262.3 | `2f366e4546` |
| `soka_compressor` | matrix-compression | - | 1.00x | 70.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `sparsegpt` | matrix-compression | - | 1.00x | 60.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `subspace_projection` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `svd_linear` | matrix-compression | - | 1.00x | 51.67 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `svd_llm_compressor` | gpt2 | - | 1.00x | 94.50 | - | - | - | 9970.000 | - | 357.3 | 226,437.4 | `2f366e4546` |
| `swift_svd` | matrix-compression | 46,464,771 | 1.44x | 70.00 | - | - | - | - | - | 297.8 | 30,742.8 | `25d9fe36fa` |
| `swift_svd_act` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `swift_svd_compressor` | bert | - | 1.00x | 93.92 | - | - | - | 15020.000 | - | 292.7 | 101,528.9 | `2f366e4546` |
| `tomoe_compressor` | matrix-compression | - | 1.00x | 50.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `tt_compressor` | gpt2 | - | 1.05x | 50.92 | - | - | - | 42090.000 | - | 598.3 | 24,832.0 | `2f366e4546` |
| `tt_compressor_factory` | gpt | - | 1.05x | - | - | - | - | 272950.000 | - | 816.2 | 1,049.0 | `6267bfb8a8` |
| `tt_linear_compressor` | gpt | - | 1.05x | - | - | - | - | 310690.000 | - | 816.2 | 901.1 | `6267bfb8a8` |
| `uniql_compressor` | bert | - | 1.00x | 93.23 | - | - | - | 5770.000 | - | 292.7 | 42,953.8 | `2f366e4546` |
| `zerokv_neo_compressor` | matrix-compression | - | 1.00x | 70.00 | - | - | - | - | - | - | - | `25d9fe36fa` |
| `zerokv_neo_compressor_factory` | gpt | - | 1.00x | 0.43 | - | - | - | 301210.000 | - | - | 0.0 | `6267bfb8a8` |
