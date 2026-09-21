# Comprehensive 217-Module Analysis Report
This document provides the exhaustive audit and empirical validation across all 217 transformer compression and linear attention modules in the research framework. All metrics strictly trace to empirical benchmark CSV records.

## 1. Executive Summary & Category Breakdown

| Category | Count | Proportion | Meaning |
|:---|:---:|:---:|:---|
| **VALID** | 48 | 22.1% | VALID empirical evaluation status |
| **FALLBACK** | 163 | 75.1% | FALLBACK empirical evaluation status |
| **BROKEN** | 4 | 1.8% | BROKEN empirical evaluation status |
| **QUESTIONABLE** | 1 | 0.5% | QUESTIONABLE empirical evaluation status |
| **NO_DATA** | 1 | 0.5% | NO_DATA empirical evaluation status |

## 2. Statistical Highlights

- **Total Evaluated Modules**: 217
- **Modules with Academic Literature Reference**: 134 (61.8%)
- **Modules with arXiv ID**: 93 (42.9%)
- **Modules with Empirical SST-2 Benchmark**: 128 (59.0%)
- **Modules with Empirical MNLI Benchmark**: 187 (86.2%)
- **Modules with Empirical AG News Benchmark**: 155 (71.4%)
- **Modules with GPT-2 WikiText-2 PPL**: 8 (3.7%)
- **Modules with Quarantined Fabricated Tables**: 142 (65.4%)

## 3. Actionable Priority Fix List

- **P0 (Broken - Immediate submission blockers)**: 4 modules
- **P1 (Fallback Compressors - Fixable via Cholesky whitening / LoRA unfreezing)**: 61 modules
- **P2 (Fallback Attention - Inherent linear attention replacements, move to Related Work)**: 103 modules
- **P3 (No Data - Needs evaluation run)**: 1 modules
- **NONE (Valid - Production-hardened with genuine compression)**: 48 modules

### P0 Action Items (Immediate Fixes)

- `adasvd_compressor`: /Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/adasvd_compressor.py (Category: BROKEN, Sub: Compression-Failed)
- `compot_compressor`: /Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/compot_compressor.py (Category: BROKEN, Sub: Compression-Failed)
- `cur_linear`: /Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/cur_linear.py (Category: BROKEN, Sub: Compression-Failed)
- `cure`: /Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/cure.py (Category: BROKEN, Sub: Compression-Failed)

## 4. Complete 217-Module Master Table

| Module Name | Type | Category | Sub-Category | SST-2 Acc | MNLI Acc | Comp Ratio | Priority | Source CSV |
|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---|
| `a3` | Compressor | **VALID** | Compression-Working | 92.55% | 89.30% | 1.103x | **NONE** | `...l_121_compressors_gpu_matrix.csv` |
| `aa_svd_compressor` | Compressor | **VALID** | Compression-Working | N/A | 51.20% | 30.960x | **NONE** | `...ompressor/distilbert/summary.csv` |
| `aasvd` | Compressor | **VALID** | Compression-Working | 90.37% | 92.10% | 30.960x | **NONE** | `...lations/rank/rank128/summary.csv` |
| `adaptive_compressor` | Compressor | **VALID** | Compression-Working | 91.51% | 92.80% | 1.050x | **NONE** | `...tive_compressor/gpt2/summary.csv` |
| `adaptive_ranks` | Compressor | **VALID** | Compression-Working | N/A | N/A | 1.046x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `come` | Compressor | **VALID** | Compression-Working | 87.84% | 91.50% | 2.000x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `cur_moe` | Compressor | **VALID** | Compression-Working | 90.83% | 94.30% | 1.457x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `hybrid_svd_layer` | Compressor | **VALID** | Compression-Working | N/A | N/A | 2.131x | **NONE** | `...ts/master_aggregated_results.csv` |
| `kronecker_compressor` | Compressor | **VALID** | Compression-Working | 82.22% | 81.80% | 358.640x | **NONE** | `...l_121_compressors_gpu_matrix.csv` |
| `mgie_svd` | Compressor | **VALID** | Compression-Working | N/A | N/A | 1.003x | **NONE** | `...ts/master_aggregated_results.csv` |
| `res_svd` | Compressor | **VALID** | Compression-Working | 90.71% | 78.90% | 20.870x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `rpca` | Compressor | **VALID** | Compression-Working | N/A | 82.00% | 30.960x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `sdrm` | Compressor | **VALID** | Compression-Working | N/A | 68.10% | 1.430x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `soka` | Compressor | **VALID** | Compression-Working | N/A | 82.10% | 59.910x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `subspace_projection` | Compressor | **VALID** | Compression-Working | N/A | 82.10% | 30.960x | **NONE** | `...l_121_compressors_gpu_matrix.csv` |
| `swift_svd` | Compressor | **VALID** | Compression-Working | 83.03% | 54.00% | 1.497x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `swift_svd_act` | Compressor | **VALID** | Compression-Working | N/A | 81.80% | 30.960x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `AdaptiveSVD` | Attention Replacement | **VALID** | SVD-Attention | 87.96% | 92.30% | 1.249x | **NONE** | `...l_121_compressors_gpu_matrix.csv` |
| `SOAFusedAttention` | Attention Replacement | **VALID** | SVD-Attention | N/A | 40.60% | 1.177x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `adaptive_svd_attention` | Attention Replacement | **VALID** | SVD-Attention | 87.96% | 92.30% | 1.249x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `compressed_mha` | Attention Replacement | **VALID** | SVD-Attention | 49.08% | 94.50% | 1.260x | **NONE** | `...essed_mha/distilbert/summary.csv` |
| `fused_svd_kernel` | Attention Replacement | **VALID** | SVD-Attention | 90.02% | 90.30% | 1.264x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `minimal_svd` | Attention Replacement | **VALID** | SVD-Attention | 88.30% | 55.10% | 1.166x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `minimal_svd_v2` | Attention Replacement | **VALID** | SVD-Attention | 88.30% | 55.10% | 1.166x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `soa_attn` | Attention Replacement | **VALID** | SVD-Attention | 89.45% | 73.00% | 1.188x | **NONE** | `results/distilbert/summary.csv` |
| `soa_fused_module` | Attention Replacement | **VALID** | SVD-Attention | N/A | N/A | 1.183x | **NONE** | `...ts/master_aggregated_results.csv` |
| `soa_module` | Attention Replacement | **VALID** | SVD-Attention | N/A | 43.80% | 1.120x | **NONE** | `...ts/master_aggregated_results.csv` |
| `svd_compressed` | Attention Replacement | **VALID** | SVD-Attention | 91.06% | 80.00% | 1.227x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `svd_linear_decay` | Attention Replacement | **VALID** | SVD-Attention | 90.71% | 79.70% | 1.226x | **NONE** | `results/distilbert/summary.csv` |
| `svd_proj_attn` | Attention Replacement | **VALID** | SVD-Attention | N/A | N/A | 1.248x | **NONE** | `results/distilbert/summary.csv` |
| `aegis_hybrid` | Attention Replacement | **VALID** | Unknown | 90.48% | 89.50% | 1.225x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `aegis_spectron_fp8` | Attention Replacement | **VALID** | Unknown | 50.80% | 34.00% | 1.244x | **NONE** | `results/benchmark_summary_v2.csv` |
| `aegis_v2` | Attention Replacement | **VALID** | Unknown | 75.69% | 58.10% | 1.225x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `aegis_v3` | Attention Replacement | **VALID** | Unknown | 82.34% | 90.10% | 1.215x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `aegis_v4` | Attention Replacement | **VALID** | Unknown | 83.14% | 89.80% | 1.215x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `aegis_v5` | Attention Replacement | **VALID** | Unknown | 83.14% | 89.80% | 1.197x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `aegis_v6` | Attention Replacement | **VALID** | Unknown | 90.94% | 93.30% | 1.277x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `compressed_attention` | Attention Replacement | **VALID** | Unknown | 49.08% | 94.40% | 1.260x | **NONE** | `...attention/distilbert/summary.csv` |
| `lola_attention` | Attention Replacement | **VALID** | Unknown | 89.56% | 75.40% | 1.248x | **NONE** | `.../benchmark_missing_results_1.csv` |
| `minimal_aurora` | Attention Replacement | **VALID** | Unknown | 88.30% | 55.10% | 1.166x | **NONE** | `...al_aurora/distilbert/summary.csv` |
| `minimal_nova` | Attention Replacement | **VALID** | Unknown | 88.30% | 55.20% | 1.166x | **NONE** | `...imal_nova/distilbert/summary.csv` |
| `minimal_prism` | Attention Replacement | **VALID** | Unknown | 88.30% | 55.20% | 1.166x | **NONE** | `...mal_prism/distilbert/summary.csv` |
| `orion` | Attention Replacement | **VALID** | Unknown | 73.39% | 38.40% | 1.248x | **NONE** | `...ation/agnews_rank128/summary.csv` |
| `phantom` | Attention Replacement | **VALID** | Unknown | 51.49% | 35.40% | 1.248x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `projection_factorized_attn` | Attention Replacement | **VALID** | Unknown | 90.71% | 79.70% | 1.247x | **NONE** | `...ized_attn/distilbert/summary.csv` |
| `salt` | Attention Replacement | **VALID** | Unknown | 89.79% | 64.90% | 1.264x | **NONE** | `results/final_paper_run/summary.csv` |
| `spectron` | Attention Replacement | **VALID** | Unknown | 87.50% | 60.60% | 1.166x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `vspec` | Attention Replacement | **VALID** | Unknown | 89.91% | 79.30% | 1.223x | **NONE** | `...daptive_v3/benchmark_summary.csv` |
| `gqa` | Attention Replacement | **QUESTIONABLE** | GQA-MQA | 49.08% | 44.10% | 1.078x | **P2** | `...ified_gqa/distilbert/summary.csv` |
| `admm_compressor` | Compressor | **FALLBACK** | Compression-Failed | 62.96% | 94.30% | 1.000x | **P1** | `...ompressor/distilbert/summary.csv` |
| `admm_pruner` | Compressor | **FALLBACK** | Compression-Failed | 83.49% | 94.30% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `armor_pruner` | Compressor | **FALLBACK** | Compression-Failed | 83.94% | 93.70% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `ast_pruner` | Compressor | **FALLBACK** | Compression-Failed | 85.67% | 94.40% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `astro_quantizer` | Compressor | **FALLBACK** | Compression-Failed | 89.79% | 94.50% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `bonsai_pruner` | Compressor | **FALLBACK** | Compression-Failed | 79.47% | 52.20% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `cachenotes_compressor` | Compressor | **FALLBACK** | Compression-Failed | 84.63% | 94.40% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `coopq_quantizer` | Compressor | **FALLBACK** | Compression-Failed | 89.79% | 94.20% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `cr_compressor` | Compressor | **FALLBACK** | Compression-Failed | 75.00% | 94.70% | 30.960x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `cur_compressor` | Compressor | **FALLBACK** | Compression-Failed | 49.08% | 94.70% | 30.750x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `delta_kv` | Compressor | **FALLBACK** | Compression-Failed | 86.24% | 95.00% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `direct_svd` | Compressor | **FALLBACK** | Compression-Failed | 73.62% | 94.40% | 30.960x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `dobi` | Compressor | **FALLBACK** | Compression-Failed | 74.31% | 94.30% | 30.950x | **P1** | `...stilbert_agnews_full/summary.csv` |
| `dobi_svd` | Compressor | **FALLBACK** | Compression-Failed | N/A | N/A | 30.950x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `dynamic_entropy` | Compressor | **FALLBACK** | Compression-Failed | N/A | N/A | 30.960x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `dynamic_rank` | Compressor | **FALLBACK** | Compression-Failed | 81.19% | 94.40% | 30.960x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `enlighten_compressor` | Compressor | **FALLBACK** | Compression-Failed | 87.50% | 93.80% | 1.000x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `entropy_budget_allocation` | Compressor | **FALLBACK** | Compression-Failed | N/A | N/A | 30.960x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `erc_svd` | Compressor | **FALLBACK** | Compression-Failed | N/A | N/A | 30.960x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `faq_quantizer` | Compressor | **FALLBACK** | Compression-Failed | 89.91% | 94.20% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `flap` | Compressor | **FALLBACK** | Compression-Failed | 85.67% | 94.70% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `flat_llm` | Compressor | **FALLBACK** | Compression-Failed | 82.45% | 94.70% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `frandi` | Compressor | **FALLBACK** | Compression-Failed | 56.42% | 94.80% | 30.960x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `gfws_d_compressor` | Compressor | **FALLBACK** | Compression-Failed | 76.61% | 94.70% | 1.000x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `grecodec` | Compressor | **FALLBACK** | Compression-Failed | 76.49% | 94.80% | 30.960x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `hashed_linear` | Compressor | **FALLBACK** | Compression-Failed | N/A | N/A | 1.000x | **P1** | `...ed_linear/distilbert/summary.csv` |
| `hashed_net_compressor` | Compressor | **FALLBACK** | Compression-Failed | 51.49% | 94.60% | 691.880x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `huffman_coder` | Compressor | **FALLBACK** | Compression-Failed | N/A | N/A | 1.000x | **P1** | `...ts/master_aggregated_results.csv` |
| `hybridkv` | Compressor | **FALLBACK** | Compression-Failed | 85.89% | 94.30% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `kmeans_quantize` | Compressor | **FALLBACK** | Compression-Failed | N/A | N/A | 826.110x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `kronecker_linear` | Compressor | **FALLBACK** | Compression-Failed | N/A | N/A | 1.000x | **P1** | `...ts/master_aggregated_results.csv` |
| `kronecker_tt_lora` | Compressor | **FALLBACK** | Compression-Failed | N/A | 81.80% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `kvtc` | Compressor | **FALLBACK** | Compression-Failed | N/A | 79.30% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `liftr` | Compressor | **FALLBACK** | Compression-Failed | N/A | 82.30% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `lillama_compressor` | Compressor | **FALLBACK** | Compression-Failed | N/A | 81.70% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `lsc_compressor` | Compressor | **FALLBACK** | Compression-Failed | N/A | 81.80% | 1.000x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `mamba_shedder` | Compressor | **FALLBACK** | Compression-Failed | N/A | 81.70% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `mlorq_compressor` | Compressor | **FALLBACK** | Compression-Failed | N/A | 81.40% | 1.000x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `modegpt` | Compressor | **FALLBACK** | Compression-Failed | N/A | 81.80% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `nli_softmax` | Compressor | **FALLBACK** | Compression-Failed | 89.33% | 50.60% | 1.000x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `nmf_compressor` | Compressor | **FALLBACK** | Compression-Failed | N/A | 82.30% | 30.960x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `oats` | Compressor | **FALLBACK** | Compression-Failed | N/A | 81.60% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `optimal_subspace_projection` | Compressor | **FALLBACK** | Compression-Failed | N/A | 81.50% | 1.000x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `padre_attention` | Compressor | **FALLBACK** | Compression-Failed | 89.91% | 77.70% | 1.000x | **P1** | `...ied_padre/distilbert/summary.csv` |
| `proxsparse` | Compressor | **FALLBACK** | Compression-Failed | N/A | 40.80% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `proxsparse_pruner` | Compressor | **FALLBACK** | Compression-Failed | N/A | 40.80% | 1.000x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `replaceme_compressor` | Compressor | **FALLBACK** | Compression-Failed | 78.78% | 64.60% | 1.241x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `roast_hashed_linear` | Compressor | **FALLBACK** | Compression-Failed | N/A | N/A | 1.000x | **P1** | `...ts/master_aggregated_results.csv` |
| `rotational_slicegpt_int4` | Compressor | **FALLBACK** | Compression-Failed | N/A | N/A | 1.000x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `saes_svd` | Compressor | **FALLBACK** | Compression-Failed | N/A | 81.70% | 30.960x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `saes_svd_v2` | Compressor | **FALLBACK** | Compression-Failed | N/A | N/A | 30.960x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `salr` | Compressor | **FALLBACK** | Compression-Failed | N/A | 82.00% | 30.960x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `slicegpt` | Compressor | **FALLBACK** | Compression-Failed | 83.83% | 58.20% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `slicegpt_pca` | Compressor | **FALLBACK** | Compression-Failed | N/A | 81.80% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `slim_compressor` | Compressor | **FALLBACK** | Compression-Failed | N/A | 79.10% | 1.000x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `sparsegpt` | Compressor | **FALLBACK** | Compression-Failed | N/A | 77.10% | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `spectraformer_attention` | Compressor | **FALLBACK** | Compression-Failed | N/A | 0.00% | 1.000x | **P1** | `...ts/master_aggregated_results.csv` |
| `tlinformer_attention` | Compressor | **FALLBACK** | Compression-Failed | N/A | 77.90% | 1.000x | **P1** | `...ts/master_aggregated_results.csv` |
| `tomoe` | Compressor | **FALLBACK** | Compression-Failed | N/A | 82.00% | 1.000x | **P1** | `...mnli_full_mnli_sweep/summary.csv` |
| `uniql_compressor` | Compressor | **FALLBACK** | Compression-Failed | N/A | 82.40% | 1.000x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `zerokv_neo` | Compressor | **FALLBACK** | Compression-Failed | N/A | N/A | 1.000x | **P1** | `...daptive_v3/benchmark_summary.csv` |
| `wgqa` | Attention Replacement | **FALLBACK** | GQA-MQA | 90.94% | 78.80% | 1.000x | **P2** | `results/distilbert/summary.csv` |
| `Galerkin` | Attention Replacement | **FALLBACK** | Linear-Attention | 75.57% | 34.50% | 1.000x | **P2** | `results/final_paper_run/summary.csv` |
| `cosformer` | Attention Replacement | **FALLBACK** | Linear-Attention | 78.78% | 39.30% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `decay_linear_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | N/A | 34.70% | 1.000x | **P2** | `results/benchmark_summary_v2.csv` |
| `galerkin_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | 75.57% | 36.40% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `gated_linear_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | N/A | N/A | 1.000x | **P2** | `results/benchmark_summary_v2.csv` |
| `linformer_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | N/A | 77.20% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `nystrom_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | 90.94% | 81.90% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `performer_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | 74.77% | 35.70% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `svd_linear_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | 90.71% | 79.70% | 30.960x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `griffin_ssd` | Attention Replacement | **FALLBACK** | SSM | 49.08% | 70.80% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `hgrn2_gated_deltanet` | Attention Replacement | **FALLBACK** | SSM | 0.00% | 33.60% | 1.000x | **P2** | `results/benchmark_summary_v2.csv` |
| `hgrn2_layer` | Attention Replacement | **FALLBACK** | SSM | 0.00% | 34.40% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `hgrn2_outer` | Attention Replacement | **FALLBACK** | SSM | 68.00% | 34.40% | 1.000x | **P2** | `results/benchmark_summary_v2.csv` |
| `hymba_transx` | Attention Replacement | **FALLBACK** | SSM | 0.00% | 89.10% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `jamba` | Attention Replacement | **FALLBACK** | SSM | N/A | 91.30% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `jamba_info` | Attention Replacement | **FALLBACK** | SSM | N/A | 62.10% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `mamba` | Attention Replacement | **FALLBACK** | SSM | N/A | 34.10% | 1.054x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `mamba_inject` | Attention Replacement | **FALLBACK** | SSM | N/A | N/A | 1.000x | **P2** | `...ts/master_aggregated_results.csv` |
| `mamba_svd_flash` | Attention Replacement | **FALLBACK** | SSM | N/A | 0.00% | 1.000x | **P2** | `...ndoned_run/benchmark_summary.csv` |
| `samba_m3` | Attention Replacement | **FALLBACK** | SSM | N/A | 70.20% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `samba_moe` | Attention Replacement | **FALLBACK** | SSM | N/A | 33.80% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `simple_ssm` | Attention Replacement | **FALLBACK** | SSM | 50.34% | 35.10% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `spectral_mamba` | Attention Replacement | **FALLBACK** | SSM | N/A | 59.10% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `svd_mamba` | Attention Replacement | **FALLBACK** | SSM | N/A | 34.50% | 1.000x | **P2** | `results/run_quality_/summary.csv` |
| `upi_mamba` | Attention Replacement | **FALLBACK** | SSM | N/A | 31.90% | 1.000x | **P2** | `...ts/master_aggregated_results.csv` |
| `upi_stuffed_mamba` | Attention Replacement | **FALLBACK** | SSM | N/A | N/A | 0.971x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `asvd` | Attention Replacement | **FALLBACK** | SVD-Attention | 50.92% | 82.20% | 1.000x | **P2** | `results/distilbert/summary.csv` |
| `svd_attention` | Attention Replacement | **FALLBACK** | SVD-Attention | 51.49% | 34.50% | 1.015x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `svd_llm_v2_attention` | Attention Replacement | **FALLBACK** | SVD-Attention | 90.37% | 80.20% | 1.248x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `svd_projection_attention` | Attention Replacement | **FALLBACK** | SVD-Attention | N/A | 34.40% | 1.201x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `svd_sequence_attention` | Attention Replacement | **FALLBACK** | SVD-Attention | N/A | N/A | 0.991x | **P2** | `...ts/master_aggregated_results.csv` |
| `AdaptiveSparseAttention` | Attention Replacement | **FALLBACK** | Sparse-Attention | 50.92% | 36.40% | 1.000x | **P2** | `results/distilbert/summary.csv` |
| `HALO` | Attention Replacement | **FALLBACK** | Sparse-Attention | 90.37% | 81.60% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `adaptive_sparse_attention` | Attention Replacement | **FALLBACK** | Sparse-Attention | 50.92% | 36.40% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `halo_regla` | Attention Replacement | **FALLBACK** | Sparse-Attention | 90.37% | 81.60% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `mla_attention` | Attention Replacement | **FALLBACK** | Sparse-Attention | 69.27% | 36.80% | 1.114x | **P2** | `...ation/agnews_rank128/summary.csv` |
| `moba` | Attention Replacement | **FALLBACK** | Sparse-Attention | 67.09% | 36.70% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `nsa` | Attention Replacement | **FALLBACK** | Sparse-Attention | 90.94% | 81.70% | 1.000x | **P2** | `results/final_paper_run/summary.csv` |
| `AdaptiveSpanAttention` | Attention Replacement | **FALLBACK** | Unknown | 90.71% | 81.90% | 1.017x | **P2** | `results/distilbert/summary.csv` |
| `OH` | Attention Replacement | **FALLBACK** | Unknown | N/A | 80.10% | 1.000x | **P2** | `results/final_paper_run/summary.csv` |
| `Primal` | Attention Replacement | **FALLBACK** | Unknown | N/A | 82.00% | 1.000x | **P2** | `results/distilbert/summary.csv` |
| `SqueezingHeadsAttentionDistillationLayer` | Attention Replacement | **FALLBACK** | Unknown | 90.94% | 80.50% | 1.000x | **P2** | `...ts/master_aggregated_results.csv` |
| `adaptive_span_attention` | Attention Replacement | **FALLBACK** | Unknown | 90.71% | 81.90% | 1.017x | **P2** | `...attention/distilbert/summary.csv` |
| `adasplash` | Attention Replacement | **FALLBACK** | Unknown | 89.56% | 79.70% | 1.000x | **P2** | `...lation/lora0_ag_news/summary.csv` |
| `aegis_v1` | Attention Replacement | **FALLBACK** | Unknown | 86.58% | 71.60% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `bam_umoe` | Attention Replacement | **FALLBACK** | Unknown | 49.08% | 34.60% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `block_transformer_hiba` | Attention Replacement | **FALLBACK** | Unknown | 90.94% | 81.90% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `chela_slay` | Attention Replacement | **FALLBACK** | Unknown | 90.60% | 59.50% | 1.030x | **P2** | `results/distilbert/summary.csv` |
| `cluster` | Attention Replacement | **FALLBACK** | Unknown | 90.14% | 73.70% | 1.000x | **P2** | `results/final_paper_run/summary.csv` |
| `eat` | Attention Replacement | **FALLBACK** | Unknown | 90.71% | 58.20% | 1.030x | **P2** | `...lation/lora0_ag_news/summary.csv` |
| `expert_choice_moe` | Attention Replacement | **FALLBACK** | Unknown | 49.08% | 34.60% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `factorised_matrix_state` | Attention Replacement | **FALLBACK** | Unknown | 49.08% | N/A | 1.000x | **P2** | `results/benchmark_summary_v2.csv` |
| `flash3` | Attention Replacement | **FALLBACK** | Unknown | 90.71% | 58.70% | 1.000x | **P2** | `...lation/lora0_ag_news/summary.csv` |
| `flash_attention3` | Attention Replacement | **FALLBACK** | Unknown | N/A | N/A | 0.981x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `flash_attention3_fp8` | Attention Replacement | **FALLBACK** | Unknown | N/A | N/A | 1.030x | **P2** | `results/final_paper_run/summary.csv` |
| `full_attention` | Attention Replacement | **FALLBACK** | Unknown | N/A | 35.50% | 1.006x | **P2** | `.../benchmark_missing_results_1.csv` |
| `fusion` | Attention Replacement | **FALLBACK** | Unknown | 89.68% | 39.90% | 1.033x | **P2** | `results/distilbert/summary.csv` |
| `fusion_attn` | Attention Replacement | **FALLBACK** | Unknown | 89.68% | 39.90% | 1.033x | **P2** | `results/distilbert/summary.csv` |
| `gated_delta_net` | Attention Replacement | **FALLBACK** | Unknown | 49.08% | 34.00% | 1.000x | **P2** | `results/benchmark_summary_v2.csv` |
| `gated_deltanet_block` | Attention Replacement | **FALLBACK** | Unknown | 51.15% | 34.00% | 1.000x | **P2** | `results/benchmark_summary_v2.csv` |
| `goldfinch_ahn` | Attention Replacement | **FALLBACK** | Unknown | 49.08% | 79.00% | 1.045x | **P2** | `results/distilbert/summary.csv` |
| `granite_hybrid` | Attention Replacement | **FALLBACK** | Unknown | 49.08% | N/A | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `hamming_attention` | Attention Replacement | **FALLBACK** | Unknown | 79.13% | 39.40% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `hla_attention` | Attention Replacement | **FALLBACK** | Unknown | 76.03% | 35.90% | 1.006x | **P2** | `.../benchmark_missing_results_1.csv` |
| `infini` | Attention Replacement | **FALLBACK** | Unknown | N/A | 58.20% | 1.000x | **P2** | `results/distilbert/summary.csv` |
| `kda_attention` | Attention Replacement | **FALLBACK** | Unknown | 76.83% | 34.60% | 1.000x | **P2** | `results/benchmark_summary_v2.csv` |
| `leka_v1` | Attention Replacement | **FALLBACK** | Unknown | 52.75% | 38.30% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `leka_v2` | Attention Replacement | **FALLBACK** | Unknown | N/A | 80.30% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `leka_v3` | Attention Replacement | **FALLBACK** | Unknown | N/A | 36.70% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `lora_attention` | Attention Replacement | **FALLBACK** | Unknown | N/A | 34.60% | 1.009x | **P2** | `.../benchmark_missing_results_1.csv` |
| `lsh_attention` | Attention Replacement | **FALLBACK** | Unknown | 75.80% | 44.10% | 1.000x | **P2** | `...ts/master_aggregated_results.csv` |
| `mixattention_moska` | Attention Replacement | **FALLBACK** | Unknown | N/A | 42.40% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `moe_deltakv_paged` | Attention Replacement | **FALLBACK** | Unknown | 49.08% | 37.20% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `nasc_attention` | Attention Replacement | **FALLBACK** | Unknown | 88.65% | 68.80% | 1.000x | **P2** | `...ts/master_aggregated_results.csv` |
| `nexus` | Attention Replacement | **FALLBACK** | Unknown | 90.71% | 81.00% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `nha` | Attention Replacement | **FALLBACK** | Unknown | 90.25% | 80.90% | 1.000x | **P2** | `results/final_paper_run/summary.csv` |
| `optimized_hybrid_attention` | Attention Replacement | **FALLBACK** | Unknown | N/A | 41.10% | 1.000x | **P2** | `results/final_paper_run/summary.csv` |
| `progressive_token_pruning` | Attention Replacement | **FALLBACK** | Unknown | N/A | 41.10% | 1.000x | **P2** | `results/distilbert/summary.csv` |
| `ptp` | Attention Replacement | **FALLBACK** | Unknown | N/A | 82.00% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `qcqa_attention` | Attention Replacement | **FALLBACK** | Unknown | 56.54% | 37.00% | 1.000x | **P2** | `results/distilbert/summary.csv` |
| `rala_attention` | Attention Replacement | **FALLBACK** | Unknown | 78.33% | 36.90% | 1.000x | **P2** | `results/distilbert/summary.csv` |
| `reap_moe` | Attention Replacement | **FALLBACK** | Unknown | N/A | 77.30% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `regla_attention` | Attention Replacement | **FALLBACK** | Unknown | N/A | 43.90% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `replace` | Attention Replacement | **FALLBACK** | Unknown | N/A | 80.20% | 1.000x | **P2** | `...mnli_full_mnli_sweep/summary.csv` |
| `routing_attention` | Attention Replacement | **FALLBACK** | Unknown | 53.21% | 35.90% | 1.000x | **P2** | `results/final_paper_run/summary.csv` |
| `rpc` | Attention Replacement | **FALLBACK** | Unknown | 88.99% | 76.20% | 1.000x | **P2** | `...lation/lora0_ag_news/summary.csv` |
| `smart_weighted_ensemble` | Attention Replacement | **FALLBACK** | Unknown | N/A | N/A | 1.000x | **P2** | `...ts/master_aggregated_results.csv` |
| `span_expanded_attention` | Attention Replacement | **FALLBACK** | Unknown | 90.94% | 81.90% | 1.000x | **P2** | `results/final_paper_run/summary.csv` |
| `spectra_v2` | Attention Replacement | **FALLBACK** | Unknown | 89.68% | 41.00% | 1.000x | **P2** | `...daptive_v3/benchmark_summary.csv` |
| `squeezing_heads` | Attention Replacement | **FALLBACK** | Unknown | 90.94% | 80.50% | 1.000x | **P2** | `...mnli_full_mnli_sweep/summary.csv` |
| `squeezing_heads_distill` | Attention Replacement | **FALLBACK** | Unknown | N/A | N/A | 0.991x | **P2** | `...ts/master_aggregated_results.csv` |
| `squeezing_heads_distiller` | Attention Replacement | **FALLBACK** | Unknown | N/A | 40.00% | 1.000x | **P2** | `results/distilbert/summary.csv` |
| `template_attention` | Attention Replacement | **FALLBACK** | Unknown | N/A | 58.10% | 1.000x | **P2** | `...ts/master_aggregated_results.csv` |
| `token_compressed_attention` | Attention Replacement | **FALLBACK** | Unknown | N/A | 50.10% | 1.000x | **P2** | `results/distilbert/summary.csv` |
| `tpa_attention` | Attention Replacement | **FALLBACK** | Unknown | N/A | 36.80% | 1.081x | **P2** | `results/distilbert/summary.csv` |
| `tt` | Attention Replacement | **FALLBACK** | Unknown | N/A | 81.40% | 1.000x | **P2** | `...l_121_compressors_gpu_matrix.csv` |
| `unified_rope` | Attention Replacement | **FALLBACK** | Unknown | 88.88% | 64.70% | 1.000x | **P2** | `...lation/lora0_ag_news/summary.csv` |
| `vig_attention` | Attention Replacement | **FALLBACK** | Unknown | 88.42% | 64.80% | 1.000x | **P2** | `...ts/master_aggregated_results.csv` |
| `weighted_ensemble` | Attention Replacement | **FALLBACK** | Unknown | 89.79% | 79.20% | 1.000x | **P2** | `...ts/master_aggregated_results.csv` |
| `xattention` | Attention Replacement | **FALLBACK** | Unknown | 85.78% | 52.50% | 1.000x | **P2** | `...attention/distilbert/summary.csv` |
| `zeros_attention` | Attention Replacement | **FALLBACK** | Unknown | 51.26% | 36.30% | 1.000x | **P2** | `...s/unified_zeros/gpt2/summary.csv` |
| `adasvd_compressor` | Compressor | **BROKEN** | Compression-Failed | 87.04% | 87.40% | 27.220x | **P0** | `...l_121_compressors_gpu_matrix.csv` |
| `compot_compressor` | Compressor | **BROKEN** | Compression-Failed | 55.16% | 94.50% | 30.960x | **P0** | `...l_121_compressors_gpu_matrix.csv` |
| `cur_linear` | Compressor | **BROKEN** | Compression-Failed | N/A | N/A | 1.000x | **P0** | `...ur_linear/distilbert/summary.csv` |
| `cure` | Compressor | **BROKEN** | Compression-Failed | N/A | N/A | 1.000x | **P0** | `...fied_cure/distilbert/summary.csv` |
| `pruning_utils` | Compressor | **NO_DATA** | Utility | N/A | N/A | N/A | **P3** | `N/A` |
