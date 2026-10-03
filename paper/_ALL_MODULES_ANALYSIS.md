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
- **Modules with Academic Literature Reference**: 129 (59.4%)
- **Modules with arXiv ID**: 93 (42.9%)
- **Modules with Empirical SST-2 Benchmark**: 217 (100.0%)
- **Modules with Empirical MNLI Benchmark**: 215 (99.1%)
- **Modules with Empirical AG News Benchmark**: 137 (63.1%)
- **Modules with GPT-2 WikiText-2 PPL**: 149 (68.7%)
- **Modules with Quarantined Fabricated Tables**: 137 (63.1%)

## 3. Actionable Priority Fix List

- **P0 (Broken - Immediate submission blockers)**: 4 modules
- **P1 (Fallback Compressors - Fixable via Cholesky whitening / LoRA unfreezing)**: 44 modules
- **P2 (Fallback Attention - Inherent linear attention replacements, move to Related Work)**: 120 modules
- **P3 (No Data - Needs evaluation run)**: 1 modules
- **NONE (Valid - Production-hardened with genuine compression)**: 48 modules

### P0 Action Items (Immediate Fixes)

- `adasvd_compressor`: /Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/adasvd_compressor.py (Category: BROKEN, Sub: Compression-Failed)
- `compot_compressor`: /Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/compot_compressor.py (Category: BROKEN, Sub: Compression-Failed)
- `cur_linear`: N/A (Category: BROKEN, Sub: Unknown)
- `cure`: N/A (Category: BROKEN, Sub: Unknown)

## 4. Complete 217-Module Master Table

| Module Name | Type | Category | Sub-Category | SST-2 Acc | MNLI Acc | Comp Ratio | Priority | Source CSV |
|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---|
| `aa_svd_compressor` | Compressor | **VALID** | Compression-Working | 90.71% | 67.00% | 30.960x | **NONE** | `...mmary_aa_svd_compressor_mnli.csv` |
| `adaptive_ranks` | Compressor | **VALID** | Compression-Working | 90.25% | 69.00% | 1.504x | **NONE** | `...aptive_ranks/qwen3_5/summary.csv` |
| `hybrid_svd_layer` | Compressor | **VALID** | Compression-Working | 94.15% | 81.50% | 2.131x | **NONE** | `...svd_layer/distilbert/summary.csv` |
| `kronecker_compressor` | Compressor | **VALID** | Compression-Working | 93.81% | 81.80% | 358.640x | **NONE** | `..._compressor/rank_128/summary.csv` |
| `mgie_svd` | Compressor | **VALID** | Compression-Working | 94.27% | 81.80% | 1.271x | **NONE** | `..._mgie_svd/distilbert/summary.csv` |
| `res_svd` | Compressor | **VALID** | Compression-Working | 88.76% | 65.80% | 20.870x | **NONE** | `...d_res_svd/distilbert/summary.csv` |
| `subspace_projection` | Compressor | **VALID** | Compression-Working | 94.38% | 82.30% | 30.960x | **NONE** | `...rojection/distilbert/summary.csv` |
| `swift_svd` | Compressor | **VALID** | Compression-Working | 90.94% | 53.30% | 2.578x | **NONE** | `...ied_sweep/distilbert/summary.csv` |
| `swift_svd_act` | Compressor | **VALID** | Compression-Working | 94.38% | 81.80% | 30.960x | **NONE** | `...ied_sweep/distilbert/summary.csv` |
| `AdaptiveSVD` | Attention Replacement | **VALID** | SVD-Attention | 88.30% | 92.30% | 1.249x | **NONE** | `...ied_sweep/distilbert/summary.csv` |
| `SOAFusedAttention` | Unknown | **VALID** | SVD-Attention | 89.11% | 72.70% | 1.182x | **NONE** | `...attention/distilbert/summary.csv` |
| `aasvd` | Unknown | **VALID** | SVD-Attention | 90.02% | 92.10% | 30.960x | **NONE** | `...ied_aasvd/distilbert/summary.csv` |
| `adaptive_svd_attention` | Attention Replacement | **VALID** | SVD-Attention | 88.30% | 92.30% | 1.249x | **NONE** | `...s_archived/benchmark_summary.csv` |
| `compressed_mha` | Unknown | **VALID** | SVD-Attention | 90.14% | 94.50% | 1.260x | **NONE** | `...baselines/distilbert/summary.csv` |
| `fused_svd_kernel` | Attention Replacement | **VALID** | SVD-Attention | 87.61% | 90.30% | 1.264x | **NONE** | `...vd_kernel/distilbert/summary.csv` |
| `minimal_svd` | Attention Replacement | **VALID** | SVD-Attention | 88.19% | 55.10% | 1.172x | **NONE** | `...nimal_svd/distilbert/summary.csv` |
| `minimal_svd_v2` | Unknown | **VALID** | SVD-Attention | 88.19% | 55.10% | 1.166x | **NONE** | `...al_svd_v2/distilbert/summary.csv` |
| `soa_attn` | Attention Replacement | **VALID** | SVD-Attention | 89.22% | 73.00% | 1.188x | **NONE** | `..._soa_attn/distilbert/summary.csv` |
| `soa_fused_module` | Unknown | **VALID** | SVD-Attention | 89.11% | 72.70% | 1.183x | **NONE** | `...ed_module/distilbert/summary.csv` |
| `soa_module` | Unknown | **VALID** | SVD-Attention | 89.11% | 72.70% | 1.182x | **NONE** | `...oa_module/distilbert/summary.csv` |
| `svd_compressed` | Attention Replacement | **VALID** | SVD-Attention | 87.96% | 57.10% | 1.220x | **NONE** | `...ompressed/distilbert/summary.csv` |
| `svd_linear_decay` | Attention Replacement | **VALID** | SVD-Attention | 87.96% | 53.90% | 1.218x | **NONE** | `...ied_sweep/distilbert/summary.csv` |
| `svd_proj_attn` | Unknown | **VALID** | SVD-Attention | 87.04% | 81.40% | 1.248x | **NONE** | `...proj_attn/distilbert/summary.csv` |
| `a3` | Unknown | **VALID** | Unknown | 92.55% | 89.30% | 1.103x | **NONE** | `...d/unified_a3_4models_summary.csv` |
| `adaptive_compressor` | Unknown | **VALID** | Unknown | 91.51% | 92.80% | 1.152x | **NONE** | `..._runs/MASTER_UNIFIED_SUMMARY.csv` |
| `aegis_hybrid` | Attention Replacement | **VALID** | Unknown | 87.27% | 89.50% | 1.225x | **NONE** | `...is_hybrid/distilbert/summary.csv` |
| `aegis_spectron_fp8` | Attention Replacement | **VALID** | Unknown | 88.19% | 34.00% | 1.244x | **NONE** | `...ctron_fp8/distilbert/summary.csv` |
| `aegis_v2` | Attention Replacement | **VALID** | Unknown | 91.86% | 58.10% | 1.225x | **NONE** | `..._aegis_v2/distilbert/summary.csv` |
| `aegis_v3` | Unknown | **VALID** | Unknown | 82.80% | 90.10% | 1.215x | **NONE** | `...ied_sweep/distilbert/summary.csv` |
| `aegis_v4` | Unknown | **VALID** | Unknown | 83.03% | 89.80% | 1.215x | **NONE** | `...ied_sweep/distilbert/summary.csv` |
| `aegis_v5` | Attention Replacement | **VALID** | Unknown | 86.70% | 89.80% | 1.201x | **NONE** | `...ied_sweep/distilbert/summary.csv` |
| `aegis_v6` | Attention Replacement | **VALID** | Unknown | 90.83% | 93.30% | 1.277x | **NONE** | `..._aegis_v6/distilbert/summary.csv` |
| `come` | Unknown | **VALID** | Unknown | 87.84% | 91.50% | 2.000x | **NONE** | `...fied_come/distilbert/summary.csv` |
| `compressed_attention` | Attention Replacement | **VALID** | Unknown | 89.68% | 94.40% | 1.260x | **NONE** | `...attention/distilbert/summary.csv` |
| `cur_moe` | Unknown | **VALID** | Unknown | 93.69% | 94.30% | 2.539x | **NONE** | `...d_cur_moe/distilbert/summary.csv` |
| `lola_attention` | Attention Replacement | **VALID** | Unknown | 87.16% | 59.00% | 1.251x | **NONE** | `...attention/distilbert/summary.csv` |
| `minimal_aurora` | Unknown | **VALID** | Unknown | 88.19% | 55.10% | 1.172x | **NONE** | `...al_aurora/distilbert/summary.csv` |
| `minimal_nova` | Attention Replacement | **VALID** | Unknown | 87.61% | 55.20% | 1.220x | **NONE** | `...imal_nova/distilbert/summary.csv` |
| `minimal_prism` | Unknown | **VALID** | Unknown | 87.61% | 55.20% | 1.172x | **NONE** | `...mal_prism/distilbert/summary.csv` |
| `orion` | Attention Replacement | **VALID** | Unknown | 79.47% | 38.40% | 1.251x | **NONE** | `...ied_orion/distilbert/summary.csv` |
| `phantom` | Attention Replacement | **VALID** | Unknown | 52.06% | 35.40% | 1.248x | **NONE** | `...d_phantom/distilbert/summary.csv` |
| `projection_factorized_attn` | Attention Replacement | **VALID** | Unknown | 87.96% | 68.80% | 1.252x | **NONE** | `...ized_attn/distilbert/summary.csv` |
| `rpca` | Unknown | **VALID** | Unknown | 94.38% | 82.00% | 30.960x | **NONE** | `...fied_rpca/distilbert/summary.csv` |
| `salt` | Unknown | **VALID** | Unknown | 88.88% | 65.60% | 1.264x | **NONE** | `...fied_salt/distilbert/summary.csv` |
| `sdrm` | Unknown | **VALID** | Unknown | 91.40% | 78.80% | 1.430x | **NONE** | `...fied_sdrm/distilbert/summary.csv` |
| `soka` | Unknown | **VALID** | Unknown | 92.66% | 82.10% | 59.910x | **NONE** | `...fied_soka/distilbert/summary.csv` |
| `spectron` | Attention Replacement | **VALID** | Unknown | 88.07% | 60.60% | 1.240x | **NONE** | `..._spectron/distilbert/summary.csv` |
| `vspec` | Attention Replacement | **VALID** | Unknown | 87.16% | 46.90% | 1.223x | **NONE** | `...ied_vspec/distilbert/summary.csv` |
| `gqa` | Unknown | **QUESTIONABLE** | GQA-MQA | 54.70% | 44.10% | 1.084x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `admm_compressor` | Compressor | **FALLBACK** | Compression-Failed | 91.86% | 94.30% | 1.000x | **P1** | `..._compressor/rank_128/summary.csv` |
| `admm_pruner` | Compressor | **FALLBACK** | Compression-Failed | 90.25% | 94.30% | 1.000x | **P1** | `...mm_pruner/distilbert/summary.csv` |
| `armor_pruner` | Compressor | **FALLBACK** | Compression-Failed | 91.06% | 93.70% | 1.000x | **P1** | `...s_archived/benchmark_summary.csv` |
| `ast_pruner` | Compressor | **FALLBACK** | Compression-Failed | 90.83% | 94.40% | 1.000x | **P1** | `...s_archived/benchmark_summary.csv` |
| `astro_quantizer` | Compressor | **FALLBACK** | Compression-Failed | 94.04% | 94.50% | 1.000x | **P1** | `...s_archived/benchmark_summary.csv` |
| `bonsai_pruner` | Compressor | **FALLBACK** | Compression-Failed | 93.35% | 80.90% | 1.000x | **P1** | `...s_archived/benchmark_summary.csv` |
| `cachenotes_compressor` | Compressor | **FALLBACK** | Compression-Failed | 92.55% | 94.40% | 1.000x | **P1** | `...s_archived/benchmark_summary.csv` |
| `coopq_quantizer` | Compressor | **FALLBACK** | Compression-Failed | 93.35% | 94.20% | 1.000x | **P1** | `...s_archived/benchmark_summary.csv` |
| `cr_compressor` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 94.70% | 30.960x | **P1** | `...p/summary_cr_compressor_mnli.csv` |
| `cur_compressor` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 94.70% | 30.750x | **P1** | `.../summary_cur_compressor_mnli.csv` |
| `delta_kv` | Compressor | **FALLBACK** | Compression-Failed | 90.94% | 95.00% | 1.000x | **P1** | `...ied_sweep/distilbert/summary.csv` |
| `direct_svd` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 94.40% | 30.960x | **P1** | `...ied_sweep/distilbert/summary.csv` |
| `dobi_svd` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 81.50% | 30.950x | **P1** | `...unified_dobi/qwen3_5/summary.csv` |
| `dynamic_rank` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 94.40% | 30.960x | **P1** | `...ied_sweep/distilbert/summary.csv` |
| `enlighten_compressor` | Compressor | **FALLBACK** | Compression-Failed | 90.94% | 93.80% | 1.000x | **P1** | `...enlighten_compressor/summary.csv` |
| `entropy_budget_allocation` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 81.50% | 30.960x | **P1** | `...ndoned_run/benchmark_summary.csv` |
| `erc_svd` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 78.60% | 30.960x | **P1** | `...ert/erc_svd/rank_128/summary.csv` |
| `faq_quantizer` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 94.20% | 1.000x | **P1** | `...s_archived/benchmark_summary.csv` |
| `flap` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 94.70% | 1.000x | **P1** | `...ied_sweep/distilbert/summary.csv` |
| `flat_llm` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 94.70% | 1.258x | **P1** | `...ied_sweep/distilbert/summary.csv` |
| `gfws_d_compressor` | Compressor | **FALLBACK** | Compression-Failed | 90.83% | 94.70% | 1.000x | **P1** | `...ummary_gfws_d_bert_sst2_r128.csv` |
| `hashed_net_compressor` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 94.60% | 691.880x | **P1** | `...y_hashed_net_compressor_mnli.csv` |
| `huffman_coder` | Compressor | **FALLBACK** | Compression-Failed | 94.38% | 81.50% | 1.000x | **P1** | `...man_coder/distilbert/summary.csv` |
| `kmeans_quantize` | Compressor | **FALLBACK** | Compression-Failed | 93.58% | 81.50% | 826.110x | **P1** | `...s_archived/benchmark_summary.csv` |
| `kronecker_tt_lora` | Compressor | **FALLBACK** | Compression-Failed | 93.81% | 81.80% | 1.000x | **P1** | `...r_tt_lora/distilbert/summary.csv` |
| `liftr` | Compressor | **FALLBACK** | Compression-Failed | 93.58% | 82.30% | 1.000x | **P1** | `...ied_liftr/distilbert/summary.csv` |
| `lillama_compressor` | Compressor | **FALLBACK** | Compression-Failed | 94.38% | 81.70% | 1.000x | **P1** | `...s_archived/benchmark_summary.csv` |
| `lsc_compressor` | Compressor | **FALLBACK** | Compression-Failed | 94.27% | 81.80% | 1.000x | **P1** | `.../summary_lsc_compressor_mnli.csv` |
| `mamba_shedder` | Compressor | **FALLBACK** | Compression-Failed | 94.38% | 81.70% | 1.044x | **P1** | `...a_shedder/distilbert/summary.csv` |
| `mlorq_compressor` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 81.80% | 3.150x | **P1** | `..._compressor/rank_128/summary.csv` |
| `nli_softmax` | Compressor | **FALLBACK** | Compression-Failed | 92.89% | 50.60% | 1.000x | **P1** | `...i_softmax/distilbert/summary.csv` |
| `nmf_compressor` | Compressor | **FALLBACK** | Compression-Failed | 94.38% | 82.30% | 30.960x | **P1** | `.../summary_nmf_compressor_mnli.csv` |
| `optimal_subspace_projection` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 82.00% | 1.000x | **P1** | `...e_projection/roberta/summary.csv` |
| `proxsparse` | Compressor | **FALLBACK** | Compression-Failed | 94.27% | 81.80% | 1.000x | **P1** | `...roxsparse/distilbert/summary.csv` |
| `replaceme_compressor` | Compressor | **FALLBACK** | Compression-Failed | 94.15% | 81.60% | 1.241x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `roast_hashed_linear` | Compressor | **FALLBACK** | Compression-Failed | 94.27% | 81.80% | 1.000x | **P1** | `...ed_linear/distilbert/summary.csv` |
| `rotational_slicegpt_int4` | Compressor | **FALLBACK** | Compression-Failed | 94.38% | 81.10% | 1.000x | **P1** | `...egpt_int4/distilbert/summary.csv` |
| `saes_svd_v2` | Compressor | **FALLBACK** | Compression-Failed | 94.38% | 81.90% | 30.960x | **P1** | `...es_svd_v2/distilbert/summary.csv` |
| `salr` | Compressor | **FALLBACK** | Compression-Failed | 94.38% | 82.00% | 30.960x | **P1** | `...fied_salr/distilbert/summary.csv` |
| `slicegpt` | Compressor | **FALLBACK** | Compression-Failed | 94.38% | 81.60% | 1.000x | **P1** | `..._slicegpt/distilbert/summary.csv` |
| `slicegpt_pca` | Compressor | **FALLBACK** | Compression-Failed | 94.27% | 81.80% | 1.000x | **P1** | `...cegpt_pca/distilbert/summary.csv` |
| `slim_compressor` | Compressor | **FALLBACK** | Compression-Failed | 91.51% | 81.10% | 1.000x | **P1** | `...l_121_compressors_gpu_matrix.csv` |
| `sparsegpt` | Compressor | **FALLBACK** | Compression-Failed | 92.66% | 80.40% | 1.000x | **P1** | `...sparsegpt/distilbert/summary.csv` |
| `uniql_compressor` | Compressor | **FALLBACK** | Compression-Failed | 94.38% | 82.40% | 1.000x | **P1** | `..._compressor/rank_128/summary.csv` |
| `wgqa` | Unknown | **FALLBACK** | GQA-MQA | 90.71% | 78.80% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `Galerkin` | Unknown | **FALLBACK** | Linear-Attention | 78.90% | 48.60% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `cosformer` | Attention Replacement | **FALLBACK** | Linear-Attention | 78.78% | 84.50% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `decay_linear_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | 77.75% | 34.80% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `galerkin_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | 78.90% | 48.60% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `gated_linear_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | 50.80% | 32.90% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `linformer_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | 94.04% | 77.20% | 1.201x | **P2** | `...s_archived/benchmark_summary.csv` |
| `nystrom_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | 94.27% | 81.90% | 1.195x | **P2** | `...s_archived/benchmark_summary.csv` |
| `performer_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | 78.90% | 38.40% | 1.195x | **P2** | `...s_archived/benchmark_summary.csv` |
| `svd_linear_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | 90.83% | 53.90% | 30.960x | **P2** | `...s_archived/benchmark_summary.csv` |
| `tlinformer_attention` | Attention Replacement | **FALLBACK** | Linear-Attention | 94.04% | 77.90% | 1.000x | **P2** | `...attention/distilbert/summary.csv` |
| `griffin_ssd` | Attention Replacement | **FALLBACK** | SSM | 91.97% | 92.60% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `hgrn2_gated_deltanet` | Attention Replacement | **FALLBACK** | SSM | 0.00% | 33.60% | 1.000x | **P2** | `...tilbert/summary_buggy_backup.csv` |
| `hgrn2_layer` | Attention Replacement | **FALLBACK** | SSM | 51.72% | 34.40% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `hgrn2_outer` | Attention Replacement | **FALLBACK** | SSM | 54.93% | 34.40% | 1.000x | **P2** | `...rn2_outer/distilbert/summary.csv` |
| `hymba_transx` | Attention Replacement | **FALLBACK** | SSM | 83.37% | 89.10% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `jamba` | Attention Replacement | **FALLBACK** | SSM | 87.61% | 91.30% | 1.000x | **P2** | `...ied_jamba/distilbert/summary.csv` |
| `jamba_info` | Attention Replacement | **FALLBACK** | SSM | 89.79% | 62.10% | 1.000x | **P2** | `...baselines/distilbert/summary.csv` |
| `mamba` | Unknown | **FALLBACK** | SSM | 50.92% | 34.10% | 1.054x | **P2** | `...ndoned_run/benchmark_summary.csv` |
| `mamba_inject` | Unknown | **FALLBACK** | SSM | 50.92% | 0.00% | 1.050x | **P2** | `...ba_inject/distilbert/summary.csv` |
| `mamba_svd_flash` | Attention Replacement | **FALLBACK** | SSM | 91.86% | 61.60% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `samba_m3` | Attention Replacement | **FALLBACK** | SSM | 91.86% | 70.20% | 1.000x | **P2** | `...d_sambam3/distilbert/summary.csv` |
| `samba_moe` | Attention Replacement | **FALLBACK** | SSM | 91.63% | 33.80% | 1.000x | **P2** | `..._sambamoe/distilbert/summary.csv` |
| `simple_ssm` | Attention Replacement | **FALLBACK** | SSM | 50.92% | 35.10% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `spectral_mamba` | Attention Replacement | **FALLBACK** | SSM | 91.51% | 59.10% | 1.000x | **P2** | `...ral_mamba/distilbert/summary.csv` |
| `svd_mamba` | Attention Replacement | **FALLBACK** | SSM | 90.02% | 34.50% | 1.000x | **P2** | `...svd_mamba/distilbert/summary.csv` |
| `upi_mamba` | Unknown | **FALLBACK** | SSM | 50.92% | 31.90% | 1.042x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `upi_stuffed_mamba` | Attention Replacement | **FALLBACK** | SSM | 50.92% | N/A | 1.042x | **P2** | `...s_archived/benchmark_summary.csv` |
| `asvd` | Unknown | **FALLBACK** | SVD-Attention | 90.71% | 94.60% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `saes_svd` | Unknown | **FALLBACK** | SVD-Attention | 94.38% | 81.90% | 30.960x | **P2** | `..._sweep/summary_saes_svd_mnli.csv` |
| `svd_attention` | Attention Replacement | **FALLBACK** | SVD-Attention | 90.71% | 81.40% | 1.015x | **P2** | `...s_archived/benchmark_summary.csv` |
| `svd_llm_v2_attention` | Attention Replacement | **FALLBACK** | SVD-Attention | 87.73% | 46.20% | 1.248x | **P2** | `...s_archived/benchmark_summary.csv` |
| `svd_projection_attention` | Attention Replacement | **FALLBACK** | SVD-Attention | 87.04% | 34.40% | 1.201x | **P2** | `...s_archived/benchmark_summary.csv` |
| `svd_sequence_attention` | Unknown | **FALLBACK** | SVD-Attention | 90.83% | 0.00% | 1.000x | **P2** | `..._svd_sequence_attention_sst2.csv` |
| `AdaptiveSparseAttention` | Attention Replacement | **FALLBACK** | Sparse-Attention | 92.20% | 36.40% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `HALO` | Attention Replacement | **FALLBACK** | Sparse-Attention | 93.58% | 94.40% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `adaptive_sparse_attention` | Attention Replacement | **FALLBACK** | Sparse-Attention | 92.20% | 36.40% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `halo_regla` | Attention Replacement | **FALLBACK** | Sparse-Attention | 93.58% | 94.40% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `mla_attention` | Attention Replacement | **FALLBACK** | Sparse-Attention | 57.68% | 35.40% | 1.114x | **P2** | `...p/summary_mla_attention_sst2.csv` |
| `moba` | Unknown | **FALLBACK** | Sparse-Attention | 92.89% | 36.70% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `nsa` | Unknown | **FALLBACK** | Sparse-Attention | 90.60% | 81.70% | 1.000x | **P2** | `...ified_nsa/distilbert/summary.csv` |
| `proxsparse_pruner` | Unknown | **FALLBACK** | Sparse-Attention | 94.27% | 81.80% | 1.000x | **P2** | `...se_pruner/distilbert/summary.csv` |
| `AdaptiveSpanAttention` | Attention Replacement | **FALLBACK** | Unknown | 94.15% | 94.20% | 1.017x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `OH` | Attention Replacement | **FALLBACK** | Unknown | 94.15% | 80.50% | 1.000x | **P2** | `...ified_oha/distilbert/summary.csv` |
| `Primal` | Unknown | **FALLBACK** | Unknown | 91.40% | 82.00% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `SqueezingHeadsAttentionDistillationLayer` | Unknown | **FALLBACK** | Unknown | 93.00% | 80.50% | 1.000x | **P2** | `...s_distill/distilbert/summary.csv` |
| `adaptive_span_attention` | Attention Replacement | **FALLBACK** | Unknown | 94.15% | 94.20% | 1.017x | **P2** | `...adaptive_span_attention_mnli.csv` |
| `adasplash` | Unknown | **FALLBACK** | Unknown | 89.45% | 94.20% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `aegis_v1` | Unknown | **FALLBACK** | Unknown | 90.14% | 94.10% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `bam_umoe` | Attention Replacement | **FALLBACK** | Unknown | 90.60% | 38.70% | 1.166x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `block_transformer_hiba` | Attention Replacement | **FALLBACK** | Unknown | 90.02% | 94.10% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `chela_slay` | Attention Replacement | **FALLBACK** | Unknown | 92.89% | 94.30% | 1.030x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `cluster` | Unknown | **FALLBACK** | Unknown | 90.14% | 93.40% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `dobi` | Unknown | **FALLBACK** | Unknown | 94.15% | 94.30% | 30.950x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `dynamic_entropy` | Unknown | **FALLBACK** | Unknown | 94.15% | 81.50% | 30.960x | **P2** | `...summary_dynamic_entropy_mnli.csv` |
| `eat` | Attention Replacement | **FALLBACK** | Unknown | 90.02% | 94.30% | 1.030x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `expert_choice_moe` | Attention Replacement | **FALLBACK** | Unknown | 90.14% | 38.70% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `factorised_matrix_state` | Attention Replacement | **FALLBACK** | Unknown | 77.29% | 33.70% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `flash3` | Unknown | **FALLBACK** | Unknown | 90.60% | 94.20% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `flash_attention3` | Attention Replacement | **FALLBACK** | Unknown | 92.78% | 50.40% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `flash_attention3_fp8` | Attention Replacement | **FALLBACK** | Unknown | 92.55% | 50.40% | 1.030x | **P2** | `...ry_flash_attention3_fp8_mnli.csv` |
| `frandi` | Unknown | **FALLBACK** | Unknown | 94.61% | 94.80% | 30.960x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `full_attention` | Attention Replacement | **FALLBACK** | Unknown | 90.83% | 35.50% | 1.006x | **P2** | `.../benchmark_missing_results_1.csv` |
| `fusion` | Attention Replacement | **FALLBACK** | Unknown | 90.02% | 94.20% | 1.033x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `fusion_attn` | Attention Replacement | **FALLBACK** | Unknown | 90.02% | 94.20% | 1.033x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `gated_delta_net` | Attention Replacement | **FALLBACK** | Unknown | 49.08% | 34.00% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `gated_deltanet_block` | Attention Replacement | **FALLBACK** | Unknown | 51.15% | 34.00% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `goldfinch_ahn` | Attention Replacement | **FALLBACK** | Unknown | 94.04% | 79.20% | 1.045x | **P2** | `...baselines/distilbert/summary.csv` |
| `granite_hybrid` | Attention Replacement | **FALLBACK** | Unknown | 88.88% | 48.00% | 1.000x | **P2** | `...te_hybrid/distilbert/summary.csv` |
| `grecodec` | Unknown | **FALLBACK** | Unknown | 94.15% | 94.80% | 30.960x | **P2** | `..._grecodec/distilbert/summary.csv` |
| `hamming_attention` | Attention Replacement | **FALLBACK** | Unknown | 82.22% | 89.70% | 1.195x | **P2** | `...s_archived/benchmark_summary.csv` |
| `hashed_linear` | Unknown | **FALLBACK** | Unknown | 94.15% | 81.50% | 1.000x | **P2** | `...ed_linear/distilbert/summary.csv` |
| `hla_attention` | Attention Replacement | **FALLBACK** | Unknown | 75.92% | 84.80% | 1.006x | **P2** | `.../benchmark_missing_results_1.csv` |
| `hybridkv` | Unknown | **FALLBACK** | Unknown | 94.15% | 94.30% | 1.000x | **P2** | `..._hybridkv/distilbert/summary.csv` |
| `infini` | Unknown | **FALLBACK** | Unknown | 90.60% | 94.30% | 1.000x | **P2** | `...ed_infini/distilbert/summary.csv` |
| `kda_attention` | Attention Replacement | **FALLBACK** | Unknown | 77.29% | 85.70% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `kronecker_linear` | Unknown | **FALLBACK** | Unknown | 92.32% | 80.10% | 1.039x | **P2** | `...er_linear/distilbert/summary.csv` |
| `kvtc` | Unknown | **FALLBACK** | Unknown | 94.15% | 81.50% | 3.150x | **P2** | `...fied_kvtc/distilbert/summary.csv` |
| `leka_v1` | Attention Replacement | **FALLBACK** | Unknown | 79.59% | 38.30% | 1.220x | **P2** | `...s_archived/benchmark_summary.csv` |
| `leka_v2` | Attention Replacement | **FALLBACK** | Unknown | 94.15% | 80.30% | 1.209x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `leka_v3` | Attention Replacement | **FALLBACK** | Unknown | 69.38% | 36.70% | 1.210x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `lora_attention` | Attention Replacement | **FALLBACK** | Unknown | 92.89% | 50.30% | 1.009x | **P2** | `..._bs16/neuralops_mrpc_results.csv` |
| `lsh_attention` | Attention Replacement | **FALLBACK** | Unknown | 83.26% | 44.40% | 1.220x | **P2** | `...attention/distilbert/summary.csv` |
| `mixattention_moska` | Attention Replacement | **FALLBACK** | Unknown | 92.89% | 80.70% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `modegpt` | Unknown | **FALLBACK** | Unknown | 94.27% | 81.80% | 2.543x | **P2** | `...d_modegpt/distilbert/summary.csv` |
| `moe_deltakv_paged` | Attention Replacement | **FALLBACK** | Unknown | 50.92% | 37.20% | 1.007x | **P2** | `...s_archived/benchmark_summary.csv` |
| `nasc_attention` | Attention Replacement | **FALLBACK** | Unknown | 89.22% | 68.80% | 1.000x | **P2** | `...reening/run_quality_/summary.csv` |
| `nexus` | Attention Replacement | **FALLBACK** | Unknown | 93.35% | 81.00% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `nha` | Unknown | **FALLBACK** | Unknown | 90.60% | 80.90% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `oats` | Unknown | **FALLBACK** | Unknown | 93.23% | 82.10% | 1.000x | **P2** | `...fied_oats/distilbert/summary.csv` |
| `optimized_hybrid_attention` | Attention Replacement | **FALLBACK** | Unknown | 94.15% | 80.50% | 1.000x | **P2** | `...attention/distilbert/summary.csv` |
| `padre_attention` | Attention Replacement | **FALLBACK** | Unknown | 92.66% | 77.70% | 1.000x | **P2** | `...summary_padre_attention_mnli.csv` |
| `progressive_token_pruning` | Attention Replacement | **FALLBACK** | Unknown | 94.38% | 41.10% | 1.000x | **P2** | `...baselines/distilbert/summary.csv` |
| `ptp` | Unknown | **FALLBACK** | Unknown | 90.60% | 82.00% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `qcqa_attention` | Attention Replacement | **FALLBACK** | Unknown | 55.62% | 37.00% | 1.073x | **P2** | `...fied_qcqa/distilbert/summary.csv` |
| `rala_attention` | Attention Replacement | **FALLBACK** | Unknown | 78.56% | 36.90% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `reap_moe` | Attention Replacement | **FALLBACK** | Unknown | 90.02% | 77.30% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `regla_attention` | Attention Replacement | **FALLBACK** | Unknown | 81.77% | 43.90% | 1.000x | **P2** | `...s_archived/benchmark_summary.csv` |
| `replace` | Attention Replacement | **FALLBACK** | Unknown | 90.94% | 80.50% | 1.214x | **P2** | `..._modules/replace_r64/summary.csv` |
| `routing_attention` | Attention Replacement | **FALLBACK** | Unknown | 55.50% | 36.80% | 1.220x | **P2** | `...g/run_quality_resume/summary.csv` |
| `rpc` | Unknown | **FALLBACK** | Unknown | 91.63% | 76.20% | 1.220x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `smart_weighted_ensemble` | Unknown | **FALLBACK** | Unknown | 69.61% | 53.20% | 1.000x | **P2** | `..._ensemble/distilbert/summary.csv` |
| `span_expanded_attention` | Attention Replacement | **FALLBACK** | Unknown | 94.38% | 81.90% | 1.000x | **P2** | `...span_expanded_attention_mnli.csv` |
| `spectra_v2` | Attention Replacement | **FALLBACK** | Unknown | 90.02% | 81.70% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `spectraformer_attention` | Attention Replacement | **FALLBACK** | Unknown | 78.10% | 0.00% | 1.000x | **P2** | `...attention/distilbert/summary.csv` |
| `squeezing_heads` | Unknown | **FALLBACK** | Unknown | 93.00% | 80.50% | 1.000x | **P2** | `...ing_heads/distilbert/summary.csv` |
| `squeezing_heads_distill` | Attention Replacement | **FALLBACK** | Unknown | 93.00% | 0.00% | 1.000x | **P2** | `...s_distill/distilbert/summary.csv` |
| `squeezing_heads_distiller` | Unknown | **FALLBACK** | Unknown | 93.00% | 40.00% | 1.000x | **P2** | `...distiller/distilbert/summary.csv` |
| `template_attention` | Attention Replacement | **FALLBACK** | Unknown | 93.12% | 58.10% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `token_compressed_attention` | Attention Replacement | **FALLBACK** | Unknown | 90.60% | 50.10% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `tomoe` | Unknown | **FALLBACK** | Unknown | 94.04% | 82.00% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `tpa_attention` | Attention Replacement | **FALLBACK** | Unknown | 77.98% | 36.80% | 1.081x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `tt` | Unknown | **FALLBACK** | Unknown | 59.06% | 81.40% | 1.024x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `unified_rope` | Attention Replacement | **FALLBACK** | Unknown | 88.88% | 64.70% | 1.000x | **P2** | `...fied_rope/distilbert/summary.csv` |
| `vig_attention` | Attention Replacement | **FALLBACK** | Unknown | 90.48% | 64.80% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `weighted_ensemble` | Unknown | **FALLBACK** | Unknown | 90.83% | 79.20% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `xattention` | Attention Replacement | **FALLBACK** | Unknown | 86.70% | 52.50% | 1.000x | **P2** | `...ied_sweep/distilbert/summary.csv` |
| `zerokv_neo` | Unknown | **FALLBACK** | Unknown | 90.48% | N/A | 1.000x | **P2** | `...erokv_neo/distilbert/summary.csv` |
| `zeros_attention` | Attention Replacement | **FALLBACK** | Unknown | 61.24% | 36.30% | 1.000x | **P2** | `...d/unified_zeros/gpt2/summary.csv` |
| `adasvd_compressor` | Compressor | **BROKEN** | Compression-Failed | 88.76% | 87.40% | 27.220x | **P0** | `...rank_roberta/all_rank_curves.csv` |
| `compot_compressor` | Compressor | **BROKEN** | Compression-Failed | 93.69% | 94.50% | 30.960x | **P0** | `...rank_roberta/all_rank_curves.csv` |
| `cur_linear` | Unknown | **BROKEN** | Unknown | 94.15% | 81.50% | 1.000x | **P0** | `...weep/summary_cur_linear_mnli.csv` |
| `cure` | Unknown | **BROKEN** | Unknown | 94.15% | 82.00% | 1.000x | **P0** | `...seed/multiseed_rigor_summary.csv` |
| `pruning_utils` | Utility | **NO_DATA** | Utility | 0.00% | 0.00% | 1.000x | **P3** | `...p/summary_pruning_utils_mnli.csv` |
