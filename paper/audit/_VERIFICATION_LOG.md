# Module Verification Log
Generated: 2026-09-21T10:19:14.041801
Total modules: 217

| # | Module | Classification | Accuracy | Compression | Fabricated? | CSV Source |
|---|--------|----------------|----------|-------------|-------------|------------|
| 1 | `AdaptiveSVD` | **VALID** | 100.00% | 1.216x | YES (4 vals) | `results/long_context/summary.csv` |
| 2 | `AdaptiveSpanAttention` | **FALLBACK** | 91.74% | 1.000x | no | `results/roberta/summary.csv` |
| 3 | `AdaptiveSparseAttention` | **FALLBACK** | 91.86% | 1.000x | YES (4 vals) | `results/roberta/summary.csv` |
| 4 | `Galerkin` | **FALLBACK** | 75.57% | 1.000x | YES (6 vals) | `.../rank_ablation/lora0_sst2/summary.csv` |
| 5 | `HALO` | **FALLBACK** | 90.94% | 1.000x | no | `results/roberta/summary.csv` |
| 6 | `OH` | **FALLBACK** | 90.48% | 1.000x | YES (1 vals) | `...ts/unified_oha/distilbert/summary.csv` |
| 7 | `Primal` | **FALLBACK** | 82.00% | 1.000x | YES (3 vals) | `...t_mnli_full_mnli_sweep/summary_v3.csv` |
| 8 | `SOAFusedAttention` | **VALID** | 76.30% | 1.145x | YES (2 vals) | `...fusedattention/distilbert/summary.csv` |
| 9 | `SqueezingHeadsAttentionDistillationLayer` | **FALLBACK** | 94.30% | 1.000x | no | `...ep/distilbert_agnews_full/summary.csv` |
| 10 | `a3` | **VALID** | 100.00% | 1.094x | no | `results/long_context/summary.csv` |
| 11 | `aa_svd_compressor` | **VALID** | 85.80% | 1.133x | no | `...svd_compressor/distilbert/summary.csv` |
| 12 | `aasvd` | **VALID** | 94.50% | 1.016x | no | `...nk_ablation/agnews_rank32/summary.csv` |
| 13 | `adaptive_compressor` | **VALID** | 84.50% | 1.002x | YES (1 vals) | `..._adaptive_compressor/gpt2/summary.csv` |
| 14 | `adaptive_ranks` | **VALID** | 84.50% | 1.002x | no | `...ified_adaptive_ranks/gpt2/summary.csv` |
| 15 | `adaptive_span_attention` | **FALLBACK** | 82.00% | 1.000x | YES (4 vals) | `...ptive_span_attention/gpt2/summary.csv` |
| 16 | `adaptive_sparse_attention` | **FALLBACK** | 62.50% | 1.000x | YES (6 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 17 | `adaptive_svd_attention` | **VALID** | 74.60% | 1.152x | YES (6 vals) | `..._svd_attention/distilbert/summary.csv` |
| 18 | `adasplash` | **FALLBACK** | 93.00% | 1.000x | no | `...ank_ablation/lora0_agnews/summary.csv` |
| 19 | `adasvd_compressor` | **BROKEN** | 0.00% | 1.000x | YES (7 vals) | `...ed_adasvd_compressor/gpt2/summary.csv` |
| 20 | `admm_compressor` | **FALLBACK** | 81.30% | 1.000x | YES (6 vals) | `...d_admm_compressor/roberta/summary.csv` |
| 21 | `admm_pruner` | **FALLBACK** | 91.60% | 1.000x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 22 | `aegis_hybrid` | **VALID** | 93.00% | 1.050x | no | `...k_ablation/agnews_rank256/summary.csv` |
| 23 | `aegis_spectron_fp8` | **VALID** | 79.59% | 1.223x | no | `...s_spectron_fp8/distilbert/summary.csv` |
| 24 | `aegis_v1` | **FALLBACK** | 87.04% | 1.000x | YES (1 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 25 | `aegis_v2` | **VALID** | 84.17% | 1.181x | YES (3 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 26 | `aegis_v3` | **VALID** | 90.10% | 1.197x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 27 | `aegis_v4` | **VALID** | 89.80% | 1.197x | YES (1 vals) | `...weep/distilbert_mnli_full/summary.csv` |
| 28 | `aegis_v5` | **VALID** | 89.80% | 1.197x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 29 | `aegis_v6` | **VALID** | 93.30% | 1.252x | no | `...nk_ablation/agnews_rank64/summary.csv` |
| 30 | `armor_pruner` | **FALLBACK** | 62.50% | 1.000x | YES (1 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 31 | `ast_pruner` | **FALLBACK** | 68.38% | 1.000x | no | `...lts/adaptive_v3/benchmark_summary.csv` |
| 32 | `astro_quantizer` | **FALLBACK** | 83.30% | 1.000x | YES (8 vals) | `...d_astro_quantizer/roberta/summary.csv` |
| 33 | `asvd` | **FALLBACK** | 100.00% | 0.964x | no | `results/long_context/summary.csv` |
| 34 | `bam_umoe` | **FALLBACK** | 90.60% | 1.000x | no | `...p/distilbert/summary_buggy_backup.csv` |
| 35 | `block_transformer_hiba` | **FALLBACK** | 94.60% | 1.000x | no | `...ank_ablation/lora0_agnews/summary.csv` |
| 36 | `bonsai_pruner` | **FALLBACK** | 50.00% | 1.000x | no | `...r/abandoned_run/benchmark_summary.csv` |
| 37 | `cachenotes_compressor` | **FALLBACK** | 81.30% | 1.000x | YES (6 vals) | `...achenotes_compressor/gpt2/summary.csv` |
| 38 | `chela_slay` | **FALLBACK** | 90.60% | 1.000x | no | `...p/distilbert/summary_buggy_backup.csv` |
| 39 | `cluster` | **FALLBACK** | 90.48% | 1.000x | no | `...p/distilbert/summary_buggy_backup.csv` |
| 40 | `come` | **VALID** | 91.50% | 1.241x | YES (1 vals) | `...weep/distilbert_mnli_full/summary.csv` |
| 41 | `compot_compressor` | **BROKEN** | 82.30% | 1.000x | no | `...compot_compressor/roberta/summary.csv` |
| 42 | `compressed_attention` | **VALID** | 78.60% | 1.110x | YES (1 vals) | `...weep/summary_snapshot_post_group7.csv` |
| 43 | `compressed_mha` | **VALID** | 76.60% | 1.110x | no | `...weep/summary_snapshot_post_group7.csv` |
| 44 | `coopq_quantizer` | **FALLBACK** | 50.00% | 1.000x | no | `...r/abandoned_run/benchmark_summary.csv` |
| 45 | `cosformer` | **FALLBACK** | 100.00% | 0.992x | YES (6 vals) | `results/long_context/summary.csv` |
| 46 | `cr_compressor` | **FALLBACK** | 94.70% | 1.000x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 47 | `cur_compressor` | **FALLBACK** | 94.70% | 1.000x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 48 | `cur_linear` | **BROKEN** | 90.71% | 1.000x | YES (1 vals) | `...ied_cur_linear/distilbert/summary.csv` |
| 49 | `cur_moe` | **VALID** | 94.30% | 1.056x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 50 | `cure` | **BROKEN** | 90.71% | 1.000x | YES (1 vals) | `...s/unified_cure/distilbert/summary.csv` |
| 51 | `decay_linear_attention` | **FALLBACK** | 62.50% | 1.000x | YES (3 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 52 | `delta_kv` | **FALLBACK** | 95.00% | 1.000x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 53 | `direct_svd` | **FALLBACK** | 94.40% | 1.000x | YES (2 vals) | `...weep/distilbert_mnli_full/summary.csv` |
| 54 | `dobi` | **FALLBACK** | 94.30% | 1.000x | YES (2 vals) | `...weep/distilbert_mnli_full/summary.csv` |
| 55 | `dobi_svd` | **FALLBACK** | 89.45% | 1.000x | no | `.../unified_dobi_svd/qwen3_5/summary.csv` |
| 56 | `dynamic_entropy` | **FALLBACK** | 0.00% | 1.000x | YES (4 vals) | `...inal_paper_run/sst2_plotting_data.csv` |
| 57 | `dynamic_rank` | **FALLBACK** | 94.40% | 1.000x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 58 | `eat` | **FALLBACK** | 91.10% | 1.000x | no | `...ank_ablation/lora0_agnews/summary.csv` |
| 59 | `enlighten_compressor` | **FALLBACK** | 93.80% | 1.000x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 60 | `entropy_budget_allocation` | **FALLBACK** | 68.38% | 1.000x | no | `...lts/adaptive_v3/benchmark_summary.csv` |
| 61 | `erc_svd` | **FALLBACK** | 62.50% | 1.000x | no | `...r/abandoned_run/benchmark_summary.csv` |
| 62 | `expert_choice_moe` | **FALLBACK** | 90.60% | 1.000x | no | `...p/distilbert/summary_buggy_backup.csv` |
| 63 | `factorised_matrix_state` | **FALLBACK** | 75.00% | 1.000x | YES (3 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 64 | `faq_quantizer` | **FALLBACK** | 62.50% | 1.000x | YES (6 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 65 | `flap` | **FALLBACK** | 94.70% | 1.000x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 66 | `flash3` | **FALLBACK** | 91.10% | 1.000x | YES (1 vals) | `...ank_ablation/lora0_agnews/summary.csv` |
| 67 | `flash_attention3` | **FALLBACK** | 68.38% | 1.000x | YES (3 vals) | `...lts/adaptive_v3/benchmark_summary.csv` |
| 68 | `flash_attention3_fp8` | **FALLBACK** | 0.00% | 1.000x | YES (14 vals) | `...inal_paper_run/sst2_plotting_data.csv` |
| 69 | `flat_llm` | **FALLBACK** | 94.70% | 1.000x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 70 | `frandi` | **FALLBACK** | 94.80% | 1.000x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 71 | `full_attention` | **FALLBACK** | 35.20% | 1.000x | YES (3 vals) | `...t_mnli_full_mnli_sweep/summary_v3.csv` |
| 72 | `fused_svd_kernel` | **VALID** | 91.90% | 1.076x | no | `...k_ablation/agnews_rank256/summary.csv` |
| 73 | `fusion` | **FALLBACK** | 93.12% | 1.000x | no | `results/roberta/summary.csv` |
| 74 | `fusion_attn` | **FALLBACK** | 93.12% | 1.000x | no | `results/roberta/summary.csv` |
| 75 | `galerkin_attention` | **FALLBACK** | 68.63% | 1.000x | YES (2 vals) | `...lts/adaptive_v3/benchmark_summary.csv` |
| 76 | `gated_delta_net` | **FALLBACK** | 70.30% | 1.000x | YES (5 vals) | `results/distilbert/summary.csv` |
| 77 | `gated_deltanet_block` | **FALLBACK** | 75.11% | 1.000x | YES (7 vals) | `results/distilbert/summary.csv` |
| 78 | `gated_linear_attention` | **FALLBACK** | 62.50% | 1.000x | YES (2 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 79 | `gfws_d_compressor` | **FALLBACK** | 94.70% | 1.000x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 80 | `goldfinch_ahn` | **FALLBACK** | 91.17% | 1.000x | no | `results/roberta/summary.csv` |
| 81 | `gqa` | **QUESTIONABLE** | 44.10% | 1.078x | YES (7 vals) | `...weep/distilbert_mnli_full/summary.csv` |
| 82 | `granite_hybrid` | **FALLBACK** | 90.14% | 1.000x | YES (2 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 83 | `grecodec` | **FALLBACK** | 94.80% | 1.000x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 84 | `griffin_ssd` | **FALLBACK** | 90.14% | 1.000x | YES (2 vals) | `results/roberta/summary.csv` |
| 85 | `halo_regla` | **FALLBACK** | 90.94% | 1.000x | no | `results/roberta/summary.csv` |
| 86 | `hamming_attention` | **FALLBACK** | 68.38% | 1.000x | YES (5 vals) | `...lts/adaptive_v3/benchmark_summary.csv` |
| 87 | `hashed_linear` | **FALLBACK** | 89.68% | 1.000x | YES (3 vals) | `..._hashed_linear/distilbert/summary.csv` |
| 88 | `hashed_net_compressor` | **FALLBACK** | 94.60% | 1.000x | no | `...weep/distilbert_mnli_full/summary.csv` |
| 89 | `hgrn2_gated_deltanet` | **FALLBACK** | 62.50% | 1.000x | YES (3 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 90 | `hgrn2_layer` | **FALLBACK** | 68.38% | 1.000x | YES (3 vals) | `...lts/adaptive_v3/benchmark_summary.csv` |
| 91 | `hgrn2_outer` | **FALLBACK** | 72.36% | 1.000x | YES (2 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 92 | `hla_attention` | **FALLBACK** | 0.00% | 1.000x | YES (6 vals) | `...t_2026_08_26/benchmark_summary_v2.csv` |
| 93 | `huffman_coder` | **FALLBACK** | 90.60% | 1.000x | YES (3 vals) | `..._huffman_coder/distilbert/summary.csv` |
| 94 | `hybrid_svd_layer` | **VALID** | 77.18% | 1.907x | YES (3 vals) | `...brid_svd_layer/distilbert/summary.csv` |
| 95 | `hybridkv` | **FALLBACK** | 94.30% | 1.000x | YES (1 vals) | `...weep/distilbert_mnli_full/summary.csv` |
| 96 | `hymba_transx` | **FALLBACK** | 88.65% | 1.000x | YES (2 vals) | `results/distilbert/summary.csv` |
| 97 | `infini` | **FALLBACK** | 90.60% | 1.000x | YES (6 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 98 | `jamba` | **FALLBACK** | 88.99% | 1.000x | YES (1 vals) | `results/distilbert/summary.csv` |
| 99 | `jamba_info` | **FALLBACK** | 89.45% | 1.000x | YES (2 vals) | `results/roberta/summary.csv` |
| 100 | `kda_attention` | **FALLBACK** | 62.50% | 1.000x | YES (4 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 101 | `kmeans_quantize` | **FALLBACK** | 50.00% | 1.000x | no | `...r/abandoned_run/benchmark_summary.csv` |
| 102 | `kronecker_compressor` | **VALID** | 90.94% | 1.003x | no | `...fied_kronecker/distilbert/summary.csv` |
| 103 | `kronecker_linear` | **FALLBACK** | 90.71% | 1.000x | YES (1 vals) | `...onecker_linear/distilbert/summary.csv` |
| 104 | `kronecker_tt_lora` | **FALLBACK** | 90.48% | 0.978x | no | `...necker_tt_lora/distilbert/summary.csv` |
| 105 | `kvtc` | **FALLBACK** | 94.20% | 1.000x | no | `...ep/distilbert_agnews_full/summary.csv` |
| 106 | `leka_v1` | **FALLBACK** | 69.13% | 1.000x | YES (3 vals) | `results/t5_cola_parallel/summary.csv` |
| 107 | `leka_v2` | **FALLBACK** | 90.37% | 1.000x | no | `...p/distilbert/summary_buggy_backup.csv` |
| 108 | `leka_v3` | **FALLBACK** | 80.05% | 1.000x | YES (4 vals) | `results/distilbert/summary.csv` |
| 109 | `liftr` | **FALLBACK** | 94.90% | 1.000x | YES (1 vals) | `...ep/distilbert_agnews_full/summary.csv` |
| 110 | `lillama_compressor` | **FALLBACK** | 62.50% | 1.000x | YES (5 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 111 | `linformer_attention` | **FALLBACK** | 77.20% | 1.000x | YES (2 vals) | `...t_mnli_full_mnli_sweep/summary_v2.csv` |
| 112 | `lola_attention` | **VALID** | 77.98% | 1.227x | YES (1 vals) | `...lola_attention/distilbert/summary.csv` |
| 113 | `lora_attention` | **FALLBACK** | 0.00% | 1.000x | no | `...t_2026_08_26/benchmark_summary_v2.csv` |
| 114 | `lsc_compressor` | **FALLBACK** | 0.00% | 1.000x | YES (6 vals) | `...ts/all_121_compressors_gpu_matrix.csv` |
| 115 | `lsh_attention` | **FALLBACK** | 80.62% | 0.991x | YES (4 vals) | `...nified_lsh_attention/gpt2/summary.csv` |
| 116 | `mamba` | **FALLBACK** | 34.10% | 1.000x | YES (3 vals) | `...t_mnli_full_mnli_sweep/summary_v2.csv` |
| 117 | `mamba_inject` | **FALLBACK** | 0.00% | 1.000x | YES (3 vals) | `results/master_aggregated_results.csv` |
| 118 | `mamba_shedder` | **FALLBACK** | 94.80% | 1.000x | YES (1 vals) | `...ep/distilbert_agnews_full/summary.csv` |
| 119 | `mamba_svd_flash` | **FALLBACK** | 62.50% | 1.000x | YES (3 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 120 | `mgie_svd` | **VALID** | 60.09% | 1.003x | YES (1 vals) | `...lts/unified_mgie_svd/gpt2/summary.csv` |
| 121 | `minimal_aurora` | **VALID** | 88.30% | 1.152x | YES (14 vals) | `...nk_ablation/agnews_rank64/summary.csv` |
| 122 | `minimal_nova` | **VALID** | 88.30% | 1.152x | YES (5 vals) | `...s/rank_ablation/ablation_all_rows.csv` |
| 123 | `minimal_prism` | **VALID** | 88.30% | 1.152x | YES (5 vals) | `...s/rank_ablation/ablation_all_rows.csv` |
| 124 | `minimal_svd` | **VALID** | 88.30% | 1.152x | YES (5 vals) | `...nk_ablation/agnews_rank64/summary.csv` |
| 125 | `minimal_svd_v2` | **VALID** | 88.30% | 1.152x | YES (3 vals) | `...nk_ablation/agnews_rank64/summary.csv` |
| 126 | `mixattention_moska` | **FALLBACK** | 89.33% | 1.000x | YES (4 vals) | `results/distilbert/summary.csv` |
| 127 | `mla_attention` | **FALLBACK** | 34.70% | 1.000x | YES (6 vals) | `...t_mnli_full_mnli_sweep/summary_v2.csv` |
| 128 | `mlorq_compressor` | **FALLBACK** | 90.48% | 1.000x | YES (1 vals) | `.../unified_mlorq/distilbert/summary.csv` |
| 129 | `moba` | **FALLBACK** | 90.71% | 1.000x | YES (7 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 130 | `modegpt` | **FALLBACK** | 91.06% | 1.000x | YES (1 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 131 | `moe_deltakv_paged` | **FALLBACK** | 72.48% | 1.000x | YES (3 vals) | `results/distilbert/summary.csv` |
| 132 | `nasc_attention` | **FALLBACK** | 0.00% | 1.000x | YES (5 vals) | `...n_quality_20260826_175056/summary.csv` |
| 133 | `nexus` | **FALLBACK** | 80.90% | 1.000x | no | `...t_mnli_full_mnli_sweep/summary_v3.csv` |
| 134 | `nha` | **FALLBACK** | 90.48% | 1.000x | no | `...p/distilbert/summary_buggy_backup.csv` |
| 135 | `nli_softmax` | **FALLBACK** | 90.14% | 1.000x | YES (2 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 136 | `nmf_compressor` | **FALLBACK** | 0.00% | 1.000x | YES (6 vals) | `...ts/all_121_compressors_gpu_matrix.csv` |
| 137 | `nsa` | **FALLBACK** | 94.60% | 1.000x | YES (4 vals) | `...ank_ablation/lora0_agnews/summary.csv` |
| 138 | `nystrom_attention` | **FALLBACK** | 81.70% | 1.000x | YES (4 vals) | `...weep/summary_snapshot_post_group7.csv` |
| 139 | `oats` | **FALLBACK** | 90.60% | 1.000x | YES (1 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 140 | `optimal_subspace_projection` | **FALLBACK** | 90.48% | 1.000x | no | `...p/distilbert/summary_buggy_backup.csv` |
| 141 | `optimized_hybrid_attention` | **FALLBACK** | 0.00% | 1.000x | YES (1 vals) | `...inal_paper_run/sst2_plotting_data.csv` |
| 142 | `orion` | **VALID** | 81.08% | 1.060x | YES (4 vals) | `...lts/unified_orion/qwen3_5/summary.csv` |
| 143 | `padre_attention` | **FALLBACK** | 91.90% | 1.000x | no | `...ank_ablation/lora0_agnews/summary.csv` |
| 144 | `performer_attention` | **FALLBACK** | 62.50% | 1.000x | YES (7 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 145 | `phantom` | **VALID** | 72.59% | 1.227x | YES (5 vals) | `...nified_phantom/distilbert/summary.csv` |
| 146 | `progressive_token_pruning` | **FALLBACK** | 86.93% | 1.000x | no | `results/distilbert/summary.csv` |
| 147 | `projection_factorized_attn` | **VALID** | 94.50% | 1.065x | no | `...k_ablation/agnews_rank256/summary.csv` |
| 148 | `proxsparse` | **FALLBACK** | 78.33% | 1.000x | YES (1 vals) | `...ied_proxsparse/distilbert/summary.csv` |
| 149 | `proxsparse_pruner` | **FALLBACK** | 78.33% | 1.000x | YES (6 vals) | `...xsparse_pruner/distilbert/summary.csv` |
| 150 | `pruning_utils` | **NO_DATA** | — | — | YES (3 vals) | `—` |
| 151 | `ptp` | **FALLBACK** | 90.94% | 1.000x | YES (4 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 152 | `qcqa_attention` | **FALLBACK** | 80.39% | 1.000x | YES (3 vals) | `results/distilbert/summary.csv` |
| 153 | `rala_attention` | **FALLBACK** | 77.98% | 1.000x | YES (4 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 154 | `reap_moe` | **FALLBACK** | 90.71% | 1.000x | no | `...p/distilbert/summary_buggy_backup.csv` |
| 155 | `regla_attention` | **FALLBACK** | 68.38% | 1.000x | YES (6 vals) | `...lts/adaptive_v3/benchmark_summary.csv` |
| 156 | `replace` | **FALLBACK** | 92.60% | 1.000x | YES (3 vals) | `...ep/distilbert_agnews_full/summary.csv` |
| 157 | `replaceme_compressor` | **FALLBACK** | 0.00% | 1.000x | YES (6 vals) | `...ts/all_121_compressors_gpu_matrix.csv` |
| 158 | `res_svd` | **VALID** | 92.70% | 1.036x | no | `...k_ablation/agnews_rank256/summary.csv` |
| 159 | `roast_hashed_linear` | **FALLBACK** | 91.06% | 1.000x | YES (2 vals) | `..._hashed_linear/distilbert/summary.csv` |
| 160 | `rotational_slicegpt_int4` | **FALLBACK** | 88.99% | 1.000x | YES (1 vals) | `..._slicegpt_int4/distilbert/summary.csv` |
| 161 | `routing_attention` | **FALLBACK** | 35.90% | 1.000x | YES (13 vals) | `...t_mnli_full_mnli_sweep/summary_v3.csv` |
| 162 | `rpc` | **FALLBACK** | 93.50% | 1.000x | YES (5 vals) | `...ank_ablation/lora0_agnews/summary.csv` |
| 163 | `rpca` | **VALID** | 60.21% | 1.003x | no | `results/unified_rpca/gpt2/summary.csv` |
| 164 | `saes_svd` | **FALLBACK** | 94.80% | 1.000x | YES (2 vals) | `...ep/distilbert_agnews_full/summary.csv` |
| 165 | `saes_svd_v2` | **FALLBACK** | 87.50% | 1.000x | YES (3 vals) | `...ed_saes_svd_v2/distilbert/summary.csv` |
| 166 | `salr` | **FALLBACK** | 94.90% | 1.000x | YES (1 vals) | `...ep/distilbert_agnews_full/summary.csv` |
| 167 | `salt` | **VALID** | 91.40% | 1.076x | no | `...k_ablation/agnews_rank256/summary.csv` |
| 168 | `samba_m3` | **FALLBACK** | 89.91% | 1.000x | no | `...p/distilbert/summary_buggy_backup.csv` |
| 169 | `samba_moe` | **FALLBACK** | 88.30% | 1.000x | no | `results/distilbert/summary.csv` |
| 170 | `sdrm` | **VALID** | 91.50% | 1.212x | YES (1 vals) | `...ep/distilbert_agnews_full/summary.csv` |
| 171 | `simple_ssm` | **FALLBACK** | 68.38% | 1.000x | YES (3 vals) | `...lts/adaptive_v3/benchmark_summary.csv` |
| 172 | `slicegpt` | **FALLBACK** | 86.00% | 1.000x | YES (2 vals) | `...ep/distilbert_agnews_full/summary.csv` |
| 173 | `slicegpt_pca` | **FALLBACK** | 94.60% | 1.000x | YES (1 vals) | `...ep/distilbert_agnews_full/summary.csv` |
| 174 | `slim_compressor` | **FALLBACK** | 0.00% | 1.000x | YES (6 vals) | `...ts/all_121_compressors_gpu_matrix.csv` |
| 175 | `smart_weighted_ensemble` | **FALLBACK** | 90.14% | 1.000x | YES (1 vals) | `...ghted_ensemble/distilbert/summary.csv` |
| 176 | `soa_attn` | **VALID** | 93.90% | 1.111x | no | `...ep/distilbert_agnews_full/summary.csv` |
| 177 | `soa_fused_module` | **VALID** | 81.77% | 1.162x | YES (3 vals) | `...a_fused_module/distilbert/summary.csv` |
| 178 | `soa_module` | **VALID** | 89.22% | 1.111x | YES (2 vals) | `...ied_soa_module/distilbert/summary.csv` |
| 179 | `soka` | **VALID** | 60.78% | 1.003x | YES (1 vals) | `results/unified_soka/gpt2/summary.csv` |
| 180 | `span_expanded_attention` | **FALLBACK** | 94.60% | 1.000x | no | `...ank_ablation/lora0_agnews/summary.csv` |
| 181 | `sparsegpt` | **FALLBACK** | 93.00% | 1.000x | YES (1 vals) | `...ep/distilbert_agnews_full/summary.csv` |
| 182 | `spectra_v2` | **FALLBACK** | 94.20% | 1.000x | no | `...ep/distilbert_agnews_full/summary.csv` |
| 183 | `spectraformer_attention` | **FALLBACK** | 77.64% | 0.991x | YES (6 vals) | `...rmer_attention/distilbert/summary.csv` |
| 184 | `spectral_mamba` | **FALLBACK** | 90.71% | 0.856x | YES (1 vals) | `...spectral_mamba/distilbert/summary.csv` |
| 185 | `spectron` | **VALID** | 91.90% | 1.152x | YES (2 vals) | `...ep/distilbert_agnews_full/summary.csv` |
| 186 | `squeezing_heads` | **FALLBACK** | 94.30% | 1.000x | no | `...ep/distilbert_agnews_full/summary.csv` |
| 187 | `squeezing_heads_distill` | **FALLBACK** | 90.37% | 0.991x | YES (2 vals) | `..._heads_distill/distilbert/summary.csv` |
| 188 | `squeezing_heads_distiller` | **FALLBACK** | 90.60% | 1.000x | YES (1 vals) | `results/distilbert/summary.csv` |
| 189 | `subspace_projection` | **VALID** | 59.52% | 1.003x | no | `..._subspace_projection/gpt2/summary.csv` |
| 190 | `svd_attention` | **FALLBACK** | 75.00% | 1.000x | YES (3 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 191 | `svd_compressed` | **VALID** | 94.80% | 1.066x | no | `...k_ablation/agnews_rank256/summary.csv` |
| 192 | `svd_linear_attention` | **FALLBACK** | 67.19% | 1.000x | YES (15 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 193 | `svd_linear_decay` | **VALID** | 94.50% | 1.065x | YES (1 vals) | `...k_ablation/agnews_rank256/summary.csv` |
| 194 | `svd_llm_v2_attention` | **FALLBACK** | 70.31% | 1.000x | YES (5 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 195 | `svd_mamba` | **FALLBACK** | 90.02% | 0.904x | YES (9 vals) | `...fied_svd_mamba/distilbert/summary.csv` |
| 196 | `svd_proj_attn` | **VALID** | 76.15% | 1.227x | YES (2 vals) | `..._svd_proj_attn/distilbert/summary.csv` |
| 197 | `svd_projection_attention` | **FALLBACK** | 70.31% | 1.000x | YES (2 vals) | `...r/abandoned_run/benchmark_summary.csv` |
| 198 | `svd_sequence_attention` | **FALLBACK** | 0.00% | 1.000x | YES (6 vals) | `results/master_aggregated_results.csv` |
| 199 | `swift_svd` | **VALID** | 89.30% | 1.344x | no | `...nk_ablation/agnews_rank32/summary.csv` |
| 200 | `swift_svd_act` | **VALID** | 60.09% | 1.003x | no | `...nified_swift_svd_act/gpt2/summary.csv` |
| 201 | `template_attention` | **FALLBACK** | 90.60% | 1.000x | YES (2 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 202 | `tlinformer_attention` | **FALLBACK** | 90.25% | 0.988x | YES (4 vals) | `...rmer_attention/distilbert/summary.csv` |
| 203 | `token_compressed_attention` | **FALLBACK** | 81.08% | 1.000x | no | `results/distilbert/summary.csv` |
| 204 | `tomoe` | **FALLBACK** | 94.40% | 1.000x | YES (2 vals) | `...ep/distilbert_agnews_full/summary.csv` |
| 205 | `tpa_attention` | **FALLBACK** | 77.29% | 1.000x | YES (2 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 206 | `tt` | **FALLBACK** | 94.70% | 1.000x | no | `...ep/distilbert_agnews_full/summary.csv` |
| 207 | `unified_rope` | **FALLBACK** | 93.00% | 1.000x | YES (2 vals) | `...ep/distilbert_agnews_full/summary.csv` |
| 208 | `uniql_compressor` | **FALLBACK** | 0.00% | 1.000x | YES (6 vals) | `...ts/all_121_compressors_gpu_matrix.csv` |
| 209 | `upi_mamba` | **FALLBACK** | 0.00% | 1.000x | YES (3 vals) | `results/master_aggregated_results.csv` |
| 210 | `upi_stuffed_mamba` | **FALLBACK** | 68.38% | 1.000x | YES (3 vals) | `...lts/adaptive_v3/benchmark_summary.csv` |
| 211 | `vig_attention` | **FALLBACK** | 88.30% | 0.990x | YES (2 vals) | `...ts/unified_vig/distilbert/summary.csv` |
| 212 | `vspec` | **VALID** | 93.50% | 1.036x | no | `...k_ablation/agnews_rank256/summary.csv` |
| 213 | `weighted_ensemble` | **FALLBACK** | 90.02% | 1.000x | YES (1 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 214 | `wgqa` | **FALLBACK** | 90.60% | 1.000x | YES (1 vals) | `...p/distilbert/summary_buggy_backup.csv` |
| 215 | `xattention` | **FALLBACK** | 86.24% | 0.998x | YES (1 vals) | `...s/unified_xattention/gpt2/summary.csv` |
| 216 | `zerokv_neo` | **FALLBACK** | 80.50% | 1.000x | YES (2 vals) | `results/distilbert/summary.csv` |
| 217 | `zeros_attention` | **FALLBACK** | 59.75% | 0.991x | YES (2 vals) | `results/unified_zeros/gpt2/summary.csv` |
