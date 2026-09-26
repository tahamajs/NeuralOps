# 🗂 Results Master Directory & Quick-Access Catalog

This catalog provides an instant directory index and quick-access map for all **268 benchmark suites** stored under `results/`.

You can locate any specific experiment, summary CSV, or JSON artifact instantly using the categorized tables below.

---

## 📁 Diagnostics & Smoke Tests (39 suites)

| Directory Name | Relative Path | Subdirs | CSVs | JSONs | Primary Summary File |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `a3_tests` | [`results/a3_tests`](./a3_tests) | 3 | 1 | 32 | `roberta_zeroshot/summary.csv` |
| `concurrent_test` | [`results/concurrent_test`](./concurrent_test) | 1 | 1 | 6 | `summary.csv` |
| `diag_test` | [`results/diag_test`](./diag_test) | 1 | 1 | 3 | `summary.csv` |
| `gpt2_svd_llm_test` | [`results/gpt2_svd_llm_test`](./gpt2_svd_llm_test) | 1 | 1 | 1 | `summary.csv` |
| `isolated_test` | [`results/isolated_test`](./isolated_test) | 0 | 1 | 2 | `summary.csv` |
| `mrpc_test` | [`results/mrpc_test`](./mrpc_test) | 7 | 4 | 5 | `-` |
| `naive_svd_test` | [`results/naive_svd_test`](./naive_svd_test) | 1 | 1 | 3 | `summary.csv` |
| `performer_v2_test` | [`results/performer_v2_test`](./performer_v2_test) | 1 | 1 | 3 | `summary.csv` |
| `performer_v3_test` | [`results/performer_v3_test`](./performer_v3_test) | 1 | 1 | 3 | `summary.csv` |
| `preflight_leak_test` | [`results/preflight_leak_test`](./preflight_leak_test) | 1 | 1 | 2 | `summary.csv` |
| `smoke_test_hgrn2_layer` | [`results/smoke_test_hgrn2_layer`](./smoke_test_hgrn2_layer) | 1 | 1 | 4 | `summary.csv` |
| `smoke_test_hgrn2_layer_with_recovery` | [`results/smoke_test_hgrn2_layer_with_recovery`](./smoke_test_hgrn2_layer_with_recovery) | 1 | 1 | 4 | `summary.csv` |
| `smoke_test_replace` | [`results/smoke_test_replace`](./smoke_test_replace) | 1 | 1 | 3 | `summary.csv` |
| `smoke_test_swift_svd_act` | [`results/smoke_test_swift_svd_act`](./smoke_test_swift_svd_act) | 1 | 1 | 4 | `summary.csv` |
| `test_adasvd_gpt2` | [`results/test_adasvd_gpt2`](./test_adasvd_gpt2) | 0 | 1 | 3 | `summary.csv` |
| `test_agnews_600steps` | [`results/test_agnews_600steps`](./test_agnews_600steps) | 0 | 1 | 2 | `summary.csv` |
| `test_agnews_baseline` | [`results/test_agnews_baseline`](./test_agnews_baseline) | 0 | 1 | 2 | `summary.csv` |
| `test_agnews_svd` | [`results/test_agnews_svd`](./test_agnews_svd) | 0 | 1 | 2 | `summary.csv` |
| `test_baseline_agnews` | [`results/test_baseline_agnews`](./test_baseline_agnews) | 0 | 1 | 1 | `summary.csv` |
| `test_baseline_mnli` | [`results/test_baseline_mnli`](./test_baseline_mnli) | 0 | 1 | 2 | `summary.csv` |
| `test_batch` | [`results/test_batch`](./test_batch) | 0 | 0 | 2 | `-` |
| `test_bs128` | [`results/test_bs128`](./test_bs128) | 0 | 1 | 2 | `summary.csv` |
| `test_bs256` | [`results/test_bs256`](./test_bs256) | 0 | 1 | 2 | `summary.csv` |
| `test_bs32` | [`results/test_bs32`](./test_bs32) | 0 | 1 | 2 | `summary.csv` |
| `test_fail_below_chance` | [`results/test_fail_below_chance`](./test_fail_below_chance) | 1 | 1 | 4 | `summary.csv` |
| `test_gpt2_base` | [`results/test_gpt2_base`](./test_gpt2_base) | 1 | 0 | 4 | `-` |
| `test_gpt2_base_clean` | [`results/test_gpt2_base_clean`](./test_gpt2_base_clean) | 1 | 0 | 3 | `-` |
| `test_local` | [`results/test_local`](./test_local) | 2 | 1 | 9 | `summary.csv` |
| `test_local_final` | [`results/test_local_final`](./test_local_final) | 1 | 1 | 2 | `summary.csv` |
| `test_memory_safe` | [`results/test_memory_safe`](./test_memory_safe) | 0 | 0 | 3 | `-` |
| `test_mnli_distilbert` | [`results/test_mnli_distilbert`](./test_mnli_distilbert) | 0 | 1 | 3 | `summary.csv` |
| `test_perf` | [`results/test_perf`](./test_perf) | 0 | 0 | 2 | `-` |
| `test_replace_fixed` | [`results/test_replace_fixed`](./test_replace_fixed) | 1 | 1 | 3 | `summary.csv` |
| `test_replace_recovery` | [`results/test_replace_recovery`](./test_replace_recovery) | 1 | 1 | 4 | `summary.csv` |
| `test_roberta_base` | [`results/test_roberta_base`](./test_roberta_base) | 1 | 0 | 3 | `-` |
| `test_run` | [`results/test_run`](./test_run) | 1 | 1 | 4 | `summary.csv` |
| `test_ssm_stsb` | [`results/test_ssm_stsb`](./test_ssm_stsb) | 1 | 1 | 2 | `summary.csv` |
| `test_zeroshot` | [`results/test_zeroshot`](./test_zeroshot) | 0 | 0 | 1 | `-` |
| `tpa_test` | [`results/tpa_test`](./tpa_test) | 1 | 1 | 3 | `summary.csv` |

---

## 📁 Model/GLUE Classification Sweeps (39 suites)

| Directory Name | Relative Path | Subdirs | CSVs | JSONs | Primary Summary File |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `batch2_distilbert` | [`results/batch2_distilbert`](./batch2_distilbert) | 23 | 46 | 174 | `aasvd/rank_128/summary.csv` |
| `batch2_roberta` | [`results/batch2_roberta`](./batch2_roberta) | 23 | 23 | 84 | `aasvd/rank_256/summary.csv` |
| `batch3_distilbert` | [`results/batch3_distilbert`](./batch3_distilbert) | 34 | 34 | 140 | `hybridkv/summary.csv` |
| `batch3_roberta` | [`results/batch3_roberta`](./batch3_roberta) | 34 | 34 | 137 | `hybridkv/summary.csv` |
| `batch_cache_mnli` | [`results/batch_cache_mnli`](./batch_cache_mnli) | 0 | 0 | 1 | `-` |
| `bert_baselines` | [`results/bert_baselines`](./bert_baselines) | 1 | 1 | 0 | `adasvd/summary.csv` |
| `bert_sst2_baseline` | [`results/bert_sst2_baseline`](./bert_sst2_baseline) | 1 | 0 | 1 | `-` |
| `distilbert_agnews_baseline` | [`results/distilbert_agnews_baseline`](./distilbert_agnews_baseline) | 0 | 1 | 2 | `summary.csv` |
| `distilbert_base_uncased_finetuned_sst_2_english_sst2_sweep` | [`results/distilbert_base_uncased_finetuned_sst_2_english_sst2_sweep`](./distilbert_base_uncased_finetuned_sst_2_english_sst2_sweep) | 1 | 0 | 6 | `-` |
| `distilbert_base_uncased_sst2_sweep` | [`results/distilbert_base_uncased_sst2_sweep`](./distilbert_base_uncased_sst2_sweep) | 1 | 1 | 7 | `-` |
| `distilbert_mnli_baseline` | [`results/distilbert_mnli_baseline`](./distilbert_mnli_baseline) | 0 | 1 | 2 | `summary.csv` |
| `distilbert_mnli_full_mnli_sweep` | [`results/distilbert_mnli_full_mnli_sweep`](./distilbert_mnli_full_mnli_sweep) | 2 | 266 | 264 | `summary.csv` |
| `distilbert_rte_fast` | [`results/distilbert_rte_fast`](./distilbert_rte_fast) | 0 | 1 | 1 | `summary.csv` |
| `distilbert_sst2_ag_news_sweep` | [`results/distilbert_sst2_ag_news_sweep`](./distilbert_sst2_ag_news_sweep) | 0 | 1 | 2 | `summary.csv` |
| `distilbert_sst2_sst2_sweep` | [`results/distilbert_sst2_sst2_sweep`](./distilbert_sst2_sst2_sweep) | 1 | 599 | 388 | `summary.csv` |
| `distilbert_sst2_sst2_sweep_r192` | [`results/distilbert_sst2_sst2_sweep_r192`](./distilbert_sst2_sst2_sweep_r192) | 1 | 1 | 4 | `-` |
| `distilbert_sst2_sst2_sweep_r256` | [`results/distilbert_sst2_sst2_sweep_r256`](./distilbert_sst2_sst2_sweep_r256) | 1 | 1 | 4 | `-` |
| `gpt2_adasvd` | [`results/gpt2_adasvd`](./gpt2_adasvd) | 1 | 7 | 15 | `adasvd_compressor/rank_curve.csv` |
| `gpt2_adasvd_fixed` | [`results/gpt2_adasvd_fixed`](./gpt2_adasvd_fixed) | 1 | 4 | 6 | `adasvd_compressor/rank_curve.csv` |
| `gpt2_adasvd_qkv_split` | [`results/gpt2_adasvd_qkv_split`](./gpt2_adasvd_qkv_split) | 1 | 4 | 5 | `adasvd_compressor/rank_curve.csv` |
| `gpt2_adasvd_r256_solo` | [`results/gpt2_adasvd_r256_solo`](./gpt2_adasvd_r256_solo) | 0 | 1 | 3 | `summary.csv` |
| `gpt2_ag_news_sweep` | [`results/gpt2_ag_news_sweep`](./gpt2_ag_news_sweep) | 0 | 1 | 2 | `summary.csv` |
| `gpt2_baselines_wt2` | [`results/gpt2_baselines_wt2`](./gpt2_baselines_wt2) | 3 | 8 | 23 | `svd_llm_compressor/rank_curve.csv` |
| `gpt2_r384` | [`results/gpt2_r384`](./gpt2_r384) | 5 | 11 | 10 | `slicegpt/rank_curve.csv` |
| `gpt2_sst2_lora_baseline` | [`results/gpt2_sst2_lora_baseline`](./gpt2_sst2_lora_baseline) | 1 | 1 | 3 | `summary.csv` |
| `gpt2_sweep` | [`results/gpt2_sweep`](./gpt2_sweep) | 0 | 1 | 0 | `summary.csv` |
| `qwen2.5_0.5b_instruct_sst2_sweep` | [`results/qwen2.5_0.5b_instruct_sst2_sweep`](./qwen2.5_0.5b_instruct_sst2_sweep) | 1 | 1 | 6 | `summary.csv` |
| `roberta_base_ag_news_sweep` | [`results/roberta_base_ag_news_sweep`](./roberta_base_ag_news_sweep) | 0 | 1 | 2 | `summary.csv` |
| `roberta_mnli_baseline` | [`results/roberta_mnli_baseline`](./roberta_mnli_baseline) | 0 | 1 | 2 | `summary.csv` |
| `roberta_sst2_lora_baseline` | [`results/roberta_sst2_lora_baseline`](./roberta_sst2_lora_baseline) | 0 | 1 | 2 | `summary.csv` |
| `roberta_sst2_sst2_sweep` | [`results/roberta_sst2_sst2_sweep`](./roberta_sst2_sst2_sweep) | 1 | 264 | 292 | `summary.csv` |
| `t5_cola_parallel` | [`results/t5_cola_parallel`](./t5_cola_parallel) | 0 | 1 | 1 | `summary.csv` |
| `t5_cola_stsb_fast` | [`results/t5_cola_stsb_fast`](./t5_cola_stsb_fast) | 0 | 1 | 1 | `summary.csv` |
| `t5_cola_stsb_maxgpu` | [`results/t5_cola_stsb_maxgpu`](./t5_cola_stsb_maxgpu) | 0 | 1 | 1 | `summary.csv` |
| `t5_cola_stsb_offline` | [`results/t5_cola_stsb_offline`](./t5_cola_stsb_offline) | 0 | 1 | 1 | `summary.csv` |
| `t5_sst2_attention_compat_final` | [`results/t5_sst2_attention_compat_final`](./t5_sst2_attention_compat_final) | 0 | 1 | 0 | `summary.csv` |
| `t5_sst2_full_sweep_fixed_v3` | [`results/t5_sst2_full_sweep_fixed_v3`](./t5_sst2_full_sweep_fixed_v3) | 0 | 1 | 0 | `summary.csv` |
| `t5_sst2_recovery` | [`results/t5_sst2_recovery`](./t5_sst2_recovery) | 0 | 1 | 0 | `summary.csv` |
| `t5_sst2_sweep` | [`results/t5_sst2_sweep`](./t5_sst2_sweep) | 0 | 1 | 0 | `summary.csv` |

---

## 📁 Multi-Seed Statistical Rigor (4 suites)

| Directory Name | Relative Path | Subdirs | CSVs | JSONs | Primary Summary File |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `multiseed` | [`results/multiseed`](./multiseed) | 5 | 14 | 60 | `summary.csv` |
| `multiseed_rigor_distilbert` | [`results/multiseed_rigor_distilbert`](./multiseed_rigor_distilbert) | 11 | 22 | 82 | `adasvd_compressor/seed_456/summary.csv` |
| `multiseed_rigor_roberta` | [`results/multiseed_rigor_roberta`](./multiseed_rigor_roberta) | 11 | 22 | 82 | `adasvd_compressor/seed_456/summary.csv` |
| `multiseed_test` | [`results/multiseed_test`](./multiseed_test) | 2 | 4 | 2 | `seed123/summary.csv` |

---

## 📁 Perplexity & Long-Context Scaling (18 suites)

| Directory Name | Relative Path | Subdirs | CSVs | JSONs | Primary Summary File |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `c4_baselines` | [`results/c4_baselines`](./c4_baselines) | 8 | 10 | 67 | `kronecker/summary.csv` |
| `gpt2_c4_ppl` | [`results/gpt2_c4_ppl`](./gpt2_c4_ppl) | 1 | 0 | 0 | `-` |
| `gpt2_wikitext2_sweep` | [`results/gpt2_wikitext2_sweep`](./gpt2_wikitext2_sweep) | 18 | 127 | 154 | `summary.csv` |
| `long_context` | [`results/long_context`](./long_context) | 0 | 1 | 0 | `summary.csv` |
| `minimal_svd_gpt2_ppl` | [`results/minimal_svd_gpt2_ppl`](./minimal_svd_gpt2_ppl) | 0 | 1 | 3 | `summary.csv` |
| `naive_svd_gpt2_ppl` | [`results/naive_svd_gpt2_ppl`](./naive_svd_gpt2_ppl) | 0 | 1 | 3 | `summary.csv` |
| `seqlen_48h` | [`results/seqlen_48h`](./seqlen_48h) | 6 | 18 | 67 | `simple_ssm/seq_128/summary.csv` |
| `seqlen_scaling` | [`results/seqlen_scaling`](./seqlen_scaling) | 13 | 13 | 0 | `full_attention/summary.csv` |
| `seqlen_scaling_v2` | [`results/seqlen_scaling_v2`](./seqlen_scaling_v2) | 3 | 3 | 0 | `full_attention/summary.csv` |
| `seqlen_scaling_v2_linformer` | [`results/seqlen_scaling_v2_linformer`](./seqlen_scaling_v2_linformer) | 1 | 1 | 0 | `linformer_attention/summary.csv` |
| `test_c4_a3` | [`results/test_c4_a3`](./test_c4_a3) | 0 | 1 | 3 | `summary.csv` |
| `test_c4_a3_r128` | [`results/test_c4_a3_r128`](./test_c4_a3_r128) | 0 | 0 | 2 | `-` |
| `test_c4_adasvd` | [`results/test_c4_adasvd`](./test_c4_adasvd) | 0 | 1 | 2 | `summary.csv` |
| `test_c4_adasvd_r128` | [`results/test_c4_adasvd_r128`](./test_c4_adasvd_r128) | 0 | 1 | 2 | `summary.csv` |
| `test_c4_adasvd_r384` | [`results/test_c4_adasvd_r384`](./test_c4_adasvd_r384) | 0 | 0 | 1 | `-` |
| `test_c4_slicegpt` | [`results/test_c4_slicegpt`](./test_c4_slicegpt) | 0 | 1 | 3 | `summary.csv` |
| `test_seqlen_scaling` | [`results/test_seqlen_scaling`](./test_seqlen_scaling) | 0 | 1 | 3 | `summary.csv` |
| `wikitext` | [`results/wikitext`](./wikitext) | 46 | 0 | 52 | `-` |

---

## 📁 Rank Sweeps & SVD Scaling (20 suites)

| Directory Name | Relative Path | Subdirs | CSVs | JSONs | Primary Summary File |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `attention_rank_128` | [`results/attention_rank_128`](./attention_rank_128) | 1 | 1 | 6 | `summary.csv` |
| `attention_rank_16` | [`results/attention_rank_16`](./attention_rank_16) | 1 | 1 | 7 | `summary.csv` |
| `attention_rank_256` | [`results/attention_rank_256`](./attention_rank_256) | 1 | 1 | 6 | `summary.csv` |
| `attention_rank_32` | [`results/attention_rank_32`](./attention_rank_32) | 1 | 1 | 7 | `summary.csv` |
| `attention_rank_64` | [`results/attention_rank_64`](./attention_rank_64) | 1 | 1 | 6 | `summary.csv` |
| `deep_rank_distilbert` | [`results/deep_rank_distilbert`](./deep_rank_distilbert) | 48 | 192 | 727 | `aa_svd_compressor/rank_384/summary.csv` |
| `deep_rank_roberta` | [`results/deep_rank_roberta`](./deep_rank_roberta) | 48 | 192 | 740 | `aa_svd_compressor/rank_384/summary.csv` |
| `gpt2_adasvd_highrank` | [`results/gpt2_adasvd_highrank`](./gpt2_adasvd_highrank) | 1 | 4 | 4 | `adasvd_compressor/rank_curve.csv` |
| `rank128_bert` | [`results/rank128_bert`](./rank128_bert) | 1 | 1 | 4 | `summary.csv` |
| `rank256_bert` | [`results/rank256_bert`](./rank256_bert) | 1 | 1 | 4 | `summary.csv` |
| `rank32_bert` | [`results/rank32_bert`](./rank32_bert) | 1 | 1 | 5 | `summary.csv` |
| `rank64_bert` | [`results/rank64_bert`](./rank64_bert) | 1 | 1 | 5 | `summary.csv` |
| `rank_ablation` | [`results/rank_ablation`](./rank_ablation) | 18 | 24 | 762 | `lora0_mnli/summary.csv` |
| `rank_sweep` | [`results/rank_sweep`](./rank_sweep) | 3 | 4 | 6 | `roberta_test_lr5e5/summary.csv` |
| `rank_sweep_agnews` | [`results/rank_sweep_agnews`](./rank_sweep_agnews) | 3 | 4 | 4 | `cluster_attention/rank_curve.csv` |
| `rank_sweep_gpt2` | [`results/rank_sweep_gpt2`](./rank_sweep_gpt2) | 6 | 30 | 245 | `svd_llm_compressor/rank_16/summary.csv` |
| `rank_sweep_mnli` | [`results/rank_sweep_mnli`](./rank_sweep_mnli) | 4 | 7 | 7 | `svd_llm_compressor/rank_curve.csv` |
| `rank_sweep_roberta` | [`results/rank_sweep_roberta`](./rank_sweep_roberta) | 7 | 42 | 393 | `svd_llm_compressor/rank_16/summary.csv` |
| `rank_sweep_sst2` | [`results/rank_sweep_sst2`](./rank_sweep_sst2) | 7 | 43 | 319 | `svd_llm_compressor/rank_16/summary.csv` |
| `test_replace_rank0` | [`results/test_replace_rank0`](./test_replace_rank0) | 1 | 1 | 4 | `summary.csv` |

---

## 📁 Targeted Module Experiments (142 suites)

| Directory Name | Relative Path | Subdirs | CSVs | JSONs | Primary Summary File |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `a3_optimization` | [`results/a3_optimization`](./a3_optimization) | 0 | 1 | 0 | `-` |
| `aasvd_ablations` | [`results/aasvd_ablations`](./aasvd_ablations) | 2 | 8 | 12 | `rank/rank256/summary.csv` |
| `aatc_xkv_eval` | [`results/aatc_xkv_eval`](./aatc_xkv_eval) | 1 | 0 | 1 | `-` |
| `adaptive_v3` | [`results/adaptive_v3`](./adaptive_v3) | 0 | 1 | 0 | `-` |
| `admm_aggressive` | [`results/admm_aggressive`](./admm_aggressive) | 1 | 1 | 4 | `summary.csv` |
| `admm_s50` | [`results/admm_s50`](./admm_s50) | 1 | 1 | 4 | `summary.csv` |
| `admm_s50_q4` | [`results/admm_s50_q4`](./admm_s50_q4) | 1 | 1 | 4 | `summary.csv` |
| `admm_s50_q8` | [`results/admm_s50_q8`](./admm_s50_q8) | 1 | 1 | 4 | `summary.csv` |
| `admm_s70` | [`results/admm_s70`](./admm_s70) | 1 | 1 | 4 | `summary.csv` |
| `aegis_hybrid_fixed` | [`results/aegis_hybrid_fixed`](./aegis_hybrid_fixed) | 1 | 1 | 4 | `summary.csv` |
| `aegis_v2_fixed` | [`results/aegis_v2_fixed`](./aegis_v2_fixed) | 1 | 1 | 3 | `summary.csv` |
| `aegis_v4_fixed` | [`results/aegis_v4_fixed`](./aegis_v4_fixed) | 1 | 1 | 3 | `summary.csv` |
| `aegis_v5_fixed` | [`results/aegis_v5_fixed`](./aegis_v5_fixed) | 1 | 1 | 3 | `summary.csv` |
| `aegis_v6_fixed` | [`results/aegis_v6_fixed`](./aegis_v6_fixed) | 1 | 1 | 4 | `summary.csv` |
| `aegis_v6_single` | [`results/aegis_v6_single`](./aegis_v6_single) | 1 | 1 | 2 | `summary.csv` |
| `agnews_Dataset` | [`results/agnews_Dataset`](./agnews_Dataset) | 103 | 178 | 368 | `agnews_unified_lora_attention/distilbert/summary.csv` |
| `all_modules_paper_run` | [`results/all_modules_paper_run`](./all_modules_paper_run) | 1 | 1 | 2 | `summary.csv` |
| `armor_aggressive` | [`results/armor_aggressive`](./armor_aggressive) | 1 | 1 | 6 | `summary.csv` |
| `armor_r16` | [`results/armor_r16`](./armor_r16) | 1 | 1 | 4 | `summary.csv` |
| `armor_r32` | [`results/armor_r32`](./armor_r32) | 1 | 1 | 4 | `summary.csv` |
| `attn_distilbert` | [`results/attn_distilbert`](./attn_distilbert) | 18 | 18 | 57 | `samba_m3/summary.csv` |
| `attn_roberta` | [`results/attn_roberta`](./attn_roberta) | 18 | 17 | 63 | `samba_m3/summary.csv` |
| `baseline_check` | [`results/baseline_check`](./baseline_check) | 1 | 1 | 3 | `roberta_sst2/summary.csv` |
| `baseline_comparison` | [`results/baseline_comparison`](./baseline_comparison) | 3 | 1 | 18 | `summary.csv` |
| `benchmark_results` | [`results/benchmark_results`](./benchmark_results) | 0 | 0 | 1 | `-` |
| `bonsai_s50` | [`results/bonsai_s50`](./bonsai_s50) | 1 | 1 | 3 | `summary.csv` |
| `come_fixed_distil` | [`results/come_fixed_distil`](./come_fixed_distil) | 1 | 6 | 17 | `come/rank_curve.csv` |
| `come_fixed_gpt2` | [`results/come_fixed_gpt2`](./come_fixed_gpt2) | 1 | 6 | 15 | `come/rank_curve.csv` |
| `come_fixed_roberta` | [`results/come_fixed_roberta`](./come_fixed_roberta) | 1 | 6 | 16 | `come/rank_curve.csv` |
| `come_fixed_roberta_v2` | [`results/come_fixed_roberta_v2`](./come_fixed_roberta_v2) | 1 | 6 | 16 | `come/rank_curve.csv` |
| `cosformer_fixed` | [`results/cosformer_fixed`](./cosformer_fixed) | 1 | 1 | 3 | `summary.csv` |
| `cosformer_grid` | [`results/cosformer_grid`](./cosformer_grid) | 8 | 8 | 24 | `sst2_rank16/summary.csv` |
| `cosformer_single` | [`results/cosformer_single`](./cosformer_single) | 1 | 1 | 3 | `summary.csv` |
| `cosformer_v3` | [`results/cosformer_v3`](./cosformer_v3) | 1 | 1 | 3 | `summary.csv` |
| `distilbert` | [`results/distilbert`](./distilbert) | 0 | 1 | 0 | `summary.csv` |
| `dynamic_entropy_r64` | [`results/dynamic_entropy_r64`](./dynamic_entropy_r64) | 1 | 1 | 4 | `summary.csv` |
| `dynamic_entropy_r64_v2` | [`results/dynamic_entropy_r64_v2`](./dynamic_entropy_r64_v2) | 1 | 1 | 4 | `summary.csv` |
| `enlighten_r64` | [`results/enlighten_r64`](./enlighten_r64) | 1 | 1 | 4 | `summary.csv` |
| `enlighten_r64_v2` | [`results/enlighten_r64_v2`](./enlighten_r64_v2) | 1 | 1 | 4 | `summary.csv` |
| `erc_e20` | [`results/erc_e20`](./erc_e20) | 1 | 1 | 4 | `summary.csv` |
| `erc_fix_verify` | [`results/erc_fix_verify`](./erc_fix_verify) | 1 | 1 | 5 | `summary.csv` |
| `erc_fix_verify_r256` | [`results/erc_fix_verify_r256`](./erc_fix_verify_r256) | 1 | 1 | 5 | `summary.csv` |
| `experiments` | [`results/experiments`](./experiments) | 249 | 119 | 974 | `-` |
| `final_paper_run` | [`results/final_paper_run`](./final_paper_run) | 0 | 2 | 4 | `summary.csv` |
| `fixed_aegis` | [`results/fixed_aegis`](./fixed_aegis) | 7 | 7 | 14 | `aegis_hybrid_sst2/summary.csv` |
| `fixed_modules_verify` | [`results/fixed_modules_verify`](./fixed_modules_verify) | 1 | 1 | 5 | `summary.csv` |
| `flap_s20` | [`results/flap_s20`](./flap_s20) | 1 | 1 | 4 | `summary.csv` |
| `flap_s50` | [`results/flap_s50`](./flap_s50) | 1 | 1 | 4 | `summary.csv` |
| `full_benchmark` | [`results/full_benchmark`](./full_benchmark) | 0 | 0 | 0 | `-` |
| `full_experiment_run` | [`results/full_experiment_run`](./full_experiment_run) | 0 | 1 | 2 | `summary.csv` |
| `galerkin_v2` | [`results/galerkin_v2`](./galerkin_v2) | 1 | 1 | 3 | `summary.csv` |
| `generalization_bert_mrpc` | [`results/generalization_bert_mrpc`](./generalization_bert_mrpc) | 0 | 1 | 1 | `summary.csv` |
| `generated_csvs` | [`results/generated_csvs`](./generated_csvs) | 0 | 202 | 0 | `-` |
| `gpt2` | [`results/gpt2`](./gpt2) | 0 | 1 | 0 | `summary.csv` |
| `gpu_qwen_sweep_` | [`results/gpu_qwen_sweep_`](./gpu_qwen_sweep_) | 0 | 1 | 0 | `summary.csv` |
| `gpu_run_20260901_174218` | [`results/gpu_run_20260901_174218`](./gpu_run_20260901_174218) | 1 | 1 | 0 | `qwen_sweep/summary.csv` |
| `gpu_server` | [`results/gpu_server`](./gpu_server) | 4 | 5 | 720 | `-` |
| `hamming_fixed` | [`results/hamming_fixed`](./hamming_fixed) | 1 | 1 | 3 | `summary.csv` |
| `hybrid_svd_r32` | [`results/hybrid_svd_r32`](./hybrid_svd_r32) | 1 | 1 | 4 | `summary.csv` |
| `hybrid_svd_r64` | [`results/hybrid_svd_r64`](./hybrid_svd_r64) | 1 | 1 | 4 | `summary.csv` |
| `kda_fixed` | [`results/kda_fixed`](./kda_fixed) | 1 | 1 | 3 | `summary.csv` |
| `kl_r16` | [`results/kl_r16`](./kl_r16) | 1 | 1 | 2 | `summary.csv` |
| `kmeans_bits8` | [`results/kmeans_bits8`](./kmeans_bits8) | 1 | 1 | 4 | `summary.csv` |
| `ktl_r8` | [`results/ktl_r8`](./ktl_r8) | 1 | 1 | 4 | `summary.csv` |
| `linear_attention_v5` | [`results/linear_attention_v5`](./linear_attention_v5) | 1 | 1 | 6 | `summary.csv` |
| `lola_r128` | [`results/lola_r128`](./lola_r128) | 1 | 1 | 3 | `summary.csv` |
| `lola_r16` | [`results/lola_r16`](./lola_r16) | 1 | 1 | 3 | `summary.csv` |
| `lola_r32` | [`results/lola_r32`](./lola_r32) | 1 | 1 | 3 | `summary.csv` |
| `lola_r4` | [`results/lola_r4`](./lola_r4) | 1 | 1 | 3 | `summary.csv` |
| `lola_r64` | [`results/lola_r64`](./lola_r64) | 1 | 1 | 3 | `summary.csv` |
| `lola_r8` | [`results/lola_r8`](./lola_r8) | 1 | 1 | 3 | `summary.csv` |
| `lra_listops_adaptive_svd_attention` | [`results/lra_listops_adaptive_svd_attention`](./lra_listops_adaptive_svd_attention) | 0 | 0 | 1 | `-` |
| `lra_listops_asvd_attention` | [`results/lra_listops_asvd_attention`](./lra_listops_asvd_attention) | 0 | 0 | 1 | `-` |
| `lra_listops_cosformer` | [`results/lra_listops_cosformer`](./lra_listops_cosformer) | 0 | 0 | 1 | `-` |
| `lra_listops_galerkin` | [`results/lra_listops_galerkin`](./lra_listops_galerkin) | 0 | 0 | 1 | `-` |
| `lra_listops_performer_attention` | [`results/lra_listops_performer_attention`](./lra_listops_performer_attention) | 0 | 0 | 1 | `-` |
| `lra_listops_squeezing_heads` | [`results/lra_listops_squeezing_heads`](./lra_listops_squeezing_heads) | 0 | 0 | 1 | `-` |
| `mnli_Dataset` | [`results/mnli_Dataset`](./mnli_Dataset) | 88 | 103 | 229 | `mnli_unified_leka_v1/distilbert/summary.csv` |
| `module_plots` | [`results/module_plots`](./module_plots) | 0 | 0 | 0 | `-` |
| `mrpc_fix_majority` | [`results/mrpc_fix_majority`](./mrpc_fix_majority) | 0 | 1 | 0 | `-` |
| `mrpc_fix_oom` | [`results/mrpc_fix_oom`](./mrpc_fix_oom) | 0 | 1 | 0 | `-` |
| `mrpc_squeezing_fixed` | [`results/mrpc_squeezing_fixed`](./mrpc_squeezing_fixed) | 3 | 3 | 0 | `-` |
| `naive_svd_comparison` | [`results/naive_svd_comparison`](./naive_svd_comparison) | 3 | 3 | 9 | `baseline/summary.csv` |
| `neuralops_suite` | [`results/neuralops_suite`](./neuralops_suite) | 1 | 2 | 31 | `-` |
| `nli_softmax_distilbert` | [`results/nli_softmax_distilbert`](./nli_softmax_distilbert) | 1 | 1 | 3 | `summary.csv` |
| `nli_softmax_fixed` | [`results/nli_softmax_fixed`](./nli_softmax_fixed) | 1 | 1 | 3 | `summary.csv` |
| `nli_softmax_roberta` | [`results/nli_softmax_roberta`](./nli_softmax_roberta) | 1 | 1 | 3 | `summary.csv` |
| `oha_r64` | [`results/oha_r64`](./oha_r64) | 1 | 1 | 3 | `summary.csv` |
| `orion_favor_fixed` | [`results/orion_favor_fixed`](./orion_favor_fixed) | 1 | 1 | 3 | `summary.csv` |
| `orion_fixed` | [`results/orion_fixed`](./orion_fixed) | 1 | 1 | 3 | `summary.csv` |
| `orion_r128` | [`results/orion_r128`](./orion_r128) | 1 | 1 | 3 | `summary.csv` |
| `orion_r16` | [`results/orion_r16`](./orion_r16) | 1 | 1 | 3 | `summary.csv` |
| `orion_r32` | [`results/orion_r32`](./orion_r32) | 1 | 1 | 3 | `summary.csv` |
| `orion_r4` | [`results/orion_r4`](./orion_r4) | 1 | 1 | 3 | `summary.csv` |
| `orion_r64` | [`results/orion_r64`](./orion_r64) | 1 | 1 | 3 | `summary.csv` |
| `orion_r8` | [`results/orion_r8`](./orion_r8) | 1 | 1 | 3 | `summary.csv` |
| `primal_fixed` | [`results/primal_fixed`](./primal_fixed) | 1 | 1 | 3 | `summary.csv` |
| `primal_r16` | [`results/primal_r16`](./primal_r16) | 1 | 1 | 3 | `summary.csv` |
| `primal_r32` | [`results/primal_r32`](./primal_r32) | 1 | 1 | 3 | `summary.csv` |
| `primal_r4` | [`results/primal_r4`](./primal_r4) | 1 | 1 | 3 | `summary.csv` |
| `primal_r8` | [`results/primal_r8`](./primal_r8) | 1 | 1 | 3 | `summary.csv` |
| `queued_come_distil` | [`results/queued_come_distil`](./queued_come_distil) | 1 | 4 | 8 | `come/rank_curve.csv` |
| `queued_come_roberta` | [`results/queued_come_roberta`](./queued_come_roberta) | 1 | 5 | 12 | `come/rank_curve.csv` |
| `queued_cure_distil` | [`results/queued_cure_distil`](./queued_cure_distil) | 1 | 4 | 8 | `cure/rank_curve.csv` |
| `queued_saes_roberta` | [`results/queued_saes_roberta`](./queued_saes_roberta) | 1 | 3 | 4 | `saes_svd/rank_curve.csv` |
| `regla_r64` | [`results/regla_r64`](./regla_r64) | 1 | 1 | 3 | `summary.csv` |
| `replace_gpt2_r32` | [`results/replace_gpt2_r32`](./replace_gpt2_r32) | 0 | 1 | 0 | `summary.csv` |
| `replace_r64` | [`results/replace_r64`](./replace_r64) | 1 | 1 | 3 | `summary.csv` |
| `results` | [`results/results`](./results) | 0 | 1 | 1 | `-` |
| `retest_heavy_modules` | [`results/retest_heavy_modules`](./retest_heavy_modules) | 0 | 0 | 1 | `-` |
| `roberta` | [`results/roberta`](./roberta) | 0 | 1 | 0 | `summary.csv` |
| `run_all_broken_fix` | [`results/run_all_broken_fix`](./run_all_broken_fix) | 0 | 1 | 2 | `summary.csv` |
| `run_broken_fix` | [`results/run_broken_fix`](./run_broken_fix) | 0 | 1 | 2 | `summary.csv` |
| `run_final_verified` | [`results/run_final_verified`](./run_final_verified) | 1 | 1 | 2 | `summary.csv` |
| `run_final_verified_v2` | [`results/run_final_verified_v2`](./run_final_verified_v2) | 1 | 1 | 2 | `summary.csv` |
| `run_quality_` | [`results/run_quality_`](./run_quality_) | 0 | 1 | 2 | `summary.csv` |
| `run_quality_20260826_175056` | [`results/run_quality_20260826_175056`](./run_quality_20260826_175056) | 0 | 1 | 2 | `summary.csv` |
| `run_quality_20260828_023053` | [`results/run_quality_20260828_023053`](./run_quality_20260828_023053) | 0 | 1 | 2 | `summary.csv` |
| `run_quality_all_screening` | [`results/run_quality_all_screening`](./run_quality_all_screening) | 0 | 1 | 2 | `summary.csv` |
| `run_quality_clean` | [`results/run_quality_clean`](./run_quality_clean) | 0 | 1 | 2 | `summary.csv` |
| `run_quality_ep1` | [`results/run_quality_ep1`](./run_quality_ep1) | 1 | 1 | 2 | `summary.csv` |
| `run_quality_fast` | [`results/run_quality_fast`](./run_quality_fast) | 0 | 1 | 2 | `summary.csv` |
| `run_quality_fixed_1` | [`results/run_quality_fixed_1`](./run_quality_fixed_1) | 1 | 1 | 2 | `summary.csv` |
| `run_quality_fixed_final` | [`results/run_quality_fixed_final`](./run_quality_fixed_final) | 0 | 1 | 2 | `summary.csv` |
| `run_quality_pretrained` | [`results/run_quality_pretrained`](./run_quality_pretrained) | 0 | 1 | 1 | `summary.csv` |
| `run_quality_recovery` | [`results/run_quality_recovery`](./run_quality_recovery) | 0 | 1 | 2 | `summary.csv` |
| `run_quality_resume` | [`results/run_quality_resume`](./run_quality_resume) | 1 | 1 | 2 | `summary.csv` |
| `run_quality_screening` | [`results/run_quality_screening`](./run_quality_screening) | 0 | 1 | 2 | `summary.csv` |
| `run_quality_screening_fast` | [`results/run_quality_screening_fast`](./run_quality_screening_fast) | 0 | 1 | 2 | `summary.csv` |
| `samba_m3_r128` | [`results/samba_m3_r128`](./samba_m3_r128) | 1 | 1 | 2 | `summary.csv` |
| `samba_m3_r16` | [`results/samba_m3_r16`](./samba_m3_r16) | 1 | 1 | 2 | `summary.csv` |
| `samba_m3_r32` | [`results/samba_m3_r32`](./samba_m3_r32) | 1 | 1 | 2 | `summary.csv` |
| `samba_m3_r64` | [`results/samba_m3_r64`](./samba_m3_r64) | 1 | 1 | 2 | `summary.csv` |
| `samba_moe_r16` | [`results/samba_moe_r16`](./samba_moe_r16) | 1 | 1 | 2 | `summary.csv` |
| `slicegpt_fixed` | [`results/slicegpt_fixed`](./slicegpt_fixed) | 1 | 5 | 12 | `slicegpt/rank_curve.csv` |
| `slicegpt_real_s60` | [`results/slicegpt_real_s60`](./slicegpt_real_s60) | 1 | 5 | 12 | `slicegpt/rank_curve.csv` |
| `svd_projectmodelsdistilbert_mnli_full_mnli_sweep` | [`results/svd_projectmodelsdistilbert_mnli_full_mnli_sweep`](./svd_projectmodelsdistilbert_mnli_full_mnli_sweep) | 0 | 16 | 0 | `-` |
| `token_compressed_r64` | [`results/token_compressed_r64`](./token_compressed_r64) | 1 | 1 | 3 | `summary.csv` |
| `tpa_cp` | [`results/tpa_cp`](./tpa_cp) | 1 | 1 | 3 | `summary.csv` |
| `tpa_r8` | [`results/tpa_r8`](./tpa_r8) | 1 | 1 | 3 | `summary.csv` |
| `tpa_roberta_r8` | [`results/tpa_roberta_r8`](./tpa_roberta_r8) | 1 | 1 | 3 | `summary.csv` |
| `verification_fixed_modules` | [`results/verification_fixed_modules`](./verification_fixed_modules) | 1 | 1 | 2 | `summary.csv` |

---

## 📁 Unified Archive (4 suites)

| Directory Name | Relative Path | Subdirs | CSVs | JSONs | Primary Summary File |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `Unified` | [`results/Unified`](./Unified) | 237 | 845 | 1298 | `unified_mgie_budget/roberta/summary.csv` |
| `_archive_partial` | [`results/_archive_partial`](./_archive_partial) | 22 | 0 | 129 | `-` |
| `archive_unified` | [`results/archive_unified`](./archive_unified) | 235 | 850 | 3730 | `unified_mgie_budget/roberta/summary.csv` |
| `benchmark_results_archived` | [`results/benchmark_results_archived`](./benchmark_results_archived) | 1 | 1 | 810 | `-` |

---

## 📁 Zero-Shot Reasoning Suite (2 suites)

| Directory Name | Relative Path | Subdirs | CSVs | JSONs | Primary Summary File |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `zeroshot` | [`results/zeroshot`](./zeroshot) | 7 | 0 | 7 | `-` |
| `zeroshot_parallel` | [`results/zeroshot_parallel`](./zeroshot_parallel) | 40 | 0 | 414 | `-` |

---
