# A3 Paper and Code Inventory (Phase 0)

**Date:** 2026-09-20  
**Status:** COMPLETE  

---

## 1. Primary Source Artifact Paths

* **Paper TeX Root:** `paper/PerModulePaper/a3/main.tex`
* **Paper Modular Sections:** `paper/PerModulePaper/a3/00_meta.tex` through `13_references.tex`, `A_math.tex` through `E_ethics.tex`
* **Consolidated Paper TeX:** `paper/PerModulePaper/a3/paper_full.tex`
* **Primary Compression Engine Code:** `Codes/framework/compression_engine/a3_compressor.py`
* **Supporting Utility Code:** `Codes/framework/compression_engine/pruning_utils.py`
* **Standalone A3 Module Script:** `paper/PerModulePaper/a3/Codes/a3.py`
* **Unified Runner / Benchmark Driver:** `Codes/framework/run_all_4_a3.py`

---

## 2. Table of Label and Ref Pairs

Across all modular `.tex` files in `paper/PerModulePaper/a3/`:
* **Total Defined Labels:** 73
* **Total In-Text References:** 46
* **Undefined / Broken Internal References:** 0 (all referenced labels are defined)
* **External Section Pointer Defect:** Section 11.7 author contributions referenced in `E_ethics.tex:86` (`See 11.7.`) lacked a formal subsection anchor in `11_conclusion.tex`.

### Catalog of Labels by Component:
* **Equations:** `eq:loss_weighted`, `eq:weight_scaled`, `eq:truncated_svd`, `eq:unweighted_factor`, `eq:frobenius_error`, `eq:rank_escalation`, `eq:layernorm_scaling`, `eq:break_even_rank`, `eq:app_weighted_loss_derivation`, `eq:app_exact_recal_scale`, `eq:app_exact_recal_bias`.
* **Tables:** `tab:positioning`, `tab:prior_work_comparison`, `tab:notation_summary`, `tab:architecture_support`, `tab:setup_models`, `tab:a3_hyperparameters`, `tab:main_sst2`, `tab:svd_vs_a3`, `tab:published_comparison`, `tab:repro_expected`, `tab:complexity_analysis`, `tab:parameter_counting_formulation`, `tab:app_notation_summary`, `tab:app_failure_responses`, `tab:app_dependencies`, `tab:app_repro_expected`, `tab:distilbert_full_metrics`, `tab:roberta_full_metrics`, `tab:gpt2_full_metrics`, `tab:distilbert_layer_detail`, `tab:c_multitask_matrix`, `tab:app_full_param_counts`, `tab:app_inference_efficiency`, `tab:app_rank_ablation`, `tab:app_sweep_top`, `tab:app_weighting_ablation`, `tab:app_stabilization_ablation`, `tab:app_loss_trajectory`, `tab:app_compound_quantization`, `tab:failed_lr`, `tab:rank_fidelity`, `tab:negative_variants`, `tab:app_failure_cases`.
* **Figures:** `fig:pipeline`, `fig:results_accuracy`, `fig:singular_value_decay`, `fig:decision_flowchart`, `fig:comparison_methods`.
* **Sections:** `sec:streaming_calibration`, `sec:fisher_weighted_svd`, `sec:frobenius_validation`, `sec:layernorm_recalibration`, `sec:results`, `sec:when_to_use`, `sec:published_comparison`, `sec:conclusion`, `sec:app_algorithm1`, `sec:app_complexity`, `sec:app_parameter_counting`, `sec:app_notation`, `sec:app_defensive_handling`, `sec:app_reproduction`, `sec:app_full_parameters`, `sec:app_inference_profiling`, `sec:app_rank_ablation`, `sec:app_hyperparam_ablations`, `sec:app_recovery_steps`, `sec:app_failure_taxonomy`.

---

## 3. Grounded Verification of Paper Metrics vs. Code and Benchmark Files

| Paper Value / Metric | Paper Location | Source File & Line / Method | Verification Status |
|:---|:---|:---|:---|
| DistilBERT Acc: 90.02%, F1: 0.9043, Params Saved: +1,245,696 | Table 4 (Tab 7), C.1 | `results/a3_tests/fixed_v4/distilbert_sst2_a3_full_metrics.json:4` | VERIFIED |
| RoBERTa Acc: 83.26%, F1: 0.8460, Params Saved: +589,824 | Table 4 (Tab 7), C.2 | `results/unified_a3/roberta/roberta-base_a3_full_metrics.json:4` | VERIFIED |
| GPT-2 Acc: 63.19%, F1: 0.7216, Params Saved: +525,312 | Table 4 (Tab 7), C.3 | `results/unified_a3/gpt2/gpt2_a3_full_metrics.json:4` | VERIFIED |
| DistilBERT baseline: 90.71%, 66,955,010 params | Table 4, C.1, C.7 | `results/a3_tests/fixed_v4/distilbert_sst2_a3_full_metrics.json:2` | VERIFIED |
| RoBERTa baseline: 78.56%, 124,647,170 params | Table 4, C.2, C.7 | `results/unified_a3/roberta/roberta-base_a3_full_metrics.json:2` | VERIFIED |
| GPT-2 baseline: 79.36%, 124,441,344 params | Table 4, C.3, C.7 | `results/unified_a3/gpt2/gpt2_a3_full_metrics.json:2` | VERIFIED |
| DistilBERT factorized layers: 6 compressed, 18 fallback | Table 4, Table 21, Table 35 | `results/a3_tests/fixed_v4/a3_stats.json:12` | VERIFIED |
| RoBERTa factorized layers: 3 compressed, 33 fallback | Table 4, C.2 | `results/unified_a3/roberta/roberta-base_a3_full_metrics.json:6` | VERIFIED |
| GPT-2 factorized layers: 2 compressed, 22 fallback | Table 4, C.3 | `results/unified_a3/gpt2/gpt2_a3_full_metrics.json:6` | VERIFIED |
| Calibration tokens / samples: 500 | Table 6, Table 9 | `a3_compressor.py:250` (`n_calib = 500`) | CODE VERIFIED |
| Max error tolerance: 0.50 (sweep) / 0.40 (conservative) | Table 6, Sec 4.3, 8.2 | `a3_compressor.py:243` (`max_error = 0.50`) | CODE VERIFIED |
| Rank ceiling formula: min(0.9 * r_break_even, 320) | Sec 4.4, Sec A.7 | `a3_compressor.py:521` | CODE VERIFIED |
| Weight scale formula: sqrt(F.clamp(min=1e-8) + 1e-6) | Eq (4.5) | `a3_compressor.py:473` | CODE VERIFIED |

---

## 4. Search for "??" and "TODO"

* Execution of global search across all 21 `.tex` files in `paper/PerModulePaper/a3/`:
  - **`??` Matches:** 0 occurrences
  - **`TODO` / `FIXME` Matches:** 0 occurrences
  - **Missing Table/Section Captions:** 3 tables lacked captions (`12_reproducibility.tex` Table 1, `D_negative_results.tex` Table 4, `E_ethics.tex` Table 1).
  - **Empty Sections:** Section 1.5 (Notation) and Section 2.6 (Comparison Table) lacked narrative text prior to their respective tables.
