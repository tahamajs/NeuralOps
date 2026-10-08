# Dead Functions Audit Report (Task 5.3 - Report Only)

**Date**: 2026-10-08  
**Scope**: Identification and categorization of unreferenced functions across `neuralops/` and `scripts/`.  
**Status**: **REPORT ONLY** — No code has been modified.

---

## 1. Executive Summary

- **Total Top-Level Functions Analyzed**: 59 functions across the codebase appear exactly once (in their own definition) and have 0 external call sites.
- **Key Insight**: The vast majority of these functions (~53 functions) are **not dead or broken code**, but **valid mathematical, statistical, training, and utility routines** authored for the library's public surface that currently lack end-user call sites in internal scripts.
- **Recommendation**: In accordance with the prompt guidance:
  - **KEEP & EXPORT** the scientific utilities, optimizer/scheduler factories, and analysis tools into their respective package APIs.
  - **DELETE** only internal/private helpers with zero historical or API value.

---

## 2. Categorization & Audit Table

### Category A: Internal / Truly Dead Helpers (Delete Candidates — 4 functions)
| File | Line | Function | Refs | Notes & Recommendation |
|---|:---:|---|:---:|---|
| `neuralops/utils/compat.py` | 269 | `_is_svd_factor_param` | 0 | Private helper superseded by `is_svd_weight` in `model_utils`. Propose **DELETE**. |
| `neuralops/data/benchmark_datasets.py` | 821 | `_collate_benchmark_batch` | 0 | Unused private collation function (superseded by `ProductionDataCollator`). Propose **DELETE**. |
| `neuralops/data/augmentation.py` | 30 | `dummy_back_translation` | 0 | Placeholder stub function returning raw input without augmentation. Propose **DELETE**. |
| `neuralops/runners/run_autonomous_48h_pipeline.py` | 309 | `run_stream_worker` | 0 | Unused helper in the 48h runner script. Propose **DELETE**. |

---

### Category B: Useful Optimizer & Scheduler Factories (Keep & Export — 15 functions)
These functions in `neuralops/trainer/` instantiate standard and specialized optimizers (Muon, Riemannian Stiefel, Lamb, LARS, etc.) and learning rate schedulers:
| File | Line | Function | Recommendation |
|---|:---:|---|---|
| `neuralops/trainer/optimizers.py` | 838 | `create_muon` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/optimizers.py` | 850 | `create_muon_adamw_reg` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/optimizers.py` | 866 | `create_riemannian_stiefel` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/optimizers.py` | 886 | `create_adamw` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/optimizers.py` | 892 | `create_sgd` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/optimizers.py` | 904 | `create_adam` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/optimizers.py` | 910 | `create_adabelief` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/optimizers.py` | 920 | `create_ranger` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/optimizers.py` | 928 | `create_lamb` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/optimizers.py` | 936 | `create_lars` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/schedulers.py` | 12 | `create_linear` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/schedulers.py` | 20 | `create_cosine` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/schedulers.py` | 26 | `create_rop` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/schedulers.py` | 32 | `create_onecycle` | **KEEP** — Export in `neuralops.trainer` |
| `neuralops/trainer/schedulers.py` | 59 | `create_poly` | **KEEP** — Export in `neuralops.trainer` |

---

### Category C: Statistical Testing Suite (Keep & Export — 11 functions)
Rigorous multiple-hypothesis corrections and Bayesian comparison tests in `neuralops/analysis/comparison/statistical_tests.py`:
| File | Line | Function | Recommendation |
|---|:---:|---|---|
| `neuralops/analysis/comparison/statistical_tests.py` | 245 | `holm_correction` | **KEEP** — Public statistical utility |
| `neuralops/analysis/comparison/statistical_tests.py` | 263 | `hochberg_correction` | **KEEP** — Public statistical utility |
| `neuralops/analysis/comparison/statistical_tests.py` | 280 | `finner_correction` | **KEEP** — Public statistical utility |
| `neuralops/analysis/comparison/statistical_tests.py` | 297 | `hommel_correction` | **KEEP** — Public statistical utility |
| `neuralops/analysis/comparison/statistical_tests.py` | 316 | `holland_copenhaver_correction` | **KEEP** — Public statistical utility |
| `neuralops/analysis/comparison/statistical_tests.py` | 391 | `wilcoxon_signed_rank_test` | **KEEP** — Public statistical utility |
| `neuralops/analysis/comparison/statistical_tests.py` | 412 | `sign_test` | **KEEP** — Public statistical utility |
| `neuralops/analysis/comparison/statistical_tests.py` | 433 | `mcnemar_test` | **KEEP** — Public statistical utility |
| `neuralops/analysis/comparison/statistical_tests.py` | 589 | `bayesian_correlated_ttest` | **KEEP** — Public statistical utility |
| `neuralops/analysis/comparison/statistical_tests.py` | 640 | `bayesian_signed_rank_test` | **KEEP** — Public statistical utility |
| `neuralops/analysis/comparison/statistical_tests.py` | 704 | `run_full_statistical_analysis` | **KEEP** — Public statistical utility |

---

### Category D: Utility Helpers (Keep & Export — 29 functions)
Core utilities in `utils/` and `data/` providing profiling, formatting, serialization, architecture inspection, and dataset loading:
- **Helpers & Serialization**: `format_time`, `save_json`, `load_json`, `unflatten_dict`, `resolve_path`, `merge_configs`, `set_log_level`
- **Architecture & Profiling**: `get_compression_targets`, `get_model_info`, `get_layer_count`, `get_hidden_size`, `measure_inference_latency`, `measure_memory_usage`, `profile_context`, `init_output_scale`
- **Qwen Architecture Compatibility**: `is_qwen_attention`, `qwen_config_proxy`, `qwen_projection`, `qwen_module_policy`, `is_architecture_error`
- **Data & Metrics**: `compute_metrics_for_task`, `compute_regression_metrics`, `list_local_datasets`, `load_glue_split`, `load_superglue_dataset`, `generate_zipfian_markov_tokens`, `compute_empirical_entropy`, `setup_trainer`, `load_and_preprocess_data`
- **Recommendation**: **KEEP** — all are robust, reusable utility building blocks.

---

## 3. Proposed Action Summary

| Action | Count | Items |
|---|:---:|---|
| **DELETE (Category A)** | **4** | `_is_svd_factor_param`, `_collate_benchmark_batch`, `dummy_back_translation`, `run_stream_worker` |
| **KEEP (Categories B, C, D)** | **55** | Statistical suite (11), optimizer/scheduler factories (15), utils & data helpers (29) |
