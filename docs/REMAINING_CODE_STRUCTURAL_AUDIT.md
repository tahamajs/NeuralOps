# Remaining Code Structural Analysis (Phase 4)

**Date**: 2026-10-08  
**Scope**: All source and test files outside `core/`, `benchmarks/`, `attentions/`, and `compression_engine/`.  
**Total Candidate Files Evaluated**: 357 files (145 source files, 212 test/script files).

---

## 1. Executive Summary

| Metric | Count | Details / Threshold |
| :--- | :--- | :--- |
| **Functions > 100 lines** | 39 | Found across evaluation, runners, analysis, utils, optimizers |
| **Functions > 200 lines** | 9 | Heavy monolith functions (e.g. `evaluate_single_combination_opt`, `build_aggregated_stats`) |
| **Functions > 500 lines** | 1 | `eval_all_models_datasets.py:1373` (`evaluate_single_combination_opt`, **1066 lines**) |
| **Files > 800 lines** | 8 | Top: `eval_all_models_datasets.py` (2527 lines), `generate_all_modules_analysis.py` (1177 lines) |
| **Files > 1500 lines** | 1 | `neuralops/evaluation/eval_all_models_datasets.py` (**2527 lines**) |
| **Functions with Depth ≥ 6** | 14 | Max depth: 12 (`run_monitor_server.py:809 do_GET`) |
| **Circular Import Cycles** | 9 | Internal cycles in `neuralops.core.registry` and `triton_kernels` |
| **God Files (>500 lines, >5 classes or >15 funcs)** | 4 | `saver.py` (40 funcs), `benchmark_datasets.py` (24 funcs), `optimizers.py` (24 funcs), `compat.py` (20 funcs) |
| **Files with Imports > 30** | 8 | Top: `eval_all_models_datasets.py` (58 imports), `main.py` (45 imports) |

---

## 2. Long Functions (> 100 lines)

### Top 30 Longest Functions
| File:line | Function Name | Lines | Action |
| :--- | :--- | :--- | :--- |
| [`neuralops/evaluation/eval_all_models_datasets.py:1373`](../neuralops/evaluation/eval_all_models_datasets.py#L1373) | `evaluate_single_combination_opt` | **1066** | Extract sub-routines (setup, inference, metric extraction, logging) |
| [`neuralops/utils/compat.py:278`](../neuralops/utils/compat.py#L278) | `get_recovery_params` | 374 | Split by architecture type / dictionary dispatch |
| [`neuralops/analysis/generate_all_modules_analysis.py:812`](../neuralops/analysis/generate_all_modules_analysis.py#L812) | `main` | 361 | Decompose CLI orchestration into pipeline stages |
| [`neuralops/runners/run_monitor_server.py:186`](../neuralops/runners/run_monitor_server.py#L186) | `build_aggregated_stats` | 357 | Modularize data processing handlers |
| [`neuralops/evaluation/eval_all_models_datasets.py:632`](../neuralops/evaluation/eval_all_models_datasets.py#L632) | `apply_module_to_model` | 287 | Dispatch by module replacement strategy |
| [`neuralops/runners/run_autonomous_48h_pipeline.py:337`](../neuralops/runners/run_autonomous_48h_pipeline.py#L337) | `build_all_streams` | 265 | Extract stream builders into dedicated factories |
| [`neuralops/data/benchmark_datasets.py:489`](../neuralops/data/benchmark_datasets.py#L489) | `_register_all_benchmarks` | 257 | Refactor into modular benchmark manifest/registry table |
| [`neuralops/runners/unified_orchestrator.py:34`](../neuralops/runners/unified_orchestrator.py#L34) | `run_orchestrator` | 215 | Break into initialization, dispatch, and monitoring steps |
| [`neuralops/trainer/train_lra.py:98`](../neuralops/trainer/train_lra.py#L98) | `train_task` | 202 | Split training loop, validation, and checkpointing |
| [`scripts/run_kd_reuse_model.py:35`](../scripts/run_kd_reuse_model.py#L35) | `main` | 196 | Script CLI helper (acceptable for script) |
| [`neuralops/evaluation/full_mnli_validation.py:144`](../neuralops/evaluation/full_mnli_validation.py#L144) | `main` | 195 | Break into eval runner and result aggregation |
| [`neuralops/data/benchmark_datasets.py:846`](../neuralops/data/benchmark_datasets.py#L846) | `load_benchmark_dataset` | 185 | Delegate to specialized dataset loader strategy |
| [`neuralops/evaluation/run_lm_eval_harness.py:182`](../neuralops/evaluation/run_lm_eval_harness.py#L182) | `main` | 176 | Separate CLI args handling from evaluation run |
| [`neuralops/trainer/optimizers.py:224`](../neuralops/trainer/optimizers.py#L224) | `__init__` | 172 | Extract hyperparameter validation and state initialization |
| [`neuralops/main.py:244`](../neuralops/main.py#L244) | `run_train` | 160 | Delegate steps to `neuralops.trainer` |
| [`scripts/run_kd_all.py:101`](../scripts/run_kd_all.py#L101) | `main` | 153 | Script CLI runner |
| [`scripts/audit_causal_modules.py:22`](../scripts/audit_causal_modules.py#L22) | `analyze_module_file` | 153 | Script AST analyzer |
| [`neuralops/utils/replace_attention.py:180`](../neuralops/utils/replace_attention.py#L180) | `_forward_t5` | 153 | T5 attention wrapper hook |
| [`neuralops/data/base.py:51`](../neuralops/data/base.py#L51) | `preprocess_function` | 153 | Decompose per dataset type |
| [`neuralops/analysis/plotting.py:285`](../neuralops/analysis/plotting.py#L285) | `create_comprehensive_plots` | 153 | Split plot generators by metric group |
| [`neuralops/evaluation/eval_all_models_datasets.py:983`](../neuralops/evaluation/eval_all_models_datasets.py#L983) | `get_dataset_batch` | 148 | Delegate batch creation to `neuralops/data/` |
| [`neuralops/runners/run_monitor_server.py:809`](../neuralops/runners/run_monitor_server.py#L809) | `do_GET` | 143 | Extract HTTP endpoint routing table |
| [`neuralops/trainer/utils.py:160`](../neuralops/trainer/utils.py#L160) | `train_model` | 140 | Separate train step, validation, and epoch logging |
| [`neuralops/tests/test_all_compression_engines.py:63`](../neuralops/tests/test_all_compression_engines.py#L63) | `run_single_compressor_test` | 140 | Test helper function |
| [`neuralops/utils/replace_attention.py:39`](../neuralops/utils/replace_attention.py#L39) | `forward` | 139 | Modularize attention replacement forward pass |
| [`neuralops/evaluation/run_baseline_comparison.py:133`](../neuralops/evaluation/run_baseline_comparison.py#L133) | `main` | 137 | Script CLI orchestration |
| [`neuralops/trainer/optimizers.py:451`](../neuralops/trainer/optimizers.py#L451) | `step` | 133 | Break optimizer state update into helper math funcs |
| [`neuralops/evaluation/eval_perplexity.py:211`](../neuralops/evaluation/eval_perplexity.py#L211) | `evaluate_module_ppl` | 130 | Split tokenization, model forward, and loss computation |
| [`neuralops/utils/replace_attention.py:344`](../neuralops/utils/replace_attention.py#L344) | `_try_instantiate` | 129 | Class instantiation fallback ladder |
| [`neuralops/trainer/utils.py:28`](../neuralops/trainer/utils.py#L28) | `clip_grad_norm_factorized` | 129 | Specialized gradient clipping algorithm |

---

## 3. Long Files (> 800 lines)

| File | Lines | Classes | Functions | Split / Refactoring Plan |
| :--- | :--- | :--- | :--- | :--- |
| [`neuralops/evaluation/eval_all_models_datasets.py`](../neuralops/evaluation/eval_all_models_datasets.py) | **2527** | 1 | 15 | **Critical Split**: Split into `pipeline.py`, `model_patcher.py`, `data_loader.py`, `evaluator.py` |
| [`neuralops/analysis/generate_all_modules_analysis.py`](../neuralops/analysis/generate_all_modules_analysis.py) | **1177** | 0 | 7 | Split aggregation logic from plotting and markdown reporting |
| [`neuralops/data/benchmark_datasets.py`](../neuralops/data/benchmark_datasets.py) | **1052** | 3 | 24 | Split into `registry.py`, `synthetic.py`, and `loaders.py` |
| [`neuralops/runners/run_monitor_server.py`](../neuralops/runners/run_monitor_server.py) | **987** | 1 | 13 | Separate HTTP server handlers from HTML rendering and metrics collection |
| [`neuralops/trainer/optimizers.py`](../neuralops/trainer/optimizers.py) | **940** | 3 | 24 | Move each optimizer class (e.g. `Adafactor`, `CustomAdamW`) into separate files |
| [`neuralops/utils/compat.py`](../neuralops/utils/compat.py) | **864** | 1 | 20 | Split module parameter recovery mappings into static config / registries |
| [`neuralops/main.py`](../neuralops/main.py) | **855** | 1 | 13 | Separate subcommands into dedicated CLI modules (`cli/train.py`, `cli/eval.py`) |
| [`neuralops/results_saver/saver.py`](../neuralops/results_saver/saver.py) | **852** | 1 | 40 | Separate format writers (JSON, CSV, TensorBoard, WandB) |

---

## 4. Deep Nesting (Depth ≥ 6)

| Depth | File:line | Function Name | Analysis |
| :--- | :--- | :--- | :--- |
| **12** | [`neuralops/runners/run_monitor_server.py:809`](../neuralops/runners/run_monitor_server.py#L809) | `do_GET` | Deeply nested endpoint path/query parsing with try-except blocks |
| **9** | [`neuralops/evaluation/eval_all_models_datasets.py:1373`](../neuralops/evaluation/eval_all_models_datasets.py#L1373) | `evaluate_single_combination_opt` | Nested model/dataset loops, device contexts, error recovery |
| **8** | [`neuralops/evaluation/eval_all_models_datasets.py:983`](../neuralops/evaluation/eval_all_models_datasets.py#L983) | `get_dataset_batch` | Nested dictionary unpacking, sequence truncation, tensor conversion |
| **7** | [`neuralops/data/benchmark_datasets.py:339`](../neuralops/data/benchmark_datasets.py#L339) | `_generate_synthetic_glue` | Nested task parsing, vocabulary generation, sequence padding |
| **7** | [`neuralops/data/benchmark_datasets.py:846`](../neuralops/data/benchmark_datasets.py#L846) | `load_benchmark_dataset` | Nested caching checks, HuggingFace dataset downloads, fallbacks |
| **7** | [`neuralops/data/base.py:229`](../neuralops/data/base.py#L229) | `__call__` | Nested collation and tensor batching |
| **7** | [`neuralops/trainer/utils.py:28`](../neuralops/trainer/utils.py#L28) | `clip_grad_norm_factorized` | Nested parameter iteration, norm reduction, dimension factorization |
| **6** | [`neuralops/analysis/generate_all_modules_analysis.py:812`](../neuralops/analysis/generate_all_modules_analysis.py#L812) | `main` | Nested file reading, JSON decoding, result collection |
| **6** | [`neuralops/utils/compat.py:723`](../neuralops/utils/compat.py#L723) | `install_projection_aliases` | Nested attribute probing and alias assignment |
| **6** | [`neuralops/evaluation/eval_all_models_datasets.py:440`](../neuralops/evaluation/eval_all_models_datasets.py#L440) | `build_model_instance` | Nested config parsing and architecture checks |
| **6** | [`neuralops/evaluation/eval_all_models_datasets.py:632`](../neuralops/evaluation/eval_all_models_datasets.py#L632) | `apply_module_to_model` | Nested module matching and layer replacement logic |
| **6** | [`neuralops/evaluation/eval_all_models_datasets.py:821`](../neuralops/evaluation/eval_all_models_datasets.py#L821) | `_invoke` | Nested error handling and fallback invocation |
| **6** | [`neuralops/trainer/optimizers.py:224`](../neuralops/trainer/optimizers.py#L224) | `__init__` | Nested parameter group validations |
| **6** | [`neuralops/trainer/optimizers.py:451`](../neuralops/trainer/optimizers.py#L451) | `step` | Nested state update loops over parameter tensors |

---

## 5. Circular Imports Analysis

Across `neuralops/`, 9 circular dependency cycles exist, located in `core/` and registry layers:
1. `neuralops.core.registry → neuralops.attentions → neuralops.core.registry`
2. `neuralops.core.registry → neuralops.compression_engine → neuralops.core.registry`
3. `neuralops.core.triton_kernels._common ↔ neuralops.core.triton_kernels.gqa`
4. `neuralops.core.triton_kernels._common ↔ neuralops.core.triton_kernels.low_rank`
5. `neuralops.core.triton_kernels._common ↔ neuralops.core.triton_kernels.sparse_attn`
6. `neuralops.core.triton_kernels._common ↔ neuralops.core.triton_kernels.state_update`
7. `neuralops.core.triton_kernels._common ↔ neuralops.core.triton_kernels.linear_attn`
8. `neuralops.core._attention_helpers ↔ neuralops.core._attention_extractors`
9. `neuralops.core.triton_kernels ↔ self-import loop`

**Suggested Fix**:
- For `registry ↔ attentions` and `registry ↔ compression_engine`: Ensure `registry` defines registration interfaces without importing modules eagerly; auto-discovery or delayed registration breaks this loop cleanly.
- For `_common ↔ kernels`: Extract shared base classes and constants into an independent `_types.py` or `_base.py`.

---

## 6. Per-Directory Structural Report

| Directory | Files > 800 | Funcs > 200 | Depth ≥ 6 | Top Issues |
| :--- | :---: | :---: | :---: | :--- |
| `neuralops/tests` | 0 | 0 | 0 | Clean structure; well partitioned into discrete tests. |
| `neuralops/trainer` | 1 | 1 | 3 | `optimizers.py` (940 lines), `train_lra.py:train_task` (202 lines). |
| `neuralops/evaluation` | 1 | 2 | 5 | **Worst hotspot**: `eval_all_models_datasets.py` (2527 lines, func of 1066 lines). |
| `neuralops/analysis` | 1 | 1 | 1 | `generate_all_modules_analysis.py` (1177 lines). |
| `neuralops/data` | 1 | 1 | 3 | `benchmark_datasets.py` (1052 lines). |
| `neuralops/runners` | 1 | 3 | 1 | `run_monitor_server.py` (987 lines, depth 12). |
| `neuralops/results_saver` | 1 | 0 | 0 | `saver.py` (852 lines, 40 methods in single class). |
| `neuralops/experiment_utils` | 0 | 0 | 0 | Duplicate `runner.py` / `main.py` (identified in Phase 3). |
| `neuralops/config` | 0 | 0 | 0 | Monolithic duplication in `config.py` (identified in Phase 3). |
| `neuralops/utils` | 1 | 1 | 1 | `compat.py` (864 lines, func of 374 lines). |
| `scripts/` | 0 | 0 | 2 | Scripts contain some standalone nested AST/analysis logic. |

---

## 7. God Files (> 500 lines, > 15 functions or > 5 classes)

| File | Lines | Classes | Functions | Problem & Refactoring Strategy |
| :--- | :---: | :---: | :---: | :--- |
| [`neuralops/results_saver/saver.py`](../neuralops/results_saver/saver.py) | 852 | 1 | 40 | Monolithic `ResultsSaver` handling file I/O, schemas, metadata, formats, and stats. Split by format driver. |
| [`neuralops/data/benchmark_datasets.py`](../neuralops/data/benchmark_datasets.py) | 1052 | 3 | 24 | Combines dataset download, synthetic data generator, format mapping, and caching. Split into submodules. |
| [`neuralops/trainer/optimizers.py`](../neuralops/trainer/optimizers.py) | 940 | 3 | 24 | Houses multiple complex optimizer implementations in one file. Split per optimizer. |
| [`neuralops/utils/compat.py`](../neuralops/utils/compat.py) | 864 | 1 | 20 | Houses parameter extraction, recovery logic, and monkey patches for all HF architectures. Split by family. |

---

## 8. Prioritized Fix Plan

### Priority 1 (CRITICAL)
- **Target**: `neuralops/evaluation/eval_all_models_datasets.py` (2527 lines, `evaluate_single_combination_opt` 1066 lines).
  - *Action*: Decompose into submodules (`eval_pipeline.py`, `eval_patcher.py`, `eval_reporter.py`).
  - *Effort*: 6-8 hours | *Risk*: High (central evaluation harness).
- **Target**: Core Triton and Registry circular imports.
  - *Action*: Isolate type definitions and move eager module loading out of `registry.py`.
  - *Effort*: 2-3 hours | *Risk*: Medium.

### Priority 2 (IMPORTANT)
- **Target**: Monolithic God Files (`run_monitor_server.py`, `generate_all_modules_analysis.py`, `benchmark_datasets.py`, `optimizers.py`, `saver.py`).
  - *Action*: Modularize into logical subcomponents.
  - *Effort*: 4-6 hours | *Risk*: Low/Medium.
- **Target**: Functions 200–500 lines (`compat.py:get_recovery_params`, `run_monitor_server.py:build_aggregated_stats`).
  - *Action*: Extract sub-routines and dispatch tables.
  - *Effort*: 3-4 hours | *Risk*: Low.

### Priority 3 (NICE TO HAVE)
- **Target**: Deep nesting (depth ≥ 6) and high import count files (> 30 imports).
  - *Action*: Flatten guard clauses, early return patterns, and modularize imports.
  - *Effort*: 2-3 hours | *Risk*: Low.
