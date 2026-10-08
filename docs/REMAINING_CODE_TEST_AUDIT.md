# Remaining Code Test Coverage & Gap Analysis (Phase 6)

**Date**: 2026-10-08  
**Scope**: All source and test modules outside `core/`, `benchmarks/`, `attentions/`, and `compression_engine/`.  
**Total Target Scope**: 210 test files totaling 28,193 test lines.

---

## 1. Executive Summary

| Metric | Count | Details |
| :--- | :--- | :--- |
| **Total Test Files** | 210 | Located under `neuralops/tests/` |
| **Total Test Lines** | 28,193 lines | Average: 134 lines per test file |
| **Candidate Source Modules Evaluated** | 118 files | Modules outside cleaned directories |
| **Modules with Direct Matching Tests** | 17 files (14.4%) | Direct unit test file exists |
| **Untested Modules (No direct test)** | 101 files (85.6%) | Concentrated in `evaluation/`, `runners/`, `config/`, `data/` |
| **Empty Tests (`pass` / `assert True`)** | 0 | All test functions have actual execution logic |
| **Trivial Tests (≤ 3 lines)** | 85 | Fast smoke assertions checking registry presence |
| **Duplicate Test Bodies** | 2 groups (5 funcs) | Identical RoPE rotation safety checks across 3 files |
| **Test Decorators** | 13 `@pytest.mark.skipif` | Conditional skips for CUDA/GPU hardware unavailability |
| **Test Side Effects** | 45 `sys.path` insertions, 10 `open()` | Isolated; no destructive global environment mutations |

---

## 2. Test Coverage & Gap Analysis per Directory

| Directory | Source Files | Directly Tested | Untested | Coverage Ratio |
| :--- | :---: | :---: | :---: | :---: |
| `neuralops/trainer` | 18 | 2 | 16 | 11.1% |
| `neuralops/evaluation` | 11 | 0 | 11 | 0.0% |
| `neuralops/analysis` | 8 | 1 | 7 | 12.5% |
| `neuralops/data` | 17 | 2 | 15 | 11.8% |
| `neuralops/runners` | 6 | 0 | 6 | 0.0% |
| `neuralops/results_saver` | 2 | 1 | 1 | 50.0% |
| `neuralops/experiment_utils` | 9 | 2 | 7 | 22.2% |
| `neuralops/experiment_logging` | 5 | 1 | 4 | 20.0% |
| `neuralops/tracking` | 3 | 0 | 3 | 0.0% |
| `neuralops/config` | 9 | 0 | 9 | 0.0% |
| `neuralops/utils` | 19 | 6 | 13 | 31.6% |

---

## 3. Notable Untested Source Modules

### Critical Gaps:
- **`neuralops/evaluation/` (0% covered)**:
  - [`eval_all_models_datasets.py`](../neuralops/evaluation/eval_all_models_datasets.py) (2527 lines)
  - [`eval_perplexity.py`](../neuralops/evaluation/eval_perplexity.py)
  - [`run_baseline_comparison.py`](../neuralops/evaluation/run_baseline_comparison.py)
  - [`retraining_orchestrator.py`](../neuralops/evaluation/retraining_orchestrator.py)
- **`neuralops/runners/` (0% covered)**:
  - [`unified_orchestrator.py`](../neuralops/runners/unified_orchestrator.py)
  - [`run_monitor_server.py`](../neuralops/runners/run_monitor_server.py)
  - [`run_autonomous_48h_pipeline.py`](../neuralops/runners/run_autonomous_48h_pipeline.py)
- **`neuralops/data/` (11.8% covered)**:
  - [`benchmark_datasets.py`](../neuralops/data/benchmark_datasets.py) (1052 lines)
  - [`dataset_loader.py`](../neuralops/data/dataset_loader.py)
  - [`cached_tensors.py`](../neuralops/data/cached_tensors.py)
- **`neuralops/trainer/` (11.1% covered)**:
  - [`optimizers.py`](../neuralops/trainer/optimizers.py) (940 lines)
  - [`train_lra.py`](../neuralops/trainer/train_lra.py)
  - [`trainer_svd.py`](../neuralops/trainer/trainer_svd.py)

---

## 4. Duplicate Test Bodies

Two identical test patterns across multiple files:
1. **Group `1e9b102a`** (RoPE & even/odd dimension verification):
   - [`neuralops/tests/test_vspec_comprehensive.py:78`](../neuralops/tests/test_vspec_comprehensive.py#L78)
   - [`neuralops/tests/test_vig_comprehensive.py:79`](../neuralops/tests/test_vig_comprehensive.py#L79)
   - [`neuralops/tests/test_wgqa_comprehensive.py:95`](../neuralops/tests/test_wgqa_comprehensive.py#L95)
2. **Group `df487aac`** (RoPE half odd dimension rotation):
   - [`neuralops/tests/test_decay_linear_attention.py:54`](../neuralops/tests/test_decay_linear_attention.py#L54)
   - [`neuralops/tests/test_cosformer.py:37`](../neuralops/tests/test_cosformer.py#L37)

---

## 5. Missing Test Categories

| Test Category | Present Files | Status |
| :--- | :---: | :--- |
| **Unit Tests** | 205 | Predominant test paradigm across the project |
| **Integration Tests** | 2 | Severely under-represented (`test_integration_*.py`) |
| **End-to-End Tests** | 1 | Only `test_full_framework_end_to_end.py` |
| **Regression Tests** | 0 | No explicit regression suite |
| **Property-Based Tests** | 0 | No Hypothesis property-based testing |
| **Performance Tests** | 2 | `test_benchmarks.py`, `test_speed_regression.py` |

---

## 6. Prioritized Fix Plan

### Priority 1 (CRITICAL GAPS)
- **Target**: Create smoke/unit tests for key orchestrator and pipeline modules:
  - `test_eval_harness_smoke.py` (for `eval_all_models_datasets.py`)
  - `test_benchmark_datasets.py` (for `benchmark_datasets.py`)
  - `test_optimizers.py` (for `trainer/optimizers.py`)
- *Effort*: 6-8 hours | *Risk*: Low (additive tests only).

### Priority 2 (IMPORTANT)
- **Target**: Centralize duplicated test helpers (`test_rope_and_even_odd_dim_safety`) into a shared test utility module `neuralops/tests/conftest.py`.
- **Target**: Expand test coverage for `neuralops/config/` dataclasses and validation checks.
- *Effort*: 3-4 hours | *Risk*: Low.

### Priority 3 (NICE TO HAVE)
- **Target**: Replace 45 hardcoded `sys.path.insert()` calls in tests with standard pytest package discovery.
- **Target**: Add property-based testing for compression rank algorithms using Hypothesis.
- *Effort*: 3-4 hours | *Risk*: Low.
