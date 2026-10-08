# NeuralOps Codebase Audit Master Roadmap (Phase 7 - Revised)

**Date**: 2026-10-08  
**Scope**: Complete consolidation of Phases 1 through 6 audit findings covering all remaining codebase modules outside `core/`, `benchmarks/`, `attentions/`, and `compression_engine/`.

---

## Executive Summary

| Category | Count | Primary Impact |
| :--- | :--- | :--- |
| **Total Issues Identified** | **784** | Across dead code, duplicates, structural debt, architecture, and tests |
| **Critical Issues (Priority 1)** | 42 | Circular imports, layer violations, 2527-line monolith, 100% duplicate files |
| **Important Issues (Priority 2)** | 185 | God files (>800 lines), duplicate configs/utilities, major test coverage gaps |
| **Nice to Have (Priority 3)** | 557 | Unused imports, naming edge cases, trivial tests, import counts |
| **Estimated Total Effort** | **~79 hours** | Realistic allocation across focused work sessions and parallel GPU runs |

---

## 1. Revised Master Issue Priority Matrix

Scoring Formula:  
$$\text{Score} = \frac{\text{Severity (1--5)} \times \text{Impact (1--5)}}{\text{Effort (1--5)} \times \text{Risk (1--5)}}$$

| Issue Description | Severity | Impact | Effort | Risk | Score | Execution Order |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exact File Duplicate (`runner.py` = `main.py`)** | 4 | 4 | 1 | 1 | **16.00** | **Step 1: Session 6** |
| **Config Class Duplication (16 duplicate classes in `config.py`)** | 4 | 4 | 1 | 1 | **16.00** | **Step 3: Session 4** |
| **Consolidate Utility Duplicates (`count_parameters`, `get_logger`, etc.)** | 3 | 3 | 1 | 1 | **9.00** | **Step 1: Session 6** |
| **Prune 23 Unreferenced Dead Files** | 4 | 3 | 1 | 1 | **12.00** | **Step 2: Session 5** |
| **Prune 80 Dead Functions & 7 Dead Classes** | 3 | 3 | 2 | 1 | **4.50** | **Step 2: Session 5** |
| **Clean 327 Unused Imports & Variables** | 2 | 2 | 1 | 1 | **4.00** | **Step 2: Session 5** |
| **Split 2527-line file (`eval_all_models_datasets.py`)** | 5 | 5 | 4 | 3 | **2.08** | **Step 4: Session 2** |
| **Split God Files (`benchmark_datasets`, `optimizers`, `saver`, `compat`)** | 4 | 4 | 4 | 2 | **2.00** | **Step 5: Session 3** |
| **Circular Import Cycles (9 cycles in Core/Kernels)** | 5 | 5 | 2 | 2 | **6.25** | **Step 6: Session 1** |
| **Fix Test Infrastructure & Hardcoded Paths (45 `sys.path`)** | 3 | 3 | 2 | 1 | **4.50** | **Step 7: Session 7** |
| **Address Critical Test Coverage Gaps (Evaluation, Runners, Optimizers)** | 4 | 4 | 3 | 2 | **2.67** | **Step 8: Session 8** |
| **Expand Missing Test Categories & 101 Modules (Integration, Regression)** | 3 | 3 | 5 | 2 | **0.90** | **Step 9: Sessions 9-12** |

---

## 2. Work Session Breakdown & Corrected Execution Order

### Parallel GPU Tracks (Server-Side)
- **Session 0A: GPU Smoke Test** (Run on GPU Server)
  - Verify CUDA environment, Triton kernel compilation, and single-epoch sanity runs on GPU hardware.
  - Command: `bash neuralops/benchmarks/_scripts/smoke_test.sh`
- **Session 0B: Full 232-Module Benchmark** (Run on GPU Server)
  - Full end-to-end benchmark across all 159 attention modules + 73 compression engines.
  - Runs asynchronously / in parallel while Mac-side code refactoring proceeds.

---

### Corrected Sequential Execution Order (Mac-Side)

#### 1. Session 6: Duplicate Code Consolidation (Effort: 3h, Risk: Low)
- **Goal**: Eliminate identical files and duplicate utility implementations.
- **Tasks**:
  1. Convert [`neuralops/experiment_utils/runner.py`](../neuralops/experiment_utils/runner.py) into a shim re-exporting [`main.py`](../neuralops/experiment_utils/main.py).
  2. Consolidate `count_parameters` across `neuralops/trainer/` into [`neuralops/utils/helpers.py`](../neuralops/utils/helpers.py).
  3. Consolidate `get_logger` into [`neuralops/utils/logger.py`](../neuralops/utils/logger.py).
  4. Consolidate `LoRALinear` into [`neuralops/trainer/methods/lora.py`](../neuralops/trainer/methods/lora.py).
  5. Consolidate `ResultsSaver` into [`neuralops/results_saver/saver.py`](../neuralops/results_saver/saver.py).
- **Success Criteria**: Zero duplicated AST bodies across utilities, registry intact (159+73).

#### 2. Session 5: Dead Code Elimination (Effort: 4h, Risk: Low)
- **Goal**: Safely delete unreferenced dead files, functions, classes, and unused imports.
- **Tasks**:
  1. Delete 23 unreferenced files identified in Phase 2.
  2. Delete 80 dead helper functions and 7 dead classes.
  3. Clean 327 unused imports (F401) and variables (F841) across target directories.
- **Success Criteria**: Clean `pyflakes` pass across cleaned directories.

#### 3. Session 4: Configuration Unification (Effort: 3h, Risk: Low)
- **Goal**: Eliminate 16 duplicated dataclasses between monolithic `config.py` and modular config files.
- **Tasks**:
  1. Re-export dataclasses in [`neuralops/config/config.py`](../neuralops/config/config.py) from modular files (`method_config.py`, `hardware_config.py`, `model_config.py`, `compression_config.py`, etc.).
  2. Delete duplicate dataclass definitions in `config.py`.
- **Success Criteria**: Zero duplicate config classes, backward compatibility preserved.

#### 4. Session 2: Modular Decomposition of `eval_all_models_datasets.py` (Effort: 8h, Risk: High)
- **Goal**: Decompose monolithic 2527-line evaluation harness and extract the 1066-line function.
- **Target Structure**:
  - `eval_all_models_datasets.py` (~150 lines: unified entry point and CLI wrapper).
  - `eval_pipeline.py` (~400 lines: evaluation loop orchestration).
  - `eval_patcher.py` (~350 lines: module substitution & monkey patching).
  - `eval_reporter.py` (~300 lines: metric collection, CSV/JSON reporting).
  - `eval_data.py` (~300 lines: batch generation & collation).
- **Success Criteria**: No file > 500 lines, function > 200 lines eliminated, mock eval test passes.

#### 5. Session 3: God Files Modularization (Effort: 8h, Risk: Medium)
- **Goal**: Split 5 files exceeding 800 lines into cohesive, single-responsibility modules.
- **Affected Files**:
  - [`neuralops/data/benchmark_datasets.py`](../neuralops/data/benchmark_datasets.py) (1052 lines) → `loaders.py`, `synthetic.py`, `manifest.py`.
  - [`neuralops/runners/run_monitor_server.py`](../neuralops/runners/run_monitor_server.py) (987 lines) → `server.py`, `endpoints.py`, `stats.py`.
  - [`neuralops/trainer/optimizers.py`](../neuralops/trainer/optimizers.py) (940 lines) → `adafactor.py`, `adamw.py`, `base.py`.
  - [`neuralops/utils/compat.py`](../neuralops/utils/compat.py) (864 lines) → `param_recovery.py`, `aliases.py`.
  - [`neuralops/results_saver/saver.py`](../neuralops/results_saver/saver.py) (852 lines) → `saver.py`, `formats/`.
- **Success Criteria**: All 5 files broken into sub-500 line modules with preserved public exports.

#### 6. Session 1: Circular Imports & Layer Violations (Effort: 4h, Risk: Medium)
- **Goal**: Resolve all remaining circular import cycles and layer violations (now naturally simpler post-splitting).
- **Affected Files**:
  - Core `_attention_injectors.py` (relocate model family resolution to `core/model_utils.py`).
  - Core `multi_mode_benchmark.py` (remove circular upward shim).
  - Core `registry.py` (decouple eager registration).
  - Core Triton kernels (`_common.py` base types extraction).
- **Success Criteria**: 0 circular imports reported by AST dependency graph.

#### 7. Session 7: Test Infrastructure Hardening (Effort: 3h, Risk: Low)
- **Goal**: Standardize test environment, remove fragile path mutations, and centralize fixtures.
- **Tasks**:
  1. Create root `neuralops/tests/conftest.py` with shared fixtures.
  2. Remove 45 `sys.path.insert()` calls in tests.
  3. Create standard `pytest.ini` with test markers and filter warnings.
- **Success Criteria**: Test suite runs cleanly from repository root.

#### 8. Session 8: Critical Test Coverage Gaps (Effort: 6h, Risk: Low)
- **Goal**: Provide automated unit test coverage for currently 0% covered critical packages.
- **Deliverables**:
  - `test_eval_pipeline.py` (smoke tests for evaluation harness).
  - `test_benchmark_datasets.py` (synthetic dataset generation & registration).
  - `test_optimizers.py` (unit tests for custom optimizers).
  - `test_monitor_server.py` (endpoint test for server router).
- **Success Criteria**: Coverage in `evaluation/`, `runners/`, `trainer/` rises above 50%.

#### 9. Sessions 9–12: Comprehensive Test Coverage Across 101 Modules (Effort: 40h, Risk: Low)
- **Session 9 (10h)**: Test coverage for `neuralops/data/` (cached tensors, augmentations, loaders).
- **Session 10 (10h)**: Test coverage for `neuralops/config/` (dataclass serialization, validation checks).
- **Session 11 (10h)**: Test coverage for `neuralops/analysis/` (plotting, metrics calculation, report generators).
- **Session 12 (10h)**: Integration tests, regression suites, and end-to-end framework verification.
- **Success Criteria**: Codebase unit/integration coverage > 85%.

---

## 3. Revised Execution Timeline & Effort Budget

```
Total Effort: ~79 hours

Parallel GPU Server Track:
├── Session 0A: GPU Smoke Test (neuralops/benchmarks/_scripts/smoke_test.sh)
└── Session 0B: Full 232-Module Benchmark (159 attentions + 73 compressors)

Sequential Mac-Side Refactoring:
Phase A: Clean & Consolidate (Low Risk)
├── Step 1: Session 6 — Duplicate Code Consolidation (3h)
├── Step 2: Session 5 — Dead Code Elimination (4h)
└── Step 3: Session 4 — Configuration Unification (3h)

Phase B: Structural Monolith Decompositions (Medium/High Risk)
├── Step 4: Session 2 — Split eval_all_models_datasets.py (8h)
├── Step 5: Session 3 — Split 5 God Files (8h)
└── Step 6: Session 1 — Circular Imports & Layer Violations (4h)

Phase C: Test Infrastructure & Deep Coverage (Low Risk)
├── Step 7: Session 7 — Test Infrastructure & conftest.py (3h)
├── Step 8: Session 8 — Critical Test Gaps: eval, runners, optimizers (6h)
└── Step 9: Sessions 9-12 — Comprehensive Test Coverage for 101 modules (40h)
```

---

## 4. Success Metrics & Verification Criteria

After **every work session**, verify:
1. **Registry Intact**: Exactly 159 attention modules + 73 compression engines registered.
2. **Test Suite Integrity**: Zero regression failures across existing test suites.
3. **Circular Dependencies**: Zero circular imports.
4. **Modularity Constraints**:
   - Zero files exceeding 800 lines of code.
   - Zero functions exceeding 200 lines of code.
   - Nesting depth strictly $< 6$.
5. **Clean Git State**: Working on `ablation` branch with atomic commits.

---

## 5. Rollback & Safety Protocol

1. Before starting each session, verify working tree is clean.
2. Run baseline verification before touching code:
   ```bash
   python3 -c "from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY; assert len(ATTENTION_REGISTRY) == 159 and len(COMPRESSOR_REGISTRY) == 73"
   ```
3. If any step introduces failures or breaks imports:
   ```bash
   git reset --hard HEAD
   ```
4. Commit only after all sub-tasks and tests succeed.
