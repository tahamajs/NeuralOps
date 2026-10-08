# NeuralOps Codebase Audit Master Roadmap (Phase 7)

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
| **Estimated Total Effort** | ~48 hours | Structured across 12 focused work sessions of ~4 hours each |

---

## 1. Master Issue Priority Matrix

Scoring Formula:  
$$\text{Score} = \frac{\text{Severity (1--5)} \times \text{Impact (1--5)}}{\text{Effort (1--5)} \times \text{Risk (1--5)}}$$

| Issue Description | Severity | Impact | Effort | Risk | Score | Assigned Session |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Circular Import Cycles (9 cycles in Core/Kernels)** | 5 | 5 | 2 | 2 | **6.25** | Session 1 |
| **Exact File Duplicate (`runner.py` = `main.py`)** | 4 | 4 | 1 | 1 | **16.00** | Session 6 |
| **Config Class Duplication (16 duplicate classes in `config.py`)** | 4 | 4 | 1 | 1 | **16.00** | Session 4 |
| **Core Layer Violations (`core` → `utils`, `core` → `benchmarks`)** | 5 | 4 | 2 | 2 | **5.00** | Session 1 |
| **Split 2527-line file (`eval_all_models_datasets.py`)** | 5 | 5 | 4 | 3 | **2.08** | Session 2 |
| **Split God Files (`benchmark_datasets`, `optimizers`, `saver`, `compat`)** | 4 | 4 | 3 | 2 | **2.67** | Session 3 |
| **Consolidate Utility Duplicates (`count_parameters`, `get_logger`, etc.)** | 3 | 3 | 1 | 1 | **9.00** | Session 6 |
| **Prune 23 Unreferenced Dead Files** | 4 | 3 | 1 | 1 | **12.00** | Session 5 |
| **Prune 80 Dead Functions & 7 Dead Classes** | 3 | 3 | 2 | 1 | **4.50** | Session 5 |
| **Clean 327 Unused Imports & Variables** | 2 | 2 | 1 | 1 | **4.00** | Session 5 |
| **Fix Test Infrastructure & Hardcoded Paths (45 `sys.path`)** | 3 | 3 | 2 | 1 | **4.50** | Session 7 |
| **Address Critical Test Coverage Gaps (Evaluation, Runners, Optimizers)** | 4 | 4 | 4 | 2 | **2.00** | Session 8 |
| **Expand Missing Test Categories (Integration, Regression)** | 3 | 3 | 3 | 2 | **1.50** | Sessions 9-12 |

---

## 2. Work Session Breakdown (12 Sessions × ~4h)

### Session 1: Critical Architectural Fixes & Layer Isolation
- **Goal**: Resolve all 9 circular imports and 9 layer violations.
- **Affected Files**:
  - [`neuralops/core/_attention_injectors.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/_attention_injectors.py) (relocate model family resolution to `core/model_utils.py`).
  - [`neuralops/core/multi_mode_benchmark.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/multi_mode_benchmark.py) (remove circular upward shim).
  - [`neuralops/core/registry.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/registry.py) (lazy/decoupled module discovery to break eager cycle with attentions/compressors).
  - `neuralops/core/triton_kernels/_common.py` (extract base types to break cycle with kernel operators).
- **Estimated Effort**: 4 hours | **Risk**: Medium | **Success**: 0 circular imports reported by AST scan.

### Session 2: Modular Decomposition of `eval_all_models_datasets.py` (2527 lines)
- **Goal**: Decompose monolithic evaluation harness and extract the 1066-line function.
- **Target Structure**:
  - `neuralops/evaluation/eval_all_models_datasets.py` (~150 lines: unified entry point and CLI wrapper).
  - `neuralops/evaluation/eval_pipeline.py` (~400 lines: evaluation loop orchestration).
  - `neuralops/evaluation/eval_patcher.py` (~350 lines: module substitution & monkey patching).
  - `neuralops/evaluation/eval_reporter.py` (~300 lines: metric collection, CSV/JSON reporting).
  - `neuralops/evaluation/eval_data.py` (~300 lines: batch generation & collation).
- **Estimated Effort**: 4 hours | **Risk**: High | **Success**: No file > 500 lines, existing CLI entry points intact.

### Session 3: God Files Modularization
- **Goal**: Split 5 files exceeding 800 lines into cohesive, single-responsibility modules.
- **Affected Files**:
  - [`neuralops/data/benchmark_datasets.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/data/benchmark_datasets.py) (1052 lines) → `loaders.py`, `synthetic.py`, `manifest.py`.
  - [`neuralops/runners/run_monitor_server.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/runners/run_monitor_server.py) (987 lines) → `server.py`, `endpoints.py`, `stats.py`.
  - [`neuralops/trainer/optimizers.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/trainer/optimizers.py) (940 lines) → `adafactor.py`, `adamw.py`, `base.py`.
  - [`neuralops/utils/compat.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/utils/compat.py) (864 lines) → `param_recovery.py`, `aliases.py`.
  - [`neuralops/results_saver/saver.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/results_saver/saver.py) (852 lines) → `saver.py`, `formats/`.
- **Estimated Effort**: 4 hours | **Risk**: Medium | **Success**: All 5 files broken into sub-500 line modules with preserved public exports.

### Session 4: Configuration Centralization & De-duplication
- **Goal**: Eliminate 16 duplicated dataclasses between monolithic `config.py` and modular files.
- **Affected Files**:
  - [`neuralops/config/config.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py)
  - [`neuralops/config/method_config.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/method_config.py)
  - [`neuralops/config/hardware_config.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/hardware_config.py)
  - [`neuralops/config/model_config.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/model_config.py)
  - [`neuralops/config/compression_config.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/compression_config.py)
- **Action**: Replace duplicated classes in `config.py` with explicit re-exports from modular files.
- **Estimated Effort**: 3 hours | **Risk**: Low | **Success**: Zero duplicate class AST definitions in `neuralops/config/`.

### Session 5: Dead Code Elimination
- **Goal**: Safely delete unreferenced dead files, functions, classes, and unused imports.
- **Actions**:
  - Delete 23 unreferenced files identified in Phase 2.
  - Delete 80 dead helper functions and 7 dead classes.
  - Fix 327 unused imports (F401) and variables (F841) across remaining directories.
- **Estimated Effort**: 4 hours | **Risk**: Low/Medium | **Success**: Clean `pyflakes` lint pass across target directories.

### Session 6: Duplicate Code Consolidation
- **Goal**: Eliminate identical files and duplicate utility implementations.
- **Affected Files**:
  - Convert [`neuralops/experiment_utils/runner.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/experiment_utils/runner.py) into a clean shim re-exporting [`main.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/experiment_utils/main.py).
  - Consolidate 3 copies of `count_parameters` into [`neuralops/utils/helpers.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/utils/helpers.py).
  - Consolidate duplicate `get_logger` into [`neuralops/utils/logger.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/utils/logger.py).
  - Replace duplicate `LoRALinear` in `utils/` with import from `trainer/methods/lora.py`.
- **Estimated Effort**: 3 hours | **Risk**: Low | **Success**: AST duplicate count reduced to 0.

### Session 7: Test Infrastructure Hardening
- **Goal**: Standardize test environment, remove fragile path mutations, and centralize fixtures.
- **Actions**:
  - Create root [`neuralops/tests/conftest.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/tests/conftest.py) with shared dummy models, tensors, and mock devices.
  - Remove 45 `sys.path.insert()` calls in tests.
  - Create standard `pytest.ini` with test markers and filter warnings.
- **Estimated Effort**: 3 hours | **Risk**: Low | **Success**: Tests run seamlessly from repo root without path hacks.

### Session 8: Critical Test Coverage Gaps (Smoke & Unit Tests)
- **Goal**: Provide automated unit test coverage for currently 0% covered critical packages.
- **Deliverables**:
  - `test_eval_pipeline.py` (smoke tests for evaluation harness).
  - `test_benchmark_datasets.py` (synthetic dataset generation & registration).
  - `test_optimizers.py` (unit tests for custom optimizers).
  - `test_monitor_server.py` (endpoint test for server router).
- **Estimated Effort**: 4 hours | **Risk**: Low | **Success**: Coverage in `evaluation/`, `runners/`, `trainer/` rises above 50%.

### Sessions 9–12: Comprehensive Test Suite & Regression Coverage
- **Session 9**: Test coverage for `neuralops/data/` (cached tensors, augmentations, loaders).
- **Session 10**: Test coverage for `neuralops/config/` (dataclass serialization, validation checks).
- **Session 11**: Test coverage for `neuralops/analysis/` (plotting, metrics calculation, report generators).
- **Session 12**: Add integration & regression tests; end-to-end pipeline validation.
- **Estimated Effort**: 16 hours (4h each) | **Risk**: Low | **Success**: Codebase unit/integration coverage > 85%.

---

## 3. Four-Week Execution Timeline

```
Week 1: Core Architecture & Clean-up
├── Mon-Tue: Session 1 (Circular imports & layer violations)
├── Wed:     Session 4 (Config centralization & deduplication)
├── Thu:     Session 6 (Duplicate code consolidation & shims)
└── Fri:     Session 5 (Dead code elimination & pyflakes cleanup)

Week 2: Monolithic Decompositions
├── Mon-Tue: Session 2 (Decompose eval_all_models_datasets.py)
├── Wed-Thu: Session 3 (Modularize 5 God Files)
└── Fri:     Session 7 (Test infrastructure & conftest.py)

Week 3: Test Coverage Expansion
├── Mon-Tue: Session 8 (Critical test gaps: eval, runners, optimizers)
├── Wed:     Session 9 (Data layer test coverage)
├── Thu:     Session 10 (Config layer test coverage)
└── Fri:     Session 11 (Analysis layer test coverage)

Week 4: Final Hardening & Release
├── Mon-Tue: Session 12 (Integration tests & regression suite)
├── Wed:     End-to-end framework verification across CPU/mock environments
├── Thu:     Documentation update, release notes, and cleanup
└── Fri:     Merge branch into main, create release tag
```

---

## 4. Success Metrics & Verification Criteria

After **every work session**, verify:
1. **Registry Intact**: Exactly 159 attention modules + 73 compression engines registered.
2. **Test Suite Integrity**: Zero regression failures across existing test suites.
3. **Circular Dependencies**: Zero circular imports outside intended leaf modules.
4. **Modularity Constraints**:
   - Zero files exceeding 800 lines of code.
   - Zero functions exceeding 200 lines of code.
   - Nesting depth strictly $< 6$.
5. **Clean Git State**: Fully committed, atomic diffs with clean commit logs.

---

## 5. Rollback & Safety Protocol

1. Before starting each session, verify working tree is clean and create a feature checkpoint branch:
   ```bash
   git checkout -b refactor/session-X
   ```
2. Run baseline verification before touching code:
   ```bash
   python3 -c "from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY; assert len(ATTENTION_REGISTRY) == 159 and len(COMPRESSOR_REGISTRY) == 73"
   ```
3. If any step introduces circular dependencies or breaks imports:
   ```bash
   git reset --hard HEAD
   ```
4. Only merge back to `ablation` after all session success criteria are verified.

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation Strategy |
| :--- | :---: | :---: | :--- |
| **Breaking module import paths in external scripts** | Medium | High | Maintain backwards-compatible shims and re-exports in original file locations. |
| **Breaking registry dynamic discovery during refactor** | Medium | High | Automated registry count assertion test before and after every commit. |
| **Loss of evaluation metrics precision during split** | Low | High | Unit test asserting identical outputs on mock inputs before and after refactoring `eval_all_models_datasets.py`. |
| **Import cycle regression when unbundling configs** | Low | Medium | Static AST dependency graph verification run after Session 4. |
| **Test execution hangs on CPU-only environment** | Medium | Medium | Maintain `@pytest.mark.skipif(not torch.cuda.is_available())` on hardware-specific tests. |
