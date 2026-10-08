# Changelog

All notable changes to NeuralOps are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-08

### Added
- **Unified Benchmark CLI**: Consolidated sweep and evaluation workflows under `python -m neuralops.benchmarks` (`run`, `resume`, `plan`, `list`, `verify`, `results`).
- **Comprehensive Shared Test Infrastructure**: Introduced `neuralops/tests/conftest.py` with 20+ shared fixtures (device, dummy tensors, registries, tmp paths) and custom pytest markers (`smoke`, `gpu`, `slow`, `triton`, `integration`).
- **Full Registry Registry Key Verification**: Hardened registry containing exactly 159 attention mechanisms and 73 compression engine modules (232 total) with 641 backward-compatible alias mappings.
- **Decomposed Modular Subpackages**:
  - `neuralops/data/benchmark_datasets/`
  - `neuralops/runners/monitor/`
  - `neuralops/trainer/optimizers/`
  - `neuralops/utils/compat/`
  - `neuralops/results_saver/`
  - `neuralops/analysis/module_analysis/`
- **Exhaustive Unit Test Suites**: Full data layer, analysis, config, and runner smoke tests.
- **Documentation**: Added `MIGRATION_GUIDE.md`, `API_REFERENCE.md`, `ARCHITECTURE.md`, and `TEST_FILES_AUDIT.md`.

### Changed
- Refactored `train_task` in `neuralops/trainer/train_lra.py` into decomposed modular helper routines.
- Refactored `main` in `neuralops/evaluation/run_lm_eval_harness.py` into isolated phases.
- Accelerated cold import performance to < 3 seconds via registry lazy loading.

### Fixed
- Fixed 9 circular import cycles across core and utility packages.
- Extracted `neuralops/core/triton_kernels/_types.py` to prevent import issues in headless environments.
- Eliminated 45 legacy `sys.path.insert()` calls across test suites.

### Removed
- Archived 7 obsolete test files testing deprecated versioned module prototypes into `neuralops/tests/_archive/`.
