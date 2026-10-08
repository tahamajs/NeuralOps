# Test Files Audit Report

## Summary
- **Total Test Files Discovered**: 250
- **Obsolete / Versioned Test Files Archived**: 7 (moved to `neuralops/tests/_archive/`)
- **Active Test Files Kept**: 243
- **Merged / Deduplicated**: 0

## Archived Files
The following test suites tested legacy unversioned prototypes whose source files have been refactored or consolidated into canonical implementations:
1. `test_aegis_v2.py` → `neuralops/tests/_archive/`
2. `test_aegis_v2_comprehensive.py` → `neuralops/tests/_archive/`
3. `test_aegis_v5.py` → `neuralops/tests/_archive/`
4. `test_aegis_v5_comprehensive.py` → `neuralops/tests/_archive/`
5. `test_leka_v1_comprehensive.py` → `neuralops/tests/_archive/`
6. `test_leka_v2_comprehensive.py` → `neuralops/tests/_archive/`
7. `test_leka_v3_comprehensive.py` → `neuralops/tests/_archive/`

## Active Suites Retained
- Data Layer suites: `tests/data/` (cached tensors, synthetic, loaders, preprocessing, glue)
- Config Layer suites: `tests/config/`
- Analysis Layer suites: `tests/analysis/`
- Core Attention suites: 159 module test files
- Core Compressor suites: 73 module test files
- Infrastructure suites: `test_eval_pipeline.py`, `test_eval_imports.py`, `test_runners.py`, `test_runners_imports.py`, `test_optimizers.py`, `test_benchmark_cli.py`
