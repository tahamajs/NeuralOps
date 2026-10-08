# Archived Scripts — October 2026 Cleanup

These scripts were archived during the NeuralOps refactoring.
They are preserved for historical reference but are no longer maintained.

## Why archived?
Most were one-off audit, debug, or sequential runners that have been
superseded by the unified benchmark CLI:

    python -m neuralops.benchmarks run ...

## Replacements

| Archived | Replacement |
|---|---|
| run_kd_all.py | `python -m neuralops.benchmarks run --kd ...` |
| run_niah_all.py | `python -m neuralops.benchmarks run --adapter ruler ...` |
| run_niah_sequential.py | Same as above |
| queue_unified_runs.py | `python -m neuralops.benchmarks run --parallel N` |
| audit_causal_modules.py | Static analysis (no replacement needed) |
| detailed_module_audit.py | Same |
| test_all_causality.py | `pytest neuralops/tests/` |
| prepare_cache_once.py | `python -m neuralops.data.prepare_datasets` |
| run_targeted_b2_recovery.py | Covered by `smoke_test.sh` |

## To restore
git mv scripts/_archive/neuralops_cleanup_2026_10/<file> scripts/
