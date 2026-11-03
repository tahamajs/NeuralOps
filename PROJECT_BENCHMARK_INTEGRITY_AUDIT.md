# Benchmark Integrity Audit

This ledger records the benchmark-specific fail-closed audit. It distinguishes
production execution paths from explicitly synthetic demo fixtures.

| Finding | Location | Classification | Disposition | Evidence |
|---|---|---|---|---|
| Explicit CUDA request could be downgraded to CPU | `scripts/benchmarks/gpu_master_suite.py` | Production integrity bug | Fixed: `_resolve_requested_device` raises `RuntimeError` when CUDA is unavailable | `tests/benchmarks/test_gpu_benchmark_integrity.py::test_cuda_request_does_not_fallback_to_cpu` |
| Compressor construction failure could be represented by a generic approximation | `scripts/benchmarks/gpu_master_suite.py` | Production integrity bug | Fixed: real compressor errors are recorded as `ERROR`; no SVD substitute exists | `tests/benchmarks/test_gpu_benchmark_integrity.py::test_failed_compressor_is_not_replaced_by_svd` |
| Backward validation could pass without valid gradients | `scripts/benchmarks/gpu_master_suite.py` | Production integrity bug | Fixed: gradients must exist and every gradient must be finite | `tests/benchmarks/test_gpu_benchmark_integrity.py::test_backward_requires_finite_gradients` |
| Triton leaderboard relabeled repeated CosFormer runs | `scripts/benchmarks/triton_leaderboard.py` | Production integrity bug | Fixed: wrapper delegates to the canonical 26-kernel runner; it does not synthesize or relabel rows | Canonical runner entrypoint and GPU-only contract |
| Triton/CUDA numerical mismatch could be reported as success | `neuralops/benchmarks/benchmark_all_26_comprehensive.py` | Production integrity bug | Fixed: `max_diff` over tolerance produces `FAIL_NUMERICS` | `tests/benchmarks/test_gpu_benchmark_integrity.py::test_direct_kernel_marks_numerical_mismatch_as_failure` |
| Synthetic compressor rows | `scripts/benchmarks/results_exporter.py` | Explicit demo fixture | Retained only behind `--demo`; never used by benchmark runners or imported result paths | `main()` selects fixture only when `args.from_jsonl` is absent and `--demo` is explicit |
| Explicit `device_str` CUDA requests were downgraded to CPU | `neuralops/benchmarks/{ops_runner,benchmark_extrapolation_32k,bench_llm_bandwidth,benchmark_dynamic_range_stress,multi_seed_stress_test_131_72,stress_test_seeds_404_777_999}.py`, `scripts/benchmarks/benchmark_delta_healing.py` | Production integrity bug | Fixed: explicit CUDA requests now raise before allocation when unavailable | Static fallback scan and repository Ruff gate |

## Rules for future changes

- Production benchmark records must originate from an executed module.
- Missing modules, failed construction, unavailable CUDA, missing gradients, and
  numerical mismatches must be represented by an error/failure status or raised.
- Demo fixtures must remain behind an explicit opt-in flag and must not be used
  as defaults by benchmark or export APIs.
- A benchmark result is not evidence of GPU execution unless its recorded device
  was resolved from the requested device and CUDA availability was verified.

## Remaining device-selection classifications

The repository-wide static scan still finds a small number of expressions that
select a default device when no device was explicitly supplied. These are not
equivalent to silently downgrading an explicit CUDA request:

- `neuralops/benchmarks/benchmark_all_26_comprehensive.py` validates its
  resolved device and raises for a GPU-only run without CUDA.
- `neuralops/benchmarks/harness.py`, `adapters/compressor.py`, and
  `pareto_paper_suite.py` use auto-selection defaults for general-purpose
  workflows; their explicit device paths require separate caller-level review.
- `scripts/benchmarks/qwen_7b_transplant.py` and
  `scripts/benchmarks/gpu_stress_heavy_trainer.py` now expose explicit
  `--device` arguments and fail closed for explicit CUDA requests.
- `neuralops/benchmarks/pareto_paper_suite.py` now defaults to CUDA and fails
  closed when the default GPU is unavailable; CPU remains available when passed
  explicitly by a caller.

These paths remain listed as pending audit items rather than being silently
declared compliant. Any future change that adds an explicit `--device cuda`
contract to them must use the same fail-closed check as the canonical runners.
