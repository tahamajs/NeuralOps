# Evidence-Based Caller Map

This map records confirmed callers found by repository search. It is a safety
document for structural cleanup; absence from this table is not proof of dead code.

| Symbol or surface | Definition / canonical owner | Confirmed callers or consumers | Classification |
|---|---|---|---|
| `ATTENTION_REGISTRY` | `neuralops/core/registry.py` and public package exports | `neuralops/__init__.py`, benchmark runners, CLI surfaces, registry tests, exhaustive attention tests | Canonical public registry |
| `COMPRESSOR_REGISTRY` | `neuralops/core/registry.py` and `neuralops/compression/__init__.py` | `neuralops/__init__.py`, benchmark runners, CLI surfaces, compressor tests, exhaustive compressor tests | Canonical public registry |
| `neuralops.attentions.*` | `neuralops/attentions/` compatibility namespace | `neuralops/evaluation/eval_patcher.py`, `tests/test_cli_safety_contracts.py`, `tests/test_track2_spectral_ssm_audit.py`, lazy import machinery | Compatibility surface; keep |
| `neuralops.compression_engine.*` | `neuralops/compression_engine/` compatibility namespace | `neuralops/evaluation/eval_patcher.py`, `tests/test_cli_safety_contracts.py`, `neuralops/compression_engine/__init__.py` alias loader | Compatibility surface; keep |
| `neuralops.kernels.list_cuda_kernels()` | `neuralops/kernels/cuda/` public catalog | unified kernel tests, benchmark catalog code, package exports | Canonical catalog API |
| `neuralops.kernels.list_triton_kernels()` | `neuralops/kernels/triton/` public catalog | unified kernel tests, benchmark catalog code, package exports | Canonical catalog API |
| `neuralops.core.cuda_kernels.*` | core CUDA catalog and implementations | kernel tests, benchmark imports, public catalog wrappers | Canonical backend implementation |
| `neuralops.core.triton_kernels.*` | core Triton catalog and implementations | Triton parity tests, benchmark runners, public catalog wrappers | Canonical backend implementation |
| `_resolve_requested_device` | `scripts/benchmarks/gpu_master_suite.py` | GPU benchmark integrity regression tests and suite `main()` | Fail-closed benchmark gate |
| `benchmark_single_compressor` | `scripts/benchmarks/gpu_master_suite.py` | GPU benchmark integrity regression tests and suite `main()` | Fail-closed benchmark gate |
| `scripts/benchmarks/triton_leaderboard.py` | compatibility CLI wrapper | command-line users and canonical benchmark module | Compatibility orchestration; keep |
| `neuralops/benchmarks/static_catalog.json` | checked-in static 131/72 catalog | benchmark discovery and invariant documentation | Generated/static contract artifact |

## Dynamic import evidence

The following mechanisms prevent filename-only dead-code decisions:

- `neuralops/__init__.py` and `neuralops/_lazy.py` use importlib-based lazy
  resolution and compatibility aliases.
- `neuralops/evaluation/eval_patcher.py` builds module names from registry or
  evaluation inputs.
- `neuralops/export/`, `neuralops/inference/`, and
  `neuralops/experiment_logging/` resolve public modules through import maps.
- `scripts/utils/run_all_notebooks.py` imports CUDA packages from catalog names.

## Cleanup consequence

No listed surface is a delete candidate. A future merge/archive proposal must
include the exact callers, replacement target, compatibility behavior, and test
coverage for every symbol it moves.
