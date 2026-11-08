# AGT-61 Verification Report: Release Version 2.1.0 Bump & CHANGELOG [2.1.0]

**Date:** 2026-11-08  
**Scope:** `pyproject.toml`, `neuralops/__init__.py`, `neuralops/_version.py`, `pyproject.cuda.toml`, `CITATION.cff`, `SECURITY.md`, `README.md`, `neuralops/README.md`, `examples/README.md`, `docs/api/*.md`, `CHANGELOG.md`, `docs/changelog.md`, `neuralops/ui/cli.py`, `neuralops/ui/__init__.py`.  
**Status:** **100% COMPLETE & VERIFIED**

---

## Executive Summary
Version 2.1.0 release bump has been executed across the entire NeuralOps repository. Every authoritative metadata source tracked by `scripts/utils/sync_version.py` has been synchronized to `2.1.0` with the canonical release date `2026-11-08`. A comprehensive, strictly grounded `## [2.1.0] - 2026-11-08` entry was authored in `CHANGELOG.md` following Keep-a-Changelog formatting and mirrored to `docs/changelog.md`. Distribution build artifacts (`neuralops-2.1.0.tar.gz` and `neuralops-2.1.0-py3-none-any.whl`) were constructed cleanly and verified via clean virtual environment installation and smoke execution (`neuralops --help` and `neuralops info` passing with exit code 0).

---

## 1. Files Modified & Synchronized

1. **`pyproject.toml`**: Canonical project version updated to `2.1.0`.
2. **`neuralops/_version.py`**: Hardcoded literal updated to `__version__ = "2.1.0"`.
3. **`neuralops/__init__.py`**: Updated `__version__` resolution to prioritize the generated `_version.py` literal, falling back to package metadata and default.
4. **`pyproject.cuda.toml`**: Package version and `neuralops>=2.1.0` dependency synchronized.
5. **`CITATION.cff`**: Version set to `2.1.0` and `date-released` set to `2026-11-08`.
6. **`SECURITY.md`**: Active version updated to `| **2.1.x** | :white_check_mark: Active |`.
7. **`README.md`**: Badges, header, and release links synchronized to `v2.1.0`.
8. **`neuralops/README.md`**: Release reference updated to `v2.1.0`.
9. **`examples/README.md`**: Suite version header updated to `v2.1.0`.
10. **`docs/api/attentions.md`**: Release badge and tag synchronized to `v2.1.0`.
11. **`docs/api/compression.md`**: Release badge and tag synchronized to `v2.1.0`.
12. **`docs/api/integrations.md`**: Release badge and tag synchronized to `v2.1.0`.
13. **`docs/api/kernels.md`**: Release badge and tag synchronized to `v2.1.0`.
14. **`docs/api/results_ledger.md`**: Release badge and tag synchronized to `v2.1.0`.
15. **`CHANGELOG.md`**: Authored full `## [2.1.0] - 2026-11-08` release notes.
16. **`docs/changelog.md`**: Mirrored `CHANGELOG.md` content.
17. **`neuralops/ui/cli.py` & `neuralops/ui/__init__.py`**: Made `uvicorn` and `fastapi` imports lazy so clean standalone CLI wheel installs without UI extras execute `--help` and `info` cleanly.
18. **`tests/test_version_sync.py`**: Updated canonical assertions to `2.1.0` and `2026-11-08`.

---

## 2. Verification Evidence

### 2.1 Synchronization Tool Quality Gate
```text
$ .venv/bin/python scripts/utils/sync_version.py --check
Canonical version (from pyproject.toml): 2.1.0
OK: All downstream version sources match canonical version.
Exit code: 0
```

### 2.2 In-Python Version Introspection
```text
$ .venv/bin/python -c "import neuralops; print(neuralops.__version__)"
2.1.0
Exit code: 0
```

### 2.3 Wheel Packaging Build
```text
$ rm -rf dist/ && .venv/bin/python -m build
...
Successfully built neuralops-2.1.0.tar.gz and neuralops-2.1.0-py3-none-any.whl
Exit code: 0
```

### 2.4 Clean Virtual Environment Smoke Test
```text
$ python3 -m venv /tmp/neuralops_smoke_venv
$ /tmp/neuralops_smoke_venv/bin/pip install dist/neuralops-2.1.0-py3-none-any.whl
...
Successfully installed neuralops-2.1.0 ...

$ /tmp/neuralops_smoke_venv/bin/neuralops --help
╭──────────────────────────── ⚡ NeuralOps Engine ─────────────────────────────╮
│  NeuralOps v2.1.0 — High-Performance Attention Zoo, SVD Compression & Fused  │
│  Triton Kernels                                                              │
...
Exit code: 0

$ /tmp/neuralops_smoke_venv/bin/neuralops info
╭───────────────────────── NeuralOps System Overview ──────────────────────────╮
│ NeuralOps Deep Learning & High-Throughput Attention Suite                    │
│ Framework Version: 2.1.0                                                     │
│ Universal Sub-Quadratic Attention, Continuous Batching, and SVD Compression  │
╰──────────────────────────────────────────────────────────────────────────────╯
                   Hardware & Runtime Environment                   
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Property                  ┃ Value                                ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Python Version            │ 3.14.3                               │
│ PyTorch Version           │ not installed                        │
│ Triton JIT Version        │ Not installed (PyTorch eager active) │
│ CUDA Acceleration         │ Disabled (CPU only)                  │
│ Host OS / Platform        │ Darwin 27.2.0 (arm64)                │
└───────────────────────────┴──────────────────────────────────────┘
...
Status: Production-grade runtime ready.
Exit code: 0
```

### 2.5 Test Suite Results
```text
$ .venv/bin/python -m pytest tests/test_version_sync.py -v
tests/test_version_sync.py::test_all_version_sources_agree PASSED        [ 25%]
tests/test_version_sync.py::test_sync_version_check_passes PASSED        [ 50%]
tests/test_version_sync.py::test_security_md_supports_current_major PASSED [ 75%]
tests/test_version_sync.py::test_cuda_pyproject_forms_match_main PASSED  [100%]
============================== 4 passed in 0.50s ===============================
```

---

## 3. Full CHANGELOG [2.1.0] Release Notes

```markdown
## [2.1.0] - 2026-11-08

### Breaking Changes
- **Trust Remote Code Default (Security):** `trust_remote_code` now defaults to `False` in all model loaders, pipelines, and CLI invocations. Remote untrusted code execution from HuggingFace requires explicit `--trust-remote-code` or `NEURALOPS_TRUST_REMOTE_CODE=1`.
- **Random Model Fallback Removal:** `TextGenerationPipeline` and evaluators fail closed with `RuntimeError` on tokenizer/model failure instead of silently continuing with random weights or synthetic token IDs.
- **Serving Security Defaults:** `neuralops serve` binds strictly to `127.0.0.1` by default and enforces token authentication.
- **Invalid `--benchmark` Argument Handling:** `--benchmark <invalid>` raises `ValueError` with a list of valid names instead of silently executing all benchmarks.
- **CLI Alias Decoupling:** `neuralops benchmark` is now a dedicated command group (`run`, `plan`, `list`, `info`, `compare`, `resume`) instead of an alias for `run`. Use `neuralops benchmark run` or `neuralops run` to launch benchmarks.
- **Adapter Return Shape Modernization:** Benchmark adapters return canonical, strongly-typed `BenchmarkResult` objects rather than unstructured dictionaries or legacy `"PASS"`/`"FAIL"` status strings.
- **Benchmark Manifest Schema URL Key:** The manifest metadata key is standardized to `"$schema"` (was empty string `""`).

### Fixed
- **Elimination of Deferred Silent Error Swallowing (AGT-55, F-84 completion):**
  - **TextGenerationPipeline Decoded Output:** Replaced silent fallback to `str(seq.tolist())` with explicit `RuntimeError` raising when tokenizer decode fails, preventing integer token ID lists from masquerading as generated text. Replaced silent device transfer and tokenizer loading swallows with structured warning logs.
  - **Parallel Healing Fail-Closed Policy:** Replaced silent fallback to single-device healing with explicit `RuntimeError` when `parallel_healing=True` is requested and distributed execution fails.
  - **Extension Attention Discovery:** Narrowed FLA discovery error handling to `ImportError` (logged as warning), while propagating unexpected registry errors.
  - **Offline Checkpoint / Model Fallback:** Replaced silent `local_files_only=True` retry loops in `neuralops/main.py` with explicit warning logging capturing the initial online connection failure before attempting local fallback.
  - **Subpackage & Alias Discovery Logging:** Replaced silent swallows in `neuralops/__init__.py` (`_ModuleAliasFinder`, `_patch_torch_compat`, `_patch_pascal_svd_if_needed`) and `neuralops/compression/__init__.py` (`__dir__`) with structured debug logs.
  - **CI Quality Gate:** Extended `scripts/utils/check_no_silent_swallow.py` to audit all seven core runtime files. Added comprehensive test suite `tests/test_silent_fallbacks_deferred.py` (6 passed).
- **Benchmark Manifest Schema Key & Torch-Free `neuralops info` (AGT-63):**
  - **Manifest Schema Key:** In `neuralops.benchmarks.core.runner.BenchmarkRunner`, fixed the schema URL key from an empty string `""` to `"$schema"`, conforming to `telemetry.manifest.BenchmarkManifestBuilder` and the unified benchmark manifest JSON schema (`schemas/benchmark_manifest_v1.json`).
  - **Torch-Free `neuralops info`:** Made `torch` import lazy/guarded inside `cmd_info()` in `neuralops.cli.info`. The command now runs cleanly and exits 0 in environments where PyTorch is not installed (displaying `"PyTorch Version: not installed"`), while accurately reporting the installed PyTorch version and CUDA capability when PyTorch is present.
  - **Regression Tests:** Added `tests/benchmarks/test_manifest_schema_key.py` and `tests/test_info_without_torch.py` (both 100% passed).

### Added
- **Multi-Phase Benchmark Matrix Orchestrator (AGT-59, Roadmap Gap 6):**
  - **Multi-Axis Grid Orchestrator (`matrix.py`):** Authored `BenchmarkMatrixOrchestrator` and procedural helper `run_benchmark_matrix` in `neuralops.benchmarks.matrix`. Supports parameter sweeps across sequence lengths ($T \in \{128, 512, 1024, 2048\}$), batch sizes ($B \in \{1, 2, 4, 8\}$), hidden dimensions ($D \in \{64, 128, 256, 512\}$), dtypes (FP16, BF16, FP32), and causal masking modes.
  - **Cold-Cache Flushing & Hardware Timing:** Implemented `ColdCacheFlusher` allocating an L2 cache eviction buffer (up to 256 MB) to prevent cache hit contamination during timed iterations. Uses device synchronous event timing and percentile distributions (median, p10, p90, mean, std).
  - **Derived Metric Generation:** Automatically computes tokens/sec throughput ($\frac{B \times T}{\text{latency}}$), arithmetic intensity/TFLOPs, effective memory bandwidth (GB/s), and speedup vs baseline reference.
  - **Workload Categorization & Gating:** Classifies cells into `prefill` ($T \ge 512$), `decode` ($B=1, T \ge 1024$), and `micro` regimes. Implements speedup regression gating flagging any prefill kernel with speedup $< 0.90\times$.
  - **CLI Command Wiring:** Added `neuralops matrix` subcommand to canonical CLI with options for `--modules`, `--seq-lens`, `--batch-sizes`, `--hidden-dims`, `--dtypes`, `--causal`, `--device`, `--warmup`, `--iters`, `--flush-l2`, `--output-dir`, `--plan/--dry-run`, and `--json`.
  - **Persistence & Regression Suite:** Implemented automatic output logging to `matrix_results.jsonl`, `summary.json`, and `matrix_summary.csv`. Added test suite `tests/benchmarks/test_matrix_runner.py` (5 passed).
- **Benchmark Command Group & Alias Collision Resolution (AGT-59, Roadmap Gaps 2 & 6):**
  - **Dedicated Benchmark Command Group:** Removed the overloaded `aliases=["benchmark"]` from `run` subparser and established a clean, dedicated `benchmark` command group delegating to subcommands: `run`, `plan`, `list`, `info`, `compare`, and `resume`.
  - **Breaking Change Migration:** Users running `neuralops benchmark [options]` expecting `run` must now use `neuralops benchmark run [options]` or `neuralops run [options]`.
  - **Non-CUDA Execution Safety:** Guaranteed that `neuralops benchmark list`, `neuralops benchmark info <name>`, and `neuralops benchmark plan --benchmark <name> --device cpu` execute cleanly on CPU-only hosts without initializing CUDA or failing closed.
  - **Compare Subcommand & Numerical Gate:** Wired `neuralops compare --baseline <path> --candidate <path> [--metric <metric>] [--max-regression-pct <pct>]` calling `neuralops.benchmarks.regression_gate:compare_results`. Exits 0 within tolerance and exits 4 (`FAIL_NUMERICS`) on regression.
  - **Dry-Run Plan Inspection:** Added `--dry-run` to `neuralops run`, allowing dry-run grid estimation without allocating GPU tensors while preserving AGT-07 `--benchmark` single-module selection contracts.
  - **Regression Test Suite:** Authored `tests/benchmarks/test_compare_cli.py` (4/4 passed) verifying non-CUDA benchmark listing, benchmark info routing without falling into `handle_run`, and compare 0/4 exit codes.
- **Subprocess Isolation & Watchdog Runner (AGT-58, Roadmap Gap 5):**
  - **Subprocess Isolation:** Isolated benchmark execution into dedicated worker subprocesses using `multiprocessing.get_context("spawn")`, ensuring abnormal process termination (SIGSEGV, SIGABRT, hard exits) cannot terminate the master orchestrator sweep.
  - **Watchdog Timeout Enforcement:** Implemented strict per-benchmark timeout enforcement with SIGTERM/SIGKILL escalation in `neuralops.benchmarks.executor:execute_single`, cleanly recording `status="timeout"` and updating `timed_out=True` upon overrun.
  - **Structured Failure Telemetry:** Guaranteed that crash exit codes and unhandled exception tracebacks are recorded persistently to `events.jsonl` and embedded in canonical `BenchmarkResult` instances.
  - **Regression Suite:** Authored `tests/benchmarks/test_orchestrator_isolation.py` verifying crash isolation, timeout enforcement, and exception traceback capture.
- **Complete Kernel Registration in BENCHMARK_REGISTRY (AGT-57, Roadmap Gap 4):**
  - **Comprehensive Physical Inventory:** Registered all 369 physical modules and kernels (131 attention mechanisms, 72 compressors, 140 CUDA package units, and 26 canonical Triton kernels) in `BENCHMARK_REGISTRY` with explicit `default_timeout_s=1800`.
  - **Dynamic List & Discovery Alignment:** Updated `neuralops.cli.list_cmd` and discovery routines to query `BENCHMARK_REGISTRY` dynamically, ensuring consistent target reporting across commands.
  - **Regression Suite:** Authored `tests/benchmarks/test_registry_completeness.py` verifying coverage, canonical naming, and timeout contracts across all 369 targets.
- **Operational Protocols (AGENTS.md):** Injected 6 critical operational standards into `AGENTS.md` covering §1.5 Multi-Agent & Delegation Protocol, §1.6 Local State & Database Hygiene, §3.3 Reporting Deferred Tasks (GPU/HPC Only), §5.0 Strict Scope Discipline, §6.3 Error Traceback Mandate, and §8.2 Active Self-Monitoring.

### Changed
- **Benchmark Adapter Modernization & Unified Return Type (AGT-56, Roadmap Gap 3):**
  - **BaseBenchmarkAdapter Lifecycle Integration:** All concrete benchmark adapters (`CUDAKernelAdapter`, `TritonKernelAdapter`, `CompressorModuleAdapter`, and newly added `AttentionBenchmarkAdapter`) strictly inherit from `BaseBenchmarkAdapter` and execute the fail-closed 6-gate lifecycle (capability check -> input generation -> reference -> candidate -> numerics -> timing).
  - **Unified Return Shape (`BenchmarkResult`):** Standardized all adapter execution returns to canonical `BenchmarkResult` instances with strongly-typed `BenchmarkStatus` values (`SUCCESS`, `FAILED`, `FAIL_NUMERICS`, `DEPENDENCY_ERROR`), eliminating all raw dictionary shapes and legacy `"PASS"`/`"FAIL"` status strings across adapters.
  - **Real Batch Calibration:** Refactored `CompressorModuleAdapter` to compute relative Frobenius distortion and SQNR exclusively on real calibration batches (`calibration_batch`), strictly upholding AGENTS.md §0.2 zero fake metrics policy.
  - **Attention Benchmark Adapter:** Authored `AttentionBenchmarkAdapter` in `neuralops.benchmarks.adapters.attention` wrapping `ATTENTION_REGISTRY` modules with causal/non-causal invariance verification and autograd backward gradient checking.
  - **Regression & Unified Tests:** Added `tests/benchmarks/test_adapters_unified.py` and updated `tests/benchmarks/test_gpu_benchmark_integrity.py` to assert unified `BenchmarkResult` return shapes across all adapters.
- **Canonical CLI Unification & Shadow CLI Deprecation (AGT-54, Decision 3, F-48, F-50):**
  - **Canonical Parser:** `neuralops.benchmarks.cli:main` is established as the sole canonical entry point for all CLI console scripts (`neuralops`, `neuralops-cli`, `nops`, `nops-cli`) and `python -m neuralops`.
  - **Shadow Entry Points:** Converted `neuralops/cli/main.py` and `neuralops/cli/core/app.py` into thin backward-compatibility shims that emit `DeprecationWarning` and delegate directly to `neuralops.benchmarks.cli:main`.
  - **F-50 Broken Imports Fixed:** In `neuralops/cli/core/commands/system.py`, fixed broken imports to reference `neuralops.cli.doctor`, `neuralops.cli.info`, and `neuralops.export.runtime_compat:check_compatibility`.
  - **Regression Tests:** Extended `tests/cli/test_cli_canonical.py` with `test_no_shadow_cli_imports_missing_modules` and `test_shadow_cli_shims_emit_deprecation_warning`.
- **Exclusion of Template Attention from Canonical Counts & Logging of Triton Fallbacks (F-25, F-26):**
  - **Template Class Non-Canonical Registration (F-25):** Re-registered `YourCustomAttention` (`template_attention`) in `neuralops.layers.experimental.template_attention` with `is_alias=True`, removing it from `ATTENTION_REGISTRY.canonical_keys()` and `distinct_classes()`. The distinct attention class count correctly adjusts from 116 to 115, ensuring educational template modules do not inflate physical registry counts.
  - **Triton Fallback Observability (F-26):** Verified that bare unlogged `except Exception: pass` blocks in `neuralops/compression/low_rank/a3_compressor.py`, `neuralops/layers/hybrid/tri_branch_hybrid_attention.py`, and `neuralops/layers/modern/mla_attention.py` are eliminated and replaced by structured debug logging (`logger.debug("...", exc_info=True)`) or mathematical fallbacks, while preserving Triton-to-PyTorch runtime fallback semantics.
  - **Regression Tests:** Added `test_experimental_not_counted_as_canonical` to `tests/test_registry_counts.py` and regression guard `test_no_bare_except_pass_in_layers_compression` to `tests/test_repo_config_consistency.py` (19 passed).
```

---

## 4. Conclusion
All release version metadata and distribution gates are verified at version **2.1.0**. Proceeding immediately to **AGT-62** (Minimal README rewrite).
