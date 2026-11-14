# NeuralOps Benchmarks Remediation: FIXES Summary

This document details the architectural, security, reliability, and compatibility remediations performed across the NeuralOps Benchmarks subsystem across sub-tasks SA-B1 through SA-B18.

---

## 1. Overview & Remediation Goals

The benchmark modernization addressed several structural and operational issues:
1. **Zero Mock/Fake Implementations:** Replaced misleading mock fallbacks in benchmark suites (NIAH, serving, KD, triton/cuda) with strict `NotImplementedError` or device-aware skips.
2. **P0 Compatibility & Deprecation:** Preserved backwards compatibility across legacy and v2 execution paths without breaking existing test harnesses or callers.
3. **Security Remediation:**
   - Enforced `weights_only=True` in `torch.load` calls to prevent unsafe deserialization vulnerabilities (CVE remediation).
   - Expanded secret scanning and telemetry redaction to cover 8 industry-standard credential and token patterns (Hugging Face, OpenAI, GitHub PAT, GitHub App/Server, AWS Key ID, Google API, PyPI, generic API key assignments).
4. **PyTorch Modernization:**
   - Migrated deprecated `torch.cuda.amp` calls to modern `torch.amp` (`autocast("cuda")` and `GradScaler("cuda")`).
   - Replaced host-side wall-clock measurements (`time.time()`) with hardware-synchronized `torch.cuda.Event` intervals and device-query-based L2 cache flushes (`flush_l2`).
5. **Execution Robustness & Resource Management:**
   - Added subprocess crash handling (SIGKILL/OOM/EOFError/ConnectionResetError) so single worker failures do not terminate the entire orchestrator.
   - Cross-platform file locking (`filelock`) for Parquet and deduplication ledger indexes.
   - Safe budget fallbacks and multi-seed reproducibility controls.

---

## 2. Remediations by Sub-Task

### SA-B1: Benchmark Registry & Discovery
- **Module:** `neuralops/benchmarks/registry.py`, `discovery.py`, `catalog.py`
- **Fixes:**
  - Implemented lazy registry resolution (`_LazyBenchmarkRegistry`, `get_registry()`) eliminating import-time side effects (no CUDA, network, or file I/O on module load).
  - Added support for meta-execution modes (`all`, `all_attention`, `all_compression`, `pareto_sweep`).
  - Implemented `verify_ops_registry()` to assert valid category mapping across attention and compressor mechanism keys.

### SA-B2: Configuration Schema & Result Standardization
- **Module:** `neuralops/benchmarks/config.py`, `neuralops/benchmarks/results/schema.py`, `neuralops/benchmarks/gates.py`
- **Fixes:**
  - Added CUDA device index validation with bounds checking and proper fallback handling.
  - Enforced `BenchmarkStatus` enum across result records; added clear deprecation warnings when legacy string statuses are normalized.
  - Validated mandatory result schema fields (`run_id`, `timestamp`, `metrics`, `status`, `config_hash`).

### SA-B3: Storage Layer Deduplication & Leases
- **Module:** `neuralops/benchmarks/storage/parquet_backend.py`, `record_validator.py`, `dispatcher.py`
- **Fixes:**
  - Implemented idempotent Parquet record persistence deduplicated by `run_id`.
  - Added strict record validation to reject malformed result payloads before storage.
  - Added automatic release and leak warnings for dispatcher GPU/CPU leases on worker completion or crash.

### SA-B4: Results Ledger & Secret Redaction
- **Module:** `neuralops/benchmarks/results/saver.py`, `summarizer.py`, `index.py`, `hf_sync.py`, `safetensors_io.py`, `dedup.py`
- **Fixes:**
  - Deprecated legacy `save_all_formats`, delegating atomically to `MultiFormatSaver`.
  - Expanded `SECRET_PATTERNS` from 3 to 8 regex patterns covering HF, OpenAI, GitHub, AWS, Google, PyPI, and generic API keys.
  - Added cross-process file locking for deduplication indices.
  - Added `strict=True` option in `safetensors_io` when PyTorch is requested.

### SA-B5: Subprocess Isolation & Crash Resilience
- **Module:** `neuralops/benchmarks/executor.py`, `neuralops/benchmarks/orchestrator.py`
- **Fixes:**
  - Hardened parent/child IPC connection handling (`EOFError`, `ConnectionResetError`) to gracefully report `HARDWARE_ERROR` on SIGKILL or worker OOM.
  - Added backoff retry logic for transient memory pressure and GPU timeout handling.

### SA-B6: Core Benchmark Base, Statistics & Profiling
- **Module:** `neuralops/benchmarks/core/base.py`, `evaluation.py`, `stats.py`, `arguments.py`
- **Fixes:**
  - Dynamic L2 cache flush querying `torch.cuda.get_device_properties(device).L2_cache_size` instead of hardcoding fixed 256MB buffers.
  - Detailed memory fragmentation reporting: `allocated_mb`, `reserved_mb`, and `peak_mb`.
  - Resolved CLI argument conflicts and sanitized flag parsing.

### SA-B7: Data & Model Loaders
- **Module:** `neuralops/benchmarks/core/data_loader.py`, `model_loader.py`
- **Fixes:**
  - Implemented robust fallback branches for dataset splits and offline execution.
  - Added trust remote code sanitization and model initialization kwarg filtering.

### SA-B8: Distillation Core & Modern AMP
- **Module:** `neuralops/benchmarks/core/kd.py`, `neuralops/benchmarks/core/training.py`
- **Fixes:**
  - Added fused KD loss computation optimizing memory footprint and arithmetic intensity.
  - Replaced legacy `torch.cuda.amp` imports with modern `torch.amp` context managers and scalers.
  - Added non-finite loss guards (`math.isnan`, `torch.isinf`) to prevent silent model corruption.

### SA-B9: Attention & Roofline Evaluation Suites
- **Module:** `neuralops/benchmarks/suites/attention.py`, `roofline.py`
- **Fixes:**
  - Robust attention module execution handling sequence length and dimension edge-cases.
  - Accurate theoretical FLOPs and memory bandwidth roofline curves reflecting target GPU specifications.

### SA-B10: Triton & CUDA Kernel Suites
- **Module:** `neuralops/benchmarks/suites/triton_bench.py`, `cuda_bench.py`
- **Fixes:**
  - Aligned canonical Triton module list to 26 kernels (`CANONICAL_TRITON_MODULES`).
  - Removed fake kernel fallback stubs, enforcing `NotImplementedError` or device skips when Triton or CUDA runtimes are unavailable.
  - CUDA Event hardware synchronization for microsecond kernel timing.

### SA-B11: Compression, Serving, NIAH & Qwen KD Suites
- **Module:** `neuralops/benchmarks/suites/compression.py`, `serving.py`, `niah.py`, `qwen_kd.py`
- **Fixes:**
  - Eliminated mock/dummy data outputs in NIAH and serving benchmarks.
  - Real perplexity and accuracy tracking on distillation and compressed evaluation runs.

### SA-B12: KD Runner, Configuration & Atomic Reporting
- **Module:** `neuralops/benchmarks/kd/runner.py`, `config.py`, `reporter.py`
- **Fixes:**
  - Added fallback in KD runner to unfreeze all parameters if no target attention modules are detected.
  - Deterministic configuration hashing via `asdict()` covering all hyperparameters.
  - Atomic report writes using unique temporary file suffixes before file rename.

### SA-B13: Distillation Engine, Teacher Cache & CVE Security
- **Module:** `neuralops/benchmarks/kd/distill_engine.py`, `teacher_cache.py`, `strided_layers.py`
- **Fixes:**
  - Remediation of unsafe deserialization: set `weights_only=True` in `torch.load` calls.
  - Added `min_steps_per_stage` budget guard to guarantee minimum distillation steps.
  - Integrated `filelock` for shared cache synchronization.

### SA-B14: KD Zoo Orchestrator & Adapters
- **Module:** `neuralops/benchmarks/kd/zoo_orchestrator.py`, `neuralops/benchmarks/adapters/`
- **Fixes:**
  - Handled uploader call signature differences with graceful backward-compatible fallback shim.
  - Verified initialization and forward pass across all 10 attention adapters.
  - Updated canonical module definitions.

### SA-B15: CLI Decomposition & Modern Command Dispatch
- **Module:** `neuralops/benchmarks/cli.py`, `neuralops/benchmarks/cli_v2/`
- **Fixes:**
  - Decomposed legacy monolithic CLI into modular subcommands: `benchmark`, `doctor`, `plan`, `results`, `ui`.
  - Handled CPU execution mode without modifying global `os.environ["CUDA_VISIBLE_DEVICES"]` within library space.

### SA-B16: Benchmark Matrix & Multi-Seed Presets
- **Module:** `neuralops/benchmarks/matrix.py`, `multi_seed.py`, `results/resumer.py`
- **Fixes:**
  - Added multi-seed execution harness with deterministic per-seed RNG initialization (`set_seed`).
  - Added crash-safe resumption mechanism using `metrics.jsonl` state logging.

### SA-B17: Legacy Compatibility Entry Points
- **Module:** legacy benchmark wrappers and validation integration
- **Fixes:**
  - Maintained backward compatibility for legacy entry points with deprecation warnings.
  - Routed execution into v2 command pipeline.

### SA-B18: Hugging Face Hub Uploader & Secret Redaction Shim
- **Module:** `neuralops/hub/uploader.py`, `tests/benchmarks/test_hub_uploader_compat.py`
- **Fixes:**
  - Implemented dual-signature support in `AsyncHFHubUploader.submit_upload`:
    1. Canonical: `submit_upload(attn_key, staging_dir, metrics, repo_id=..., zoo_type=...)`
    2. Legacy: `submit_upload(folder_path, repo_id, mechanism_name=..., metrics=...)`
    3. Keyword / Path: handles arbitrary keyword variants and positional `Path` objects.
  - Expanded `_redact_secrets` in `uploader.py` to scrub all 8 credential and token patterns, ensuring zero secret leakage in logs, upload records, or error JSON files.
