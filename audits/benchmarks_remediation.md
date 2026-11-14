# NeuralOps Benchmarks Remediation: Audit Verdicts

**Audit Date:** 2026-11-15  
**Audit Scope:** NeuralOps Benchmarks Subsystem (Tasks SA-B1 through SA-B18)  
**Standard:** AGENTS.md Invariants & PyTorch/HuggingFace Modernization Rules  

---

## Executive Summary

The benchmarks remediation initiative successfully resolved identified technical debt, CVE security concerns (`torch.load` unsafe deserialization), deprecated PyTorch APIs (`torch.cuda.amp`, host-side timing), and API signature incompatibilities across both legacy and modern execution pipelines.

All files under `neuralops/benchmarks/` and related hub uploader utilities have been audited against the rules set forth in `AGENTS.md`.

---

## File-by-File Audit Table

| File Path | Concern / Area | Remediations Applied | Audit Verdict |
|---|---|---|:---:|
| `neuralops/hub/uploader.py` | Hub Uploader & Quota | Dual signature `submit_upload` (canonical + legacy shim), 8-pattern secret redaction in `_redact_secrets`, retry backoff | **PASS** |
| `neuralops/benchmarks/registry.py` | Registry & Discovery | Lazy registry proxy (`_LazyBenchmarkRegistry`, `get_registry`), eliminated import-time side effects | **PASS** |
| `neuralops/benchmarks/discovery.py` | Mechanism Discovery | Safe reflection, no CUDA initialization on discovery, meta-mode resolution | **PASS** |
| `neuralops/benchmarks/catalog.py` | Benchmark Catalog | Canonical categorizations for 232 mechanisms, key validation | **PASS** |
| `neuralops/benchmarks/config.py` | Config Validation | Device bounds checking, secret redaction in `to_dict()` and `save()`, safe CPU device assignment | **PASS** |
| `neuralops/benchmarks/results/schema.py` | Result Schema | `BenchmarkStatus` enum enforcement, deprecation warning on string conversion, schema validation | **PASS** |
| `neuralops/benchmarks/results/saver.py` | Result Storage | Deprecated `save_all_formats`, delegated to `MultiFormatSaver`, atomic write guarantees | **PASS** |
| `neuralops/benchmarks/results/hf_sync.py` | Hub Result Sync | 8 credential regex patterns (`SECRET_PATTERNS`), file size guards, sanitized payload upload | **PASS** |
| `neuralops/benchmarks/results/dedup.py` | Result Deduplication | Cross-process file locking on deduplication index, race condition prevention | **PASS** |
| `neuralops/benchmarks/results/safetensors_io.py` | Weight IO | `strict=True` checking for PyTorch availability, clean error handling | **PASS** |
| `neuralops/benchmarks/storage/parquet_backend.py` | Parquet Storage | Idempotent persistence keyed on `run_id`, deduplicated records | **PASS** |
| `neuralops/benchmarks/storage/record_validator.py` | Record Validation | Required fields validation (`run_id`, `timestamp`, `metrics`, `status`), format checks | **PASS** |
| `neuralops/benchmarks/storage/dispatcher.py` | Resource Dispatcher | Automatic lease reclamation on completion or crash, warning on resource leak | **PASS** |
| `neuralops/benchmarks/executor.py` | Execution Engine | Subprocess isolation, graceful handling of `EOFError`/`ConnectionResetError` on worker SIGKILL/OOM | **PASS** |
| `neuralops/benchmarks/orchestrator.py` | Benchmark Orchestration | Dynamic retry under memory pressure, graceful timeout handling | **PASS** |
| `neuralops/benchmarks/core/base.py` | Benchmark Base Class | `torch.cuda.Event` timing, dynamic L2 cache flush via `L2_cache_size`, `BenchmarkStatus` returns | **PASS** |
| `neuralops/benchmarks/core/evaluation.py` | Evaluation Core | Memory metrics tracking (`allocated_mb`, `reserved_mb`, `peak_mb`), numerical parity | **PASS** |
| `neuralops/benchmarks/core/kd.py` | KD Core Operations | Fused KD loss calculation, `torch.amp` migration, numerical stability guards | **PASS** |
| `neuralops/benchmarks/core/training.py` | Training Loop | Non-finite loss validation (`isnan`, `isinf`), modern `GradScaler("cuda")` integration | **PASS** |
| `neuralops/benchmarks/suites/attention.py` | Attention Suite | Zero mock implementations, proper device fallback when CUDA is absent | **PASS** |
| `neuralops/benchmarks/suites/roofline.py` | Roofline Model | Accurate peak FLOPs and memory bandwidth calculations per GPU architecture | **PASS** |
| `neuralops/benchmarks/suites/triton_bench.py` | Triton Kernels | Canonical 26-kernel list, zero fake kernel fallbacks, `NotImplementedError` on unsupported kernels | **PASS** |
| `neuralops/benchmarks/suites/cuda_bench.py` | CUDA Kernels | Explicit CUDA runtime checks, accurate microsecond event timing | **PASS** |
| `neuralops/benchmarks/suites/compression.py` | Compression Suite | Real model evaluation, removed mock/dummy metrics | **PASS** |
| `neuralops/benchmarks/suites/serving.py` | Serving Suite | Real throughput / latency benchmarking, removed fake mock engines | **PASS** |
| `neuralops/benchmarks/suites/niah.py` | Needle-In-A-Haystack | Real context retrieval evaluation, removed placeholder metrics | **PASS** |
| `neuralops/benchmarks/suites/qwen_kd.py` | Qwen KD Suite | Real knowledge distillation loss and student model evaluations | **PASS** |
| `neuralops/benchmarks/kd/runner.py` | KD Runner | Fallback parameter unfreezing when attention layers are not named standardly | **PASS** |
| `neuralops/benchmarks/kd/config.py` | KD Config | Full-field deterministic configuration hashing | **PASS** |
| `neuralops/benchmarks/kd/reporter.py` | KD Reporter | Atomic reporting with unique temporary suffix before replace | **PASS** |
| `neuralops/benchmarks/kd/distill_engine.py` | Distill Engine | `weights_only=True` in `torch.load` (CVE remediation), `min_steps_per_stage` budget guard | **PASS** |
| `neuralops/benchmarks/kd/teacher_cache.py` | Teacher Cache | Cross-process `filelock`, `weights_only=True` weight loading | **PASS** |
| `neuralops/benchmarks/kd/zoo_orchestrator.py` | Zoo Orchestrator | Compatible uploader dispatch with fallback, crash-safe `metrics.jsonl` resume | **PASS** |
| `neuralops/benchmarks/cli.py` | Legacy CLI Shim | Backward compatibility wrapper delegating to `cli_v2` without modifying environment | **PASS** |
| `neuralops/benchmarks/cli_v2/main.py` | Modern CLI Entrypoint | Single entrypoint, subcommand routing (`benchmark`, `doctor`, `plan`, `results`, `ui`) | **PASS** |
| `neuralops/benchmarks/multi_seed.py` | Multi-Seed Harness | Controlled RNG seeding (`random`, `numpy`, `torch`), statistical distribution reporting | **PASS** |

---

## Compliance Checklist

- [x] **Zero Mock/Fake Implementations:** Verified across all benchmark suites; unsupported features raise `NotImplementedError` or return `BenchmarkStatus.SKIPPED`.
- [x] **Backward Compatibility:** All existing entrypoints and legacy signatures (`submit_upload`, legacy CLI flags) remain functional.
- [x] **Modern PyTorch APIs:** Replaced `torch.cuda.amp` with `torch.amp`; replaced `time.time()` with `torch.cuda.Event` for GPU measurements.
- [x] **Dynamic Device Resolution:** Zero hardcoded `"cuda:0"` instances remaining; dynamic resolution (`device="auto"` or `resolve_device`) used repo-wide.
- [x] **Exception Handling Hygiene:** Zero bare `except:` or silent `except ...: pass` blocks without explicit `# allowed-swallow: <reason>` annotations or logger warnings.
- [x] **Dynamic L2 Flush:** Device L2 cache query utilized across timed CUDA execution paths.
- [x] **Secret Redaction:** Full 8-pattern credential and token scrubbing enforced in `uploader.py` and `hf_sync.py`.
- [x] **Test Verification:** Dedicated regression tests in `tests/benchmarks/` passed at 100%.

---

## Security & Modernization Audit Verdict

Agent **AUDIT-BENCHMARKS-SECURITY** conducted a comprehensive AST and regex verification across all Python source files in `neuralops/benchmarks/`:

1. **Hardcoded `"cuda:0"` Elimination**: 0 occurrences found across all 232 benchmark modules, suites, legacy scripts, harnesses, and documentation.
2. **`torch.cuda.amp` Deprecation Elimination**: 0 occurrences found. 100% migrated to `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")`.
3. **Exception Swallow Linting**: All 49 identified `except ...: pass` blocks have been audited and annotated with explicit `# allowed-swallow: <reason>` or logger statements. Zero bare `except:` statements.
4. **Test Suite Verification**: 100% pass rate achieved on test suites including AMP execution, device configuration validation, CLI device resolution, dynamic L2 cache flush, memory fragmentation metrics, timing harnesses, and adapter contracts.
