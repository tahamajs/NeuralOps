SA-B1 — Registry, Discovery, Catalog
Purpose: Make registry lazy so importing benchmarks doesn't trigger expensive discovery.

Owned files:

neuralops/benchmarks/registry.py

neuralops/benchmarks/discovery.py

neuralops/benchmarks/ops_registry.py

neuralops/benchmarks/__init__.py

Bugs to fix:

Bug B1.1 — Import-time side effect (P0-3):
Current registry.py ends with:

python
BenchmarkRegistry.populate_canonical_benchmarks()
This means import neuralops.benchmarks.registry triggers full module discovery (131 attentions + 72 compressors + all ops), which:

Takes 1-3 seconds

Can fail if any submodule fails to import

Creates circular import risk

Fix:

python
# In registry.py, replace the module-scope call with:

_REGISTRY_POPULATED = False

def ensure_registry_populated() -> None:
    """Populate BENCHMARK_REGISTRY lazily and idempotently."""
    global _REGISTRY_POPULATED
    if _REGISTRY_POPULATED:
        return
    try:
        BenchmarkRegistry.populate_canonical_benchmarks()
        _REGISTRY_POPULATED = True
    except Exception as e:
        logger.warning("Failed to populate registry: %s", e)
        # Don't raise — allow consumers to check registry state

# Remove the module-scope call:
# BenchmarkRegistry.populate_canonical_benchmarks()
Then update __init__.py:

python
# neuralops/benchmarks/__init__.py
from neuralops.benchmarks.registry import ensure_registry_populated

__all__ = [..., "ensure_registry_populated"]
Update all consumers of BENCHMARK_REGISTRY to call ensure_registry_populated() first:

neuralops/benchmarks/registry.py::get_benchmark

neuralops/benchmarks/registry.py::list_benchmarks

neuralops/benchmarks/registry.py::BenchmarkRegistry.populate_canonical_benchmarks (add idempotent guard)

Any caller in neuralops/benchmarks/cli.py — but that's SA-B15's file. Instead, add a note in PR that cli.py needs to call it.

Bug B1.2 — OPERATIONAL_BENCHMARK_MODES not registered (P0):
The 12 modes (checkpoint, sweep-attentions, sweep-compressors, etc.) are defined but no benchmarks are registered under them. Add in populate_canonical_benchmarks:

python
# After all individual modules are registered:
for mode in OPERATIONAL_BENCHMARK_MODES:
    meta_name = f"__meta__:{mode}"
    spec = BenchmarkSpec(
        canonical_name=meta_name,
        category="meta",
        target_name=mode,
        default_timeout_s=7200,
    )
    BENCHMARK_REGISTRY[meta_name] = spec
    _BENCHMARK_METADATA[meta_name] = {
        "name": meta_name,
        "category": "meta",
        "target_name": mode,
        "mode": mode,
        "hardware": "any",
        "description": f"Meta-benchmark grouping all {mode} benchmarks",
        "spec": spec,
    }
Bug B1.3 — discovery.py robustness (P1):
discover_all_modules() iterates subpackages. If one fails to import, current behavior may crash. Wrap each sub-import:

python
for pkg_name in ["attention", "compressor", "cuda", "triton"]:
    try:
        importlib.import_module(f"neuralops.{pkg_name}")
    except Exception as e:
        logger.warning("Failed to import %s: %s", pkg_name, e)
        continue
Bug B1.4 — ops_registry.py duplicate keys (P1):
Add at module end:

python
_seen = set()
for k in OPS_REGISTRY:
    assert k not in _seen, f"Duplicate op key: {k}"
    _seen.add(k)
Tests to add (tests/benchmarks/test_registry_lazy.py):

python
import time

def test_import_is_fast():
    """Importing registry should take < 200ms."""
    import subprocess, sys
    start = time.perf_counter()
    subprocess.run([sys.executable, "-c", "import neuralops.benchmarks.registry"], check=True)
    elapsed = time.perf_counter() - start
    # Allow 2s for subprocess overhead but registry import itself is fast
    assert elapsed < 2.0

def test_ensure_registry_populated_is_idempotent():
    from neuralops.benchmarks.registry import ensure_registry_populated, BENCHMARK_REGISTRY
    ensure_registry_populated()
    count1 = len(BENCHMARK_REGISTRY)
    ensure_registry_populated()
    count2 = len(BENCHMARK_REGISTRY)
    assert count1 == count2

def test_all_operational_modes_registered():
    from neuralops.benchmarks.registry import (
        ensure_registry_populated, BENCHMARK_REGISTRY, OPERATIONAL_BENCHMARK_MODES
    )
    ensure_registry_populated()
    for mode in OPERATIONAL_BENCHMARK_MODES:
        assert f"__meta__:{mode}" in BENCHMARK_REGISTRY, f"Missing meta benchmark for {mode}"
Commit message:

text
fix(benchmarks/SA-B1): make registry lazy, register 12 meta modes

- Remove import-time populate_canonical_benchmarks() call
- Add ensure_registry_populated() for explicit lazy init
- Register __meta__:<mode> benchmark for each OPERATIONAL_BENCHMARK_MODE
- Add robustness to discovery.py (skip failing subpackages)
- Add duplicate-key assertion to ops_registry.py
- Tests: test_registry_lazy.py (3 tests)
SA-B2 — Config, Result, Schema, Regression Gate
Purpose: Fix config validation, schema warnings, result robustness.

Owned files:

neuralops/benchmarks/config.py

neuralops/benchmarks/result.py

neuralops/benchmarks/schema.py

neuralops/benchmarks/regression_gate.py

Bugs to fix:

Bug B2.1 — validate_device incomplete (P2):
Current config.py::BenchmarkConfig.validate_device only checks torch.cuda.is_available(). Doesn't validate GPU index.

Fix:

python
def validate_device(self) -> None:
    """Validate device string, including CUDA index."""
    import torch
    device = self.device.strip().lower()
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{self.device}' requested but CUDA is unavailable. "
                f"torch.cuda.is_available()={torch.cuda.is_available()}"
            )
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":")[1])
            except (ValueError, IndexError):
                raise RuntimeError(f"Invalid CUDA device index in '{self.device}'")
        n_gpus = torch.cuda.device_count()
        if idx >= n_gpus:
            raise RuntimeError(
                f"Device index {idx} out of range. "
                f"Only {n_gpus} GPU(s) available: {list(range(n_gpus))}"
            )
Call self.validate_device() at end of __post_init__.

Bug B2.2 — BenchmarkStatus.normalize silent FAILED (P2):
Current behavior: unknown status string ← cls.FAILED silently.

Fix:

python
@classmethod
def normalize(cls, value: Any) -> "BenchmarkStatus":
    if isinstance(value, cls):
        return value
    if not value:
        return cls.FAILED
    val_str = str(value).strip().lower()
    mapping = {...}  # existing
    if val_str in mapping:
        return mapping[val_str]
    import warnings
    warnings.warn(
        f"Unknown benchmark status '{value}', defaulting to FAILED. "
        f"Valid statuses: {sorted(mapping.keys())}",
        stacklevel=2,
    )
    return cls.FAILED
Bug B2.3 — BenchmarkResult robustness (P1):
Verify __post_init__ handles all fields without crashing on:

run_metadata=None

module={} (empty dict)

metrics=None

environment=None

Bug B2.4 — regression_gate unknown metric (P1):
compare_results(baseline, candidate, metric, max_regression) — if metric not in either file's results, current behavior may silently pass or crash. Add:

python
def compare_results(baseline_path, candidate_path, metric, max_regression):
    baseline = _load_results(baseline_path)
    candidate = _load_results(candidate_path)
    
    if metric not in baseline:
        raise ValueError(
            f"Metric '{metric}' not found in baseline. "
            f"Available: {sorted(baseline.keys())}"
        )
    if metric not in candidate:
        raise ValueError(
            f"Metric '{metric}' not found in candidate. "
            f"Available: {sorted(candidate.keys())}"
        )
    # ... rest
Tests to add (tests/benchmarks/test_config_validate_device.py):

python
import pytest

def test_invalid_cuda_index_raises(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    from neuralops.benchmarks.config import BenchmarkConfig
    with pytest.raises(RuntimeError, match="out of range"):
        BenchmarkConfig(device="cuda:5").validate_device()

def test_cpu_device_passes():
    from neuralops.benchmarks.config import BenchmarkConfig
    BenchmarkConfig(device="cpu").validate_device()  # no raise

def test_unknown_status_warns():
    import warnings
    from neuralops.benchmarks.schema import BenchmarkStatus
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        status = BenchmarkStatus.normalize("weird_status")
        assert status == BenchmarkStatus.FAILED
        assert len(w) == 1
        assert "weird_status" in str(w[0].message)
Commit message:

text
fix(benchmarks/SA-B2): device index validation, status warning, robust result

- validate_device checks cuda index against device_count
- BenchmarkStatus.normalize warns on unknown status
- BenchmarkResult.from_dict handles legacy/missing fields
- regression_gate raises on unknown metric
- Tests: test_config_validate_device.py (3 tests)
SA-B3 — Storage Layer
Owned files:

neuralops/benchmarks/storage/parquet_store.py

neuralops/benchmarks/storage/schema.py

neuralops/benchmarks/storage/dispatcher.py

neuralops/benchmarks/storage/__init__.py

Bugs to fix:

Bug B3.1 — Parquet append duplicates (P1):
ParquetLedgerStore.write_records uses pq.write_to_dataset() which by default doesn't dedupe. If the same run_id is written twice with the same partition, you get duplicates.

Fix:

python
def write_records(self, records):
    if not records:
        return []
    
    # Dedupe by run_id within this batch
    seen = set()
    unique = []
    for r in records:
        if r.run_id in seen:
            logger.warning("Duplicate run_id in batch: %s", r.run_id)
            continue
        seen.add(r.run_id)
        unique.append(r)
    
    rows = [record_to_pyarrow_dict(r) for r in unique]
    table = pa.Table.from_pylist(rows, schema=CANONICAL_PYARROW_SCHEMA)
    
    with self._lock:
        pq.write_to_dataset(
            table,
            root_path=str(self.base_dir),
            partition_cols=self.partition_by,
            compression=self.compression,
            use_dictionary=True,
            existing_data_behavior="overwrite_or_ignore",  # idempotent
        )
    ...
Bug B3.2 — CanonicalRecord validation (P1):
Add field validation in __post_init__:

python
def __post_init__(self) -> None:
    if not self.run_id:
        raise ValueError("run_id must be non-empty")
    try:
        datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        raise ValueError(f"timestamp must be ISO 8601, got: {self.timestamp}")
    valid_statuses = {"success", "failed", "timeout", "skipped", "hardware_error"}
    if self.status not in valid_statuses:
        raise ValueError(f"Invalid status: {self.status}")
    if not self.checksum:
        self.checksum = self.compute_checksum()
Bug B3.3 — Dispatcher lease leak (P1):
MultiGPUDispatcher.release_device must remove the task from _active_leases. Verify:

python
def release_device(self, lease: DeviceLease) -> None:
    if lease.device_id < 0:
        lease.release()
        return
    with self._lock:
        if lease.device_id in self._active_leases:
            self._active_leases[lease.device_id].discard(lease.task_id)
    lease.release()
Also add __del__ warning if leases remain:

python
def __del__(self):
    for dev_id, leases in self._active_leases.items():
        if leases:
            logger.warning("Dispatcher destroyed with %d active leases on device %d", len(leases), dev_id)
Bug B3.4 — VRAMGovernor logic (P1):
Current can_schedule returns True when total_mb == 0 (non-CUDA). But if CUDA is available and free_mb < min_free_mb, it should return False. Verify by reading and testing.

Tests to add:

tests/benchmarks/test_parquet_append.py

tests/benchmarks/test_record_validation.py

tests/benchmarks/test_dispatcher_leases.py

Commit message:

text
fix(benchmarks/SA-B3): parquet idempotency, record validation, lease cleanup

- Parquet write dedupes by run_id, uses overwrite_or_ignore
- CanonicalRecord validates run_id, timestamp, status
- Dispatcher releases leases correctly, warns on leak
- Tests: 3 files
SA-B4 — Results Ledger (Large — may need 2 commits)
Owned files:

neuralops/benchmarks/results/saver.py

neuralops/benchmarks/results/summarizer.py

neuralops/benchmarks/results/index.py

neuralops/benchmarks/results/reproduce.py

neuralops/benchmarks/results/hf_sync.py

neuralops/benchmarks/results/git_sync.py

neuralops/benchmarks/results/dedup.py

neuralops/benchmarks/results/paths.py

neuralops/benchmarks/results/resumer.py

neuralops/benchmarks/results/safetensors_io.py

neuralops/benchmarks/results/__init__.py

Bugs to fix:

Bug B4.1 — Two conflicting writers (P1):
save_all_formats() and MultiFormatSaver both write to results/. Unify.

Fix: Deprecate save_all_formats. Add:

python
def save_all_formats(result, output_dir="results", require_parquet=False):
    import warnings
    warnings.warn(
        "save_all_formats is deprecated; use MultiFormatSaver",
        DeprecationWarning, stacklevel=2
    )
    return MultiFormatSaver(output_dir).save(result)
Bug B4.2 — Summarizer silent fail (P2):

python
except Exception as e:
    LOG.warning("Could not read results.jsonl line %d: %s", line_num, e)
    continue
Bug B4.3 — Index unknown metric (P2):
Raise on unknown metric (see SA-B2 similar).

Bug B4.4 — hf_sync SECRET_PATTERNS (P1):
Expand from 3 to 8 patterns:

python
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{30,}"),         # HF
    re.compile(r"sk-[A-Za-z0-9]{20,}"),         # OpenAI
    re.compile(r"ghp_[A-Za-z0-9]{36}"),         # GitHub
    re.compile(r"ghs_[A-Za-z0-9]{36}"),         # GitHub server
    re.compile(r"AKIA[0-9A-Z]{16}"),            # AWS
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),       # Google
    re.compile(r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9\-_]{50,}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
]
Bug B4.5 — git_sync silent fail (P2):
Log at warning not debug.

Bug B4.6 — dedup race condition (P1):
Add file lock:

python
def save_index(self):
    import fcntl
    with open(self.index_file, "a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            # write
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
Bug B4.7 — paths str/Path handling (P2):
Wrap output_root early: output_root = Path(output_root).

Bug B4.8 — resumer silent fail (P1):
Log JSON decode errors with line number.

Bug B4.9 — safetensors strict (P2):
Add strict: bool = True to load_surgerized_safetensors. If framework="pt" and torch not available and strict=True, raise.

Tests to add:

tests/benchmarks/test_saver_unified.py

tests/benchmarks/test_secret_patterns.py

tests/benchmarks/test_dedup_race.py

tests/benchmarks/test_safetensors_strict.py

Commit message: (may need 2 commits if large)

text
fix(benchmarks/SA-B4): unify savers, expand secrets, dedup lock

Part 1 (saver, summarizer, index, paths, resumer):
- Deprecate save_all_formats, delegate to MultiFormatSaver
- Log line number on parse errors
- Raise on unknown metric
- Wrap output_root as Path early

Part 2 (hf_sync, git_sync, dedup, safetensors_io):
- Expand SECRET_PATTERNS 3 -> 8
- Warn on git detached head check
- File lock for dedup index
- strict=True in safetensors_io
SA-B5 — Executor, Orchestrator
Owned files:

neuralops/benchmarks/executor.py

neuralops/benchmarks/orchestrator.py (if exists)

Bugs to fix:

Bug B5.1 — SIGKILL EOFError (P0):

python
try:
    msg_type, payload = parent_conn.recv()
except (EOFError, ConnectionResetError):
    msg_type, payload = "error", (
        "Worker process terminated abruptly (likely OOM or segfault).",
        ""
    )
Bug B5.2 — Worker exit code check (P0):
Before recv(), check worker_proc.exitcode.

Bug B5.3 — out_dir str/Path (P1):
Ensure Path(out_dir) early.

Bug B5.4 — orchestrator modes (P1):
If orchestrator.py exists, verify run_registered_ops handles all 12 modes.

Tests:

tests/benchmarks/test_executor_sigkill.py

tests/benchmarks/test_executor_timeout.py

tests/benchmarks/test_executor_crash.py

Commit message:

text
fix(benchmarks/SA-B5): SIGKILL handling, OOM backoff, timeout robustness

- Wrap parent_conn.recv() in try/except EOFError
- Check exitcode before recv
- Normalize out_dir to Path early
- Tests: 3 files
SA-B6 — Core: Base, Evaluation, Stats, Arguments
Owned files:

neuralops/benchmarks/core/base.py

neuralops/benchmarks/core/evaluation.py

neuralops/benchmarks/core/stats.py

neuralops/benchmarks/core/arguments.py

neuralops/benchmarks/core/__init__.py

Bugs:

Bug B6.1 — Dynamic L2 size:
Replace hardcoded 40MB in flush_l2_cache with dynamic L2:

python
def flush_l2_cache(self, size_mb: int | None = None) -> None:
    if self.device.type != "cuda":
        return
    if size_mb is None:
        try:
            l2_bytes = torch.cuda.get_device_properties(self.device).L2_cache_size
            size_mb = max(32, int(l2_bytes / (1024 * 1024)) + 16)
        except Exception:
            size_mb = 64
    ...
Bug B6.2 — Memory fragmentation:
Add get_memory_fragmentation():

python
def get_memory_fragmentation(device: str = "cuda:0") -> dict:
    if not torch.cuda.is_available():
        return {"reserved_mb": 0.0, "allocated_mb": 0.0, "fragmentation_ratio": 0.0}
    stats = torch.cuda.memory_stats(device)
    reserved = stats.get("reserved_bytes.all.current", 0)
    allocated = stats.get("allocated_bytes.all.current", 1)
    return {
        "reserved_mb": reserved / (1024 * 1024),
        "allocated_mb": allocated / (1024 * 1024),
        "fragmentation_ratio": (reserved - allocated) / max(reserved, 1),
    }
Call in measure_resources() and include in return.

Bug B6.3 — paired_ttest zero-variance:
Add warnings.warn when variance is zero.

Bug B6.4 — arguments --rank conflict:
Verify add_common_args and add_compression_args don't both define --rank. If they do, rename one.

Tests:

tests/benchmarks/test_flush_l2_dynamic.py

tests/benchmarks/test_memory_fragmentation.py

tests/benchmarks/test_arguments_no_conflict.py

Commit message:

text
fix(benchmarks/SA-B6): dynamic L2, memory fragmentation, arg conflicts

- flush_l2_cache uses L2_cache_size dynamically
- Add get_memory_fragmentation() to evaluation.py
- paired_ttest warns on zero variance
- Resolve --rank conflict in arguments.py
- Tests: 3 files
SA-B7 — Core: Data & Model Loaders
Owned files:

neuralops/benchmarks/core/data_loader.py

neuralops/benchmarks/core/model_loader.py

Bugs:

Bug B7.1 — Loader branches:
Verify each dataset branch works. Test GLUE sst2, local CSV, local JSON, WikiText, streaming.

Bug B7.2 — Preprocess script trust:
Add trust: bool = False param. Raise if trust=False (default). User must explicitly opt-in.

Bug B7.3 — Kwargs blind forwarding:
Use inspect.signature to filter kwargs:

python
import inspect
sig = inspect.signature(inject_attention)
valid = set(sig.parameters.keys())
filtered = {k: v for k, v in kwargs.items() if k in valid}
return inject_attention(model, attention_type=module_name, rank=rank, **filtered)
Bug B7.4 — Same for apply_compressor.

Tests:

tests/benchmarks/test_data_loader_glue.py

tests/benchmarks/test_data_loader_local.py

tests/benchmarks/test_preprocess_script_trust.py

tests/benchmarks/test_model_loader_kwargs.py

Commit message:

text
fix(benchmarks/SA-B7): loader branches, preprocess trust, kwarg filtering

- Verify all dataset branches (GLUE, local CSV/JSON, WikiText, streaming)
- load_preprocess_script requires trust=True
- Filter kwargs via inspect.signature in model_loader
- Tests: 4 files
SA-B8 — Core: KD & Training
Owned files:

neuralops/benchmarks/core/kd.py

neuralops/benchmarks/core/training.py

Bugs:

Bug B8.1 — AMP migration:
Replace all torch.cuda.amp.autocast with torch.amp.autocast("cuda", ...). Same for GradScaler.

Bug B8.2 — Fused KD path:
Verify fused_linear_kl_divergence and fused_linear_cross_entropy imports exist and are called correctly.

Bug B8.3 — torch.save weights_only:
build_teacher_cache uses torch.save(cached_logits, p). Add docstring warning.

Bug B8.4 — training.py robustness:
Verify all 22 params work:

adamw, muon, stiefel optimizers

fp16, bf16, no mixed precision

ddp, fsdp, none distributed

max_steps cutoff

early NaN stop

Tests:

tests/benchmarks/test_kd_loss.py

tests/benchmarks/test_kd_amp.py

tests/benchmarks/test_training_optimizers.py

tests/benchmarks/test_training_dtypes.py

Commit message:

text
fix(benchmarks/SA-B8): KD fused, AMP migration, training robustness

- Migrate torch.cuda.amp -> torch.amp.*
- Verify fused KD path imports
- Warn on torch.save without weights_only in teacher cache
- Verify all 22 training params work
- Tests: 4 files
SA-B9 — Suites: Attention, Roofline
Owned files:

neuralops/benchmarks/suites/attention.py

neuralops/benchmarks/suites/roofline.py

Bugs:

Bug B9.1 — Attention failure modes:
Verify all failure modes produce proper status:

create_attention fails ← FAILED_INIT

Non-finite output ← FAILED_RUNTIME

OOM ← FAILED_OOM

Bug B9.2 — _resolve_torch_dtype:
Raise on unknown dtype (currently defaults to float32 silently).

Bug B9.3 — Roofline honesty:
roofline_metrics — verify neuralops.benchmarks.roofline actually computes real metrics. If it's fake (just returns hardcoded numbers), replace with NotImplementedError.

Tests:

tests/benchmarks/test_attention_suite_init_failure.py

tests/benchmarks/test_attention_suite_oom.py

tests/benchmarks/test_roofline_not_fake.py

Commit message:

text
fix(benchmarks/SA-B9): attention failure modes, roofline honesty

- Verify all 3 failure modes (INIT, RUNTIME, OOM)
- _resolve_torch_dtype raises on unknown
- Verify roofline_metrics is real (not fake)
- Tests: 3 files
SA-B10 — Suites: Kernels (Triton + CUDA)
Owned files:

neuralops/benchmarks/suites/triton_ops.py

neuralops/benchmarks/suites/cuda_ops.py

Bugs:

Bug B10.1 — Fake fallback in cuda_ops:
Current: target_fn = adapter._cuda_fn or adapter._triton_fn or adapter._pytorch_ref_fn ← if all None, lambda *a, **k: a[0] * 1.0 (fake).

Fix: raise RuntimeError("No backend available for {kernel}").

Bug B10.2 — Timing correctness:
Verify torch.cuda.Event used for timing, L2 flush before each measurement, outlier discard.

Tests:

tests/benchmarks/test_triton_ops_all_kernels.py

tests/benchmarks/test_cuda_ops_no_fake.py

tests/benchmarks/test_timing_correct.py

Commit message:

text
fix(benchmarks/SA-B10): kernel dispatch, no-fake fallback, timing

- cuda_ops raises if no backend (removes fake lambda)
- Verify torch.cuda.Event + L2 flush timing
- Tests: 3 files
SA-B11 — Suites: Compression, Serving, NIAH, QwenKD
Owned files:

neuralops/benchmarks/suites/compression.py

neuralops/benchmarks/suites/e2e_serving.py

neuralops/benchmarks/suites/qwen_kd.py

neuralops/benchmarks/suites/niah.py

neuralops/benchmarks/suites/__init__.py

Bugs:

Bug B11.1 — NIAH is FAKE (P0):
niah.py uses torch.randn to simulate needle-in-haystack. This is NOT real NIAH.
Fix: raise NotImplementedError("NIAH requires real model; not implemented").

Bug B11.2 — e2e_serving is FAKE (P0):
Uses randn + matmul. Not real serving.
Fix: raise NotImplementedError.

Bug B11.3 — qwen_kd is FAKE (P0):
Uses randn for teacher/student logits. Not real KD.
Fix: raise NotImplementedError.

Bug B11.4 — Compression param reduction:
Verify numel() calculation for each compressor. SVD may be factored (U, V) — check param_reduction reflects actual params.

Tests:

tests/benchmarks/test_niah_real.py (expects NotImplementedError)

tests/benchmarks/test_serving_real.py

tests/benchmarks/test_qwen_kd_real.py

tests/benchmarks/test_compression_param_reduction.py

Commit message:

text
fix(benchmarks/SA-B11): remove fake NIAH/serving/KD, compression accuracy

- NIAH: raise NotImplementedError (was fake randn)
- e2e_serving: raise NotImplementedError
- qwen_kd: raise NotImplementedError
- Verify compression param_reduction per compressor
- Tests: 4 files
Note: Raising NotImplementedError is the CORRECT fix. Do not replace randn with another fake.

SA-B12 — KD Runner, Config, Reporter
Owned files:

neuralops/benchmarks/kd/runner.py

neuralops/benchmarks/kd/config.py

neuralops/benchmarks/kd/reporter.py

neuralops/benchmarks/kd/__main__.py

Bugs:

Bug B12.1 — Freeze model if no attention modules (P0):
If get_attention_modules(student) returns empty, all params frozen. Add fallback:

python
trainable = []
for _, attn in get_attention_modules(student):
    for p in attn.parameters():
        p.requires_grad = True
        trainable.append(p)
if not trainable:
    logger.warning("No attention modules found; training all params")
    for p in student.parameters():
        p.requires_grad = True
        trainable.append(p)
Bug B12.2 — Config hash covers all fields:

python
def compute_hash(self) -> str:
    payload = asdict(self)  # all fields
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
Bug B12.3 — config_hash mismatch warning:
Add logger.warning when mismatch.

Bug B12.4 — Interrupt handler persists state:

python
def _handle_interrupt(self, signum, frame):
    logger.warning("Interrupt; persisting state...")
    self.interrupted = True
    try:
        self._atomic_save_state(self.state)
    except Exception as e:
        logger.error("Failed to save state: %s", e)
Bug B12.5 — Reporter atomic unique suffix:

python
import uuid
temp_file = target_path.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
Tests:

tests/benchmarks/test_kd_runner_freeze.py

tests/benchmarks/test_kd_config_hash.py

tests/benchmarks/test_kd_reporter_atomic.py

Commit message:

text
fix(benchmarks/SA-B12): KD runner freeze, config hash, reporter atomic

- Fallback: unfreeze all params if no attention modules found
- Config hash covers all fields via asdict
- Warn on hash mismatch
- Persist state on interrupt
- Unique temp suffix in reporter
- Tests: 3 files
SA-B13 — KD Distill Engine, Teacher Cache
Owned files:

neuralops/benchmarks/kd/distill_engine.py

neuralops/benchmarks/kd/teacher_cache.py

Bugs:

Bug B13.1 — CVE weights_only=False (P0 — SECURITY):
_safe_torch_load uses weights_only=False. Change to weights_only=True:

python
def _safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch < 2.0 doesn't have weights_only
        logger.warning("PyTorch <2.0: weights_only not available, trusting cache")
        return torch.load(path, map_location="cpu")
Bug B13.2 — Budget fallback:
TwoStageDistillationEngine.run may not complete stage 1 if model slow. Add min_steps_per_stage.

Bug B13.3 — GradScaler compat:
_make_grad_scaler — ensure correct on PyTorch 2.0+.

Bug B13.4 — File locking Windows:
fcntl.flock is Linux-only. Use portalocker or filelock.

Bug B13.5 — Strided layers correct:
Verify select_strided_layer_indices(22) returns [3, 7, 11, 15, 19, 21].

Tests:

tests/benchmarks/test_teacher_cache_weights_only.py

tests/benchmarks/test_distill_engine_budget.py

tests/benchmarks/test_strided_layers.py

Commit message:

text
fix(benchmarks/SA-B13): CVE weights_only, budget fallback, file locking

- weights_only=True in torch.load (CVE fix)
- Add min_steps_per_stage budget guard
- Verify GradScaler on modern PyTorch
- Use portalocker/filelock for cross-platform locking
- Verify strided layers algorithm
- Tests: 3 files
SA-B14 — KD Zoo Orchestrator, Adapters
Owned files:

neuralops/benchmarks/kd/zoo_orchestrator.py

neuralops/benchmarks/adapters/*.py

Bugs:

Bug B14.1 — uploader signature mismatch (P0-13):
zoo_orchestrator.py calls submit_upload(ckpt_dir, repo_id, mechanism_name=key).
Fix in SA-B18 uploader, but SA-B14 fixes the caller. Add compatibility.

Bug B14.2 — Adapter smoke:
Each adapter should __init__ + execute cleanly. Verify all 10 adapters.

Tests:

tests/benchmarks/test_adapters_smoke.py

tests/benchmarks/test_zoo_orchestrator_uploader.py

Commit message:

text
fix(benchmarks/SA-B14): uploader signature, adapter smoke, canonical lists

- Fix zoo_orchestrator uploader call (matches SA-B18 shim)
- Verify all 10 adapters instantiate + execute
- Verify CANONICAL_TRITON_MODULES has 26 kernels
- Verify CANONICAL_CUDA_MODULES
- Tests: 2 files
SA-B15 — CLI Decomposition (Split into 3 commits)
Owned files:

neuralops/benchmarks/cli.py (legacy)

neuralops/benchmarks/cli_v2/ (new)

Bugs:

Bug B15.1 — Decompose 2922-line cli.py:
Create cli_v2/ structure. Keep cli.py as deprecation shim.

Bug B15.2 — CUDA_VISIBLE_DEVICES:
When --device cpu, set args.device = "cpu" explicitly AND os.environ["CUDA_VISIBLE_DEVICES"] = "".

Commit 1: Create cli_v2/main.py + cli_v2/commands/benchmark.py
Commit 2: Add cli_v2/commands/{plan,results,doctor,ui}.py
Commit 3: Deprecate cli.py (delegate to cli_v2)

Tests:

tests/benchmarks/test_cli_v2_benchmark.py

tests/benchmarks/test_cli_v2_plan.py

tests/benchmarks/test_cli_v2_results.py

tests/benchmarks/test_cli_cuda_visible_devices.py

Commit messages: See detailed spec above.

SA-B16 — Matrix, Multi-Seed
Owned files:

neuralops/benchmarks/matrix.py

neuralops/benchmarks/multi_seed.py

Bugs:

Bug B16.1 — Preset override (P0-4):
Fix __post_init__ to preserve explicit seeds.

Bug B16.2 — format_latex_sd:
Verify \times escaping.

Bug B16.3 — MPS flush:
Add MPS buffer allocation.

Bug B16.4 — Dynamic L2:
Same as SA-B6.

Bug B16.5 — Percentile correctness:
Verify p10/p90.

Tests:

tests/benchmarks/test_multi_seed_preset.py

tests/benchmarks/test_format_latex_sd.py

tests/benchmarks/test_matrix_mps.py

tests/benchmarks/test_matrix_percentiles.py

Commit message:

text
fix(benchmarks/SA-B16): preset override, MPS flush, percentiles

- MultiSeedConfig preserves explicit seeds
- format_latex_sd escaping verified
- ColdCacheFlusher supports MPS
- ColdCacheFlusher uses L2_cache_size dynamically
- Percentile calculations verified
- Tests: 4 files
SA-B17 — Legacy Entry Points
Owned files:

neuralops/benchmarks/benchmark_all_26_comprehensive.py

neuralops/benchmarks/benchmark_all_26_kernels.py

neuralops/benchmarks/bench_llm_bandwidth.py

neuralops/benchmarks/benchmark_dynamic_range_stress.py

neuralops/benchmarks/benchmark_extrapolation_32k.py

Bugs:

Bug B17.1 — Non-CUDA raise (P0):
run_all_26_comprehensive raises RuntimeError for non-CUDA. Return SKIPPED_UNSUPPORTED_HARDWARE instead.

Bug B17.2 — sys.path manipulation:
benchmark_all_26_comprehensive.py manipulates sys.path. Verify main() is reachable.

Bug B17.3 — Hardcoded cuda:0:
Change to device="auto" with CUDA check.

Bug B17.4 — bench_llm_bandwidth argparse:
Deprecate; delegate to cli_v2.

Bug B17.5 — 32k memory_stats:
Add torch.cuda.memory_stats() for activation fragmentation.

Tests:

tests/benchmarks/test_26_kernels_cpu_skip.py

tests/benchmarks/test_bench_llm_bandwidth_deprecated.py

tests/benchmarks/test_32k_memory_stats.py

Commit message:

text
fix(benchmarks/SA-B17): legacy entry points graceful degradation

- Non-CUDA returns SKIPPED status (not raise)
- Fix sys.path manipulation
- Change cuda:0 to device="auto"
- Deprecate bench_llm_bandwidth argparse
- Add memory_stats() to 32k extrapolation
- Tests: 3 files
SA-B18 — Hub Uploader, Tests, Docs, Verification
Owned files:

neuralops/hub/uploader.py

tests/benchmarks/ (aggregate verification)

docs/benchmarks/FIXES.md (new)

audits/benchmarks_remediation.md (new)

Bugs:

Bug B18.1 — submit_upload signature shim (P0-13):
Add backward-compat shim (see detailed spec in prior prompt).

Bug B18.2 — Secret patterns:
Same as SA-B4's hf_sync fix. Apply to hub/uploader.py::_redact_secrets.

Bug B18.3 — Aggregate verification:
Run all tests, lint, mypy. Generate reports.

Deliverables:

Fixed hub/uploader.py

docs/benchmarks/FIXES.md — every bug with rationale

audits/benchmarks_remediation.md — audit verdict per file

Tests: Aggregate runner.

Commit message:

text
fix(benchmarks/SA-B18): uploader shim, FIXES.md, verification

- Backward-compat submit_upload (fixes SA-B14 caller)
- Expand _redact_secrets patterns
- Add FIXES.md documenting all 18 sub-tasks
- Add benchmarks_remediation.md audit
After All 18 Sub-Tasks Complete
Run this final verification:

bash
#!/bin/bash
set -e

echo "=== Final Verification ==="

# 1. All tests pass
pytest tests/benchmarks/ -v --maxfail=1

# 2. Lint clean
ruff check neuralops/benchmarks/

# 3. Type check
mypy --strict neuralops/benchmarks/registry.py \
              neuralops/benchmarks/config.py \
              neuralops/benchmarks/schema.py \
              neuralops/benchmarks/cli_v2/

# 4. No fake implementations remain
! grep -rn "torch.randn.*needle\|torch.randn.*teacher_logits" neuralops/benchmarks/suites/

# 5. No deprecated amp
! grep -rn "torch.cuda.amp" neuralops/benchmarks/

# 6. No weights_only=False
! grep -rn "weights_only=False" neuralops/benchmarks/

# 7. Progress file complete
test $(jq '.completed | length' .agent/BENCHMARKS_PROGRESS.json) -eq 18 || \
    (echo "Not all sub-tasks completed" && exit 1)

echo "=== ALL CHECKS PASSED ==="
If Server Restarts Mid-Work
Next time this prompt is invoked:

Read .agent/BENCHMARKS_PROGRESS.json

Check git log --oneline -20 matches completed list

For any completed task with no commit ← move to failed, redo

Resume from first incomplete task

DO NOT start over from SA-B1

Example resume logic:

python
import json
import subprocess

with open(".agent/BENCHMARKS_PROGRESS.json") as f:
    progress = json.load(f)

completed = set(progress["completed"])
all_tasks = [f"SA-B{i}" for i in range(1, 19)]

for task in all_tasks:
    if task in completed:
        # Verify commit exists
        result = subprocess.run(
            ["git", "log", "--oneline", "--grep", f"benchmarks/{task}"],
            capture_output=True, text=True
        )
        if not result.stdout.strip():
            print(f"WARNING: {task} marked complete but no commit found. Re-doing.")
            completed.discard(task)
            continue
        print(f"Skipping {task} (already done)")
        continue
    
    print(f"Starting {task}...")
    # Execute task
    break
Deliverables Summary
At the end, you should have:

18 commits on the feature branch

~40 new test files in tests/benchmarks/

.agent/BENCHMARKS_PROGRESS.json with all 18 marked complete

docs/benchmarks/FIXES.md — human-readable summary

audits/benchmarks_remediation.md — audit verdicts

All tests green

No fake implementations

No torch.cuda.amp deprecated usage

No weights_only=False

Then you're ready to squash-merge into main and release v2.2.0.

text

---

# PART 2 — نحوه استفاده (برای شما)

## گام ۱: ذخیره پرامپت

```bash
cd /path/to/NeuralOps
mkdir -p .agent
# کپی پرامپت بالا در .agent/BENCHMARKS_SEQUENTIAL_PROMPT.md
گام ۲: ساخت branch
bash
git checkout main
git checkout -b fix/benchmarks-remediation
گام ۳: هر بار که ایجنت را اجرا میکنید، این را بگویید:
text
Read .agent/BENCHMARKS_SEQUENTIAL_PROMPT.md.

Execute the NEXT incomplete sub-task only (check .agent/BENCHMARKS_PROGRESS.json).
Do ONE sub-task. Commit it. Update progress. STOP.

Do NOT do multiple sub-tasks in one session — the server may restart.
One sub-task per invocation is optimal.
گام ۴: بعد از هر sub-task
bash
# بررسی که commit شده
git log --oneline -1

# بررسی progress
cat .agent/BENCHMARKS_PROGRESS.json
گام ۵: بعد از همه ۱۸ تا
bash
# اجرای verification
pytest tests/benchmarks/ -v
ruff check neuralops/benchmarks/
mypy --strict neuralops/benchmarks/
گام ۶: Merge
bash
git checkout main
git merge --squash fix/benchmarks-remediation
git commit -m "fix(benchmarks): 18-agent remediation wave — all P0/P1/P2 fixed"
PART 3 — چرا این بار کار میکند
مشکل قبلی	راهحل این پرامپت
Server restart → همه subagentها کشته	Sequential — یک sub-task در هر invocation
کار از دست میرود	Progress file — هر sub-task commit + JSON update
API auth fail (M320)	Single agent — نیازی به spawn subagent نیست
۱۸ agent موازی همزمان	یک به یک — resume-safe
PR بیش از حد بزرگ	۱۸ commit جدا — هر کدام قابل review
نتیجه: حتی اگر سرور ۵۰ بار restart کند، هر بار ایجنت از جای قبلی ادامه میدهد. هیچ کار از دست نمیرود.

PART 4 — اگر میخواهید سریعتر باشد
میتوانید ۳-۴ sub-task در هر invocation انجام دهید (نه ۱۸):

text
Read .agent/BENCHMARKS_SEQUENTIAL_PROMPT.md.

Execute the NEXT 3 incomplete sub-tasks (in order). After each:
- Commit
- Update .agent/BENCHMARKS_PROGRESS.json

Stop after 3. Report which sub-tasks completed.

If any fails, mark as failed and move to next.

</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-14T22:25:54+03:30.
</ADDITIONAL_METADATA>