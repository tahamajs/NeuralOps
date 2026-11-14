# AGENTS.md — NeuralOps Benchmark Library: Operational Rules

> **Audience:** Any AI agent (Cursor, Claude Code, Copilot, Aider, etc.) working on this repo.  
> **Status:** Binding. Violating these rules is a blocking review failure.  
> **Last updated:** 2026-11-14

---

## 0. Prime Directive

**Read first, change second, test third, delete last.**

Every change must satisfy three invariants:
1. **Backward compatibility** — existing scripts and configs must still work.
2. **Reversibility** — every PR must be revertable with `git revert` without cascading breakage.
3. **Isolation** — new code goes in a new module/directory; old code is touched only when the new path is proven.

If any of these three is violated, **stop and reconsider the design.**

---

## 1. Architectural Rules (Non-Negotiable)

### Rule 1.1 — Parallel Path, Not Big Bang
When refactoring a module, create a **new sibling** (`config_v2/`, `suites_v2/`, `cli_v2/`, `results_v2/`) alongside the old one. Migrate callers one at a time. Delete the old module only after:
- The new module has 100% test coverage.
- Every caller has been migrated.
- A migration guide exists in `docs/migrations/`.

**Forbidden:** Rewriting `cli.py`, `config.py`, `saver.py`, `schema.py` in place.

### Rule 1.2 — One Source of Truth per Concern
Each concern must have exactly one owner:

| Concern | Owner (current) | Owner (target) |
|---|---|---|
| Config schema | `config.py` | `config_v2/schema.py` |
| Result record | `results/schema.py` | `results_v2/record.py` |
| Storage | `saver.py`, `storage/` | `results_v2/ledger.py` |
| Execution | `executor.py` | `executor.py` (frozen) |
| CLI entrypoint | `cli.py` | `cli_v2/main.py` |
| Registry | `registry.py` | `registry.py` (lazy only) |

**Forbidden:** Two modules writing to the same output file (e.g., `results.jsonl`) at the same time. If you need a new writer, replace the old one atomically.

### Rule 1.3 — No Import-Time Side Effects
Module-level code must never:
- Call `populate_*()`, `discover_*()`, `load_*()`, `register_*()`.
- Touch the filesystem, CUDA, or network.
- Raise exceptions that prevent `import`.

**Required pattern:**
```python
# registry.py
_REGISTRY: dict[str, Any] | None = None

def get_registry() -> dict[str, Any]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY
```

**Forbidden:** `BenchmarkRegistry.populate_canonical_benchmarks()` at module scope.

### Rule 1.4 — No main() in Library Files
Files under `neuralops/benchmarks/` must not contain `def main()` or `if __name__ == "__main__":`. All entrypoints live in `neuralops/cli_v2/commands/` and are registered via `pyproject.toml`.

Migration path: Wrap legacy scripts as `neuralops benchmark --config <legacy-shim>.yaml` instead of adding more `main()`.

### Rule 1.5 — Single CLI Entrypoint
There must be exactly one top-level CLI: `neuralops`. Subcommands go through `cli_v2/main.py`.

- **Forbidden:** New `argparse.ArgumentParser` outside `cli_v2/commands/`.
- **Forbidden:** `if __name__ == "__main__": sys.exit(main())` in any file except `neuralops/benchmarks/cli_v2/main.py`.

---

## 2. PyTorch / HuggingFace Best Practices (Enforced)

### Rule 2.1 — AMP API
Use `torch.amp.*`, not `torch.cuda.amp.*`.

```python
# ✅ Correct
from torch.amp import autocast, GradScaler
with autocast("cuda", dtype=torch.float16):
    ...
scaler = GradScaler("cuda", enabled=True)

# ❌ Forbidden
from torch.cuda.amp import autocast, GradScaler
```

### Rule 2.2 — CUDA Timing
Always use `torch.cuda.Event` + `synchronize()`. Never use `time.time()` for GPU measurements.

```python
# ✅ Correct
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
fn()
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)

# ❌ Forbidden
t0 = time.time()
fn()
elapsed = time.time() - t0  # for GPU work
```

### Rule 2.3 — L2 Cache Flush
Before every timed measurement on CUDA, flush the L2 cache:

```python
def flush_l2(device: torch.device) -> None:
    try:
        l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
        buffer = torch.empty(l2_bytes // 4, dtype=torch.int32, device=device)
        buffer.zero_()
        torch.cuda.synchronize(device)
    except Exception:
        pass
```
Do not hardcode 256 MB. Use `L2_cache_size` from device properties.

### Rule 2.4 — Memory Reporting
Report both allocated and reserved to detect fragmentation:

```python
allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
```
Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in all benchmark entrypoints.

### Rule 2.5 — Device Fallback
Never fail hard on non-CUDA hardware unless the benchmark is explicitly CUDA-only.

```python
# ✅ Correct
if requires_cuda and not torch.cuda.is_available():
    return BenchmarkResult(status=BenchmarkStatus.SKIPPED, reason="CUDA unavailable")

# ❌ Forbidden
raise RuntimeError("GPU-only benchmark")  # without a status-based path
```

### Rule 2.6 — Subprocess Isolation
Every GPU kernel benchmark must run in a spawn subprocess. Handle crashes:

```python
try:
    msg_type, payload = parent_conn.recv()
except (EOFError, ConnectionResetError):
    return BenchmarkResult(
        status=BenchmarkStatus.HARDWARE_ERROR,
        error="Worker terminated abruptly (OOM/segfault/SIGKILL).",
    )
```
Never let a worker crash take down the orchestrator.

### Rule 2.7 — Seeding
For reproducible benchmarks, seed all RNGs:

```python
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```
Record the seed in the result record. Never claim reproducibility without it.

---

## 3. Pydantic v2 Rules (When Writing config_v2/)

### Rule 3.1 — Model Definition
```python
from pydantic import BaseModel, Field, ConfigDict

class KernelSuiteConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )
    kernels: tuple[str, ...] = Field(default_factory=tuple)
    batch_size: int = Field(default=4, ge=1, le=4096)
```

### Rule 3.2 — Forbidden
- ❌ `= []` or `= {}` as defaults → use `Field(default_factory=...)`
- ❌ `list[...]` → use `tuple[...]` for frozen models
- ❌ Mutating a `frozen=True` model → use `.model_copy(update={...})`
- ❌ `extra="allow"` anywhere except explicitly documented extension points
- ❌ `Any` type hints → use concrete types

### Rule 3.3 — YAML Loading
All YAML loads go through `config_v2/loader.py`. Never call `yaml.safe_load()` directly.

### Rule 3.4 — Error Messages
Validation errors must include:
- The exact YAML path (e.g., `suites[2].kernels[0]`)
- The offending value
- The expected type/range
- A Rich rendered output (when terminal) or a structured JSON error (when `--json`)

---

## 4. Result Contract Rules

### Rule 4.1 — One Canonical Record
Every benchmark, kernel, suite, or sweep must produce a `CanonicalRecord`:

```python
@dataclass(frozen=True)
class CanonicalRecord:
    run_id: str
    timestamp: str  # ISO 8601 UTC
    git_commit: str
    git_dirty: bool
    suite: str
    module: str
    model: str
    dataset: str
    device: str
    status: str  # from BenchmarkStatus enum only
    metrics: dict[str, float]
    config: dict[str, Any]
    telemetry: dict[str, Any]
    checksum: str
```
**Forbidden:** Ad-hoc `dict(...)` results, per-module custom schemas, CSV writes without going through `BenchmarkLedger`.

### Rule 4.2 — Status Enum
Use `BenchmarkStatus` only. Never use raw strings `"PASS"`, `"SUCCESS"`, `"FAILED"`.

If a new status is needed, add it to the enum. Never `.normalize()` silently — log a warning.

### Rule 4.3 — Storage Contract
All writes go through `results_v2/ledger.py`:

```python
ledger.write(record: CanonicalRecord) -> None
```
Under the hood, it writes atomically to Parquet + JSONL + SQLite. No other module may write to `results/`.

### Rule 4.4 — Checksums
Every record must have a deterministic SHA-256 checksum over `(run_id, git_commit, config_hash, metrics_hash)`. Duplicates are detected via checksum, not via filename.

---

## 5. Testing Rules

### Rule 5.1 — Test Pyramid
```text
tests/
├── unit/          # < 100ms each, no GPU, no network
├── contract/      # schema/contract validation, no GPU
├── integration/   # synthetic models, no real checkpoints
└── hardware/      # requires physical GPU, marked @pytest.mark.gpu
```

### Rule 5.2 — Coverage Requirements
- New module: ≥ 80% line coverage.
- Any file under `config_v2/`, `results_v2/`: ≥ 90%.
- Bug fix: must include a regression test that would have caught the bug.

### Rule 5.3 — Required Test Categories per Change
| Change type | Required tests |
|---|---|
| New config field | Round-trip serialize + validation error case |
| New subprocess path | Simulated SIGKILL/SIGTERM test |
| New CLI subcommand | CliRunner snapshot + `--json` output test |
| New benchmark suite | `@pytest.mark.gpu` + `@pytest.mark.cpu` variants |
| Bug fix | Regression test that fails on main before the fix |

### Rule 5.4 — Forbidden in Tests
- ❌ `time.sleep()` for synchronization → use events/barriers
- ❌ Hardcoded paths → use `tmp_path`
- ❌ Real model downloads → use tiny synthetic AutoModel (e.g., `hf-internal-testing/tiny-random-bert`)
- ❌ `@pytest.mark.skip` without a linked issue number

---

## 6. Code Quality Rules

### Rule 6.1 — Linting
Every PR must pass:

```bash
ruff check neuralops/ tests/
ruff format --check neuralops/ tests/
mypy --strict neuralops/benchmarks/config_v2/ neuralops/benchmarks/suites_v2/ neuralops/benchmarks/cli_v2/ neuralops/benchmarks/results_v2/
```
*Scope of `--strict` mypy:* only new modules. Do not attempt strict typing on legacy files — it will take weeks and block everything.

### Rule 6.2 — Import Order
```python
# 1. stdlib
# 2. third-party (torch, transformers, pydantic)
# 3. neuralops (absolute imports only)
```
**Forbidden:** relative imports outside the same subpackage.

### Rule 6.3 — File Size
- Soft limit: 500 lines per file.
- Hard limit: 800 lines per file.
- If a file exceeds 800 lines, it must be split before merge.
- *Exception:* Data tables, constants, or auto-generated content — mark with `# noqa: file-size` at top.

### Rule 6.4 — Function Size
- Soft limit: 50 lines per function.
- Hard limit: 100 lines.
- If a function exceeds 100 lines, extract helpers.

### Rule 6.5 — Type Hints
Every public function must have full type hints. Private helpers (`_name`) may omit hints only if they are trivial (< 5 lines).

**Forbidden:** `Any` unless justified with a `# reason: ...` comment.

### Rule 6.6 — Docstrings
Public classes and functions need Google-style docstrings with:
- One-line summary
- Args: section (if args)
- Returns: section
- Raises: section (if raises)
- Example (optional but preferred)

---

## 7. Commit & PR Rules

### Rule 7.1 — Commit Message Format
```text
<scope>: <imperative summary, ≤72 chars>

<optional body explaining why, not what>

Refs: #<issue>
Tests: <test files added/modified>
```
*Scopes:* config, executor, registry, cli, suite, results, docs, tests, ci.

### Rule 7.2 — PR Size
- Soft limit: 400 lines changed.
- Hard limit: 800 lines changed.
- If larger, split into stacked PRs. Each PR must be independently mergeable.

### Rule 7.3 — PR Checklist
Every PR description must include:
- [ ] AGENTS.md read and followed
- [ ] Backward compatible (or migration guide linked)
- [ ] Tests added/updated
- [ ] ruff + mypy pass
- [ ] Docs updated (if public API changed)
- [ ] No new `main()` added
- [ ] No new `argparse.ArgumentParser` outside `cli_v2/`
- [ ] No `torch.cuda.amp.*` imports
- [ ] No hardcoded `cuda:0` (use config or device param)
- [ ] No silent `except Exception: pass` (log or re-raise)

### Rule 7.4 — Forbidden Commit Patterns
- ❌ `git commit -m "fix"` — always descriptive
- ❌ Committing `.pt`, `.safetensors`, `.parquet`, or `results/` artifacts
- ❌ Committing generated docs without the source
- ❌ `--force` push to main or master

---

## 8. Migration Rules

### Rule 8.1 — New Feature Goes in _v2/
If you're adding a new capability (new config, new suite, new CLI command), it goes in `config_v2/`, `suites_v2/`, `cli_v2/`, or `results_v2/`.

*Exception:* If it's a bug fix to legacy code, patch in place with a regression test.

### Rule 8.2 — Deprecation Path
When migrating a legacy API:
1. Add the new API in `_v2/`.
2. Add a `DeprecationWarning` to the old API pointing to the new one.
3. Update all internal callers.
4. Wait one minor version (e.g., v2.1 → v2.2).
5. Delete the old API.

**Forbidden:** Removing a public API without a full deprecation cycle.

### Rule 8.3 — Legacy Script Deletion
A script under `legacy_root/` or `legacy_scripts/` may only be deleted after:
1. Its functionality is covered by a `suites_v2` test.
2. A `scripts/migrate_legacy_<name>.py` migration exists.
3. A `CHANGELOG` entry documents the removal.

---

## 9. Performance & Correctness Rules

### Rule 9.1 — Warmup Before Timing
Every benchmark must run at least 10 warmup iterations before timing. Triton autotune and TorchInductor compile on the first call.

**Forbidden:** Timing the first call.

### Rule 9.2 — Outlier Handling
Report median and p90 and p99. Do not report mean alone. Discard top/bottom 5% before computing statistics.

**Forbidden:** Reporting a single-number mean without distribution.

### Rule 9.3 — Numerical Parity
Every Triton/CUDA kernel benchmark must include:
- Max absolute difference vs PyTorch reference
- Cosine similarity vs PyTorch reference
- NaN/Inf check

**Forbidden:** Reporting speedup without a parity check.

### Rule 9.4 — Fail Closed
If a benchmark fails parity, OOMs, or times out, return a `BenchmarkStatus`-based record. Never silently skip or silently return zero.

**Forbidden:** `except Exception: return 0.0`

---

## 10. Anti-Patterns (Hard Forbidden)

The following patterns will be rejected at review:
- ❌ `def main()` in `neuralops/benchmarks/` (except `cli_v2/main.py`)
- ❌ `argparse.ArgumentParser` outside `cli_v2/`
- ❌ `torch.cuda.amp.*` (use `torch.amp.*`)
- ❌ `time.time()` for GPU timing (use `torch.cuda.Event`)
- ❌ Hardcoded `cuda:0` (use `device` parameter)
- ❌ Silent `except Exception: pass` (log or re-raise)
- ❌ Import-time side effects (registry, config, model loading)
- ❌ Multiple writers to `results/` (only `results_v2/ledger.py`)
- ❌ New Config dataclass outside `config_v2/`
- ❌ Raw string status (use `BenchmarkStatus`)
- ❌ `yaml.safe_load()` outside `config_v2/loader.py`
- ❌ `os.environ["CUDA_VISIBLE_DEVICES"] = ""` outside `cli_v2/`
- ❌ Deleting files without a migration test
- ❌ PR > 800 lines (split it)
- ❌ Committing artifacts (`.pt`, `.parquet`, `results/`, `docs/_build/`)

---

## 11. When You're Unsure

If a task violates any of the above:
1. Stop.
2. Document the conflict in the PR description or an issue.
3. Propose the smallest possible compliant alternative.
4. Ask for human review before proceeding.

Never proceed with a change that violates these rules "just this once." There is no "just this once."

---

## 12. Reference Reading Order

Before your first PR, read in this order:
1. This file (`AGENTS.md`)
2. `docs/architecture/overview.md`
3. `docs/migrations/README.md`
4. The module you're changing, top to bottom
5. Its tests in `tests/`
6. PyTorch benchmark conventions: https://pytorch.org/docs/stable/benchmark.html
7. HuggingFace Trainer design notes: https://huggingface.co/docs/transformers/main_classes/trainer

---

## Appendix A — Quick Reference: Correct vs Forbidden

```python
# ─────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────
# ❌ Forbidden
t0 = time.time()
kernel(x)
ms = (time.time() - t0) * 1000

# ✅ Correct
flush_l2(device)
start, end = torch.cuda.Event(True), torch.cuda.Event(True)
start.record(); kernel(x); end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)

# ─────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────
# ❌ Forbidden
device = torch.device("cuda:0")

# ✅ Correct
device = resolve_device(config.device)  # or passed in

# ─────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────
# ❌ Forbidden
return {"status": "PASS", "latency": 1.2}

# ✅ Correct
return CanonicalRecord(
    status=BenchmarkStatus.SUCCESS,
    metrics={"latency_ms": 1.2},
    ...
)

# ─────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────
# ❌ Forbidden
class MyConfig:
    kernels = []  # mutable default

# ✅ Correct
class MyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kernels: tuple[str, ...] = Field(default_factory=tuple)

# ─────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────
# ❌ Forbidden
populate_registry()  # at module scope

# ✅ Correct
def get_registry():
    if _REGISTRY is None:
        _REGISTRY = _build()
    return _REGISTRY

# ─────────────────────────────────────────────────────
# Subprocess
# ─────────────────────────────────────────────────────
# ❌ Forbidden
result = parent_conn.recv()  # may raise EOFError

# ✅ Correct
try:
    result = parent_conn.recv()
except (EOFError, ConnectionResetError):
    return failed_record("Worker crashed")
```
