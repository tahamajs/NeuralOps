# Breaking Changes & Migration Guide — NeuralOps v2.4.0

This document tracks all backward-incompatible modifications, API renames, behavioral shifts, and configuration migrations introduced in NeuralOps v2.4.0.

---

## 1. Executive Summary & Migration Principles

Version 2.4.0 hardens the NeuralOps framework into a production-grade deep learning systems library. In accordance with the project's **Zero-Fake Rule** and **Iron Laws of Architecture**, several legacy behaviors that masked runtime errors, faked metric outputs, or relied on non-standard interfaces have been remediated.

Every breaking modification documented below includes:
1. **Identifier & Classification:** Tracking catalog ID and severity rating.
2. **Architectural Justification:** The technical rationale for the modification.
3. **Before & After Migration Snippets:** Copy-paste code examples demonstrating exact migration steps.

---

## 2. Core Breaking Changes Catalog

### BC-01: Renamed API `continuous_batching_generate` → `generate_sequential`

- **Identifier:** `BC-01`
- **Component:** `neuralops.serving`
- **Severity:** High
- **Type:** API Rename & Semantic Honesty
- **Rationale:** The legacy implementation in `neuralops.serving` was labeled `continuous_batching_generate`, yet internally it executed requests sequentially through a single worker without true continuous iteration-level scheduling. Renaming the method to `generate_sequential` aligns API nomenclature with algorithmic reality and eliminates false production expectations.

#### Migration Code Snippet:
```python
# =====================================================================
# Before (v2.3) — Deprecated / Removed
# =====================================================================
from neuralops.serving.adapters import ServingAdapter

serving_adapter = ServingAdapter(model=model, tokenizer=tokenizer)
# continuous_batching_generate claimed continuous batching but ran sequentially
output = serving_adapter.continuous_batching_generate(prompts, max_new_tokens=256)

# =====================================================================
# After (v2.4) — Canonical
# =====================================================================
from neuralops.serving.adapters import ServingAdapter

serving_adapter = ServingAdapter(model=model, tokenizer=tokenizer)
# Explicit sequential decoding accurately reflects the single-worker execution path
output = serving_adapter.generate_sequential(prompts, max_new_tokens=256)
```

---

### BC-02: Behavioral Invariant: `certify_model()` Evaluates Real Booleans

- **Identifier:** `BC-02`
- **Component:** `neuralops.surgery.director`
- **Severity:** Medium
- **Type:** Behavioral Shift (Zero-Fake Policy)
- **Rationale:** In previous releases, `UniversalSurgeryDirector.certify_model()` returned hardcoded `True` values for 8 out of the 10 Universal Surgery Invariants regardless of whether the model satisfied mathematical and structural preconditions. In v2.4.0, all 10 invariants are strictly checked against actual model tensors, parameter gradients, and layer signatures. Models with uninitialized projections or mismatched rank factors will now return `False` for failing invariants.

#### Migration Code Snippet:
```python
# =====================================================================
# Before (v2.3) — Unchecked Assumption
# =====================================================================
from neuralops.surgery.director import UniversalSurgeryDirector

director = UniversalSurgeryDirector()
# Previously returned hardcoded {"invariant_1": True, "invariant_2": True, ...}
result = director.certify_model(model)
assert all(result.values()), "Assumed model always passed certification"

# =====================================================================
# After (v2.4) — Robust Verification Handling
# =====================================================================
from neuralops.surgery.director import UniversalSurgeryDirector

director = UniversalSurgeryDirector()
# Now performs authentic tensor checks across all 10 mathematical invariants
result = director.certify_model(model)

if not all(result.values()):
    unverified = [k for k, v in result.items() if not v]
    print(f"Certification failed for invariants: {unverified}")
    # Inspect failing invariants and re-run surgery calibration if necessary
    raise RuntimeError(f"Model failed surgery certification: {unverified}")
```

---

### BC-03: Strict Validation: `CUDAGraphDecodeRunner` Rejects Dynamic Caches

- **Identifier:** `BC-03`
- **Component:** `neuralops.models.cuda_graph`
- **Severity:** High
- **Type:** Validation Enforced (Memory Safety)
- **Rationale:** CUDA Graph capture and replay require perfectly static tensor addresses and buffer shapes. In v2.3, passing a dynamic KV cache (`DynamicCache`) silently succeeded during capture but resulted in silent memory corruption or CUDA memory access violations during execution. `CUDAGraphDecodeRunner` now strictly checks the cache instance and raises `ValueError` if a static cache is not provided.

#### Migration Code Snippet:
```python
# =====================================================================
# Before (v2.3) — Silent Failure / Corruption Risk
# =====================================================================
from neuralops.models.cuda_graph import CUDAGraphDecodeRunner

# Implicitly initialized dynamic KV cache inside decode loop
runner = CUDAGraphDecodeRunner(model=model)

# =====================================================================
# After (v2.4) — Explicit StaticCache Allocation
# =====================================================================
import torch
from transformers import StaticCache
from neuralops.models.cuda_graph import CUDAGraphDecodeRunner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Allocate static KV cache with explicit batch size and sequence boundaries
cache = StaticCache(
    config=model.config,
    max_batch_size=1,
    max_cache_len=512,
    device=device,
    dtype=model.dtype,
)

runner = CUDAGraphDecodeRunner(model=model, static_cache=cache)
```

---

### BC-04: Configuration Schema: YAML Loading Migrated to Pydantic v2

- **Identifier:** `BC-04`
- **Component:** `neuralops.benchmarks.config_v2`
- **Severity:** Medium
- **Type:** API Migration & Type Validation
- **Rationale:** Direct use of `yaml.safe_load()` produced unvalidated dictionary trees where misspelled configuration keys (such as `batch_szie` instead of `batch_size`) silently passed into execution pipelines with arbitrary fallback defaults. In v2.4.0, all benchmark configurations must be parsed via `neuralops.benchmarks.config_v2.loader`, enforcing frozen Pydantic v2 models with `extra="forbid"`.

#### Migration Code Snippet:
```python
# =====================================================================
# Before (v2.3) — Unvalidated Dict Loading
# =====================================================================
import yaml

with open("benchmark.yaml", "r", encoding="utf-8") as f:
    raw_config = yaml.safe_load(f)

# Unvalidated dictionary access with silent key error vulnerabilities
batch_size = raw_config.get("batch_size", 4)

# =====================================================================
# After (v2.4) — Strongly-Typed Pydantic v2 Loader
# =====================================================================
from neuralops.benchmarks.config_v2.loader import load_benchmark_config

# Strictly validated, frozen Pydantic v2 configuration model
config = load_benchmark_config("benchmark.yaml")

# Typed attribute access with immutability guarantees
batch_size = config.batch_size
```

---

## 3. CLI Entrypoint & Benchmarks Architecture

### Deprecation of `benchmarks/cli.py` in Favor of `cli_v2`
- **Old Path:** `neuralops.benchmarks.cli:main`
- **New Path:** `neuralops.benchmarks.cli_v2.main:main`
- **Console Scripts:** `neuralops`, `nops`, `neuralops-cli`, and `nops-cli` point to `cli_v2.main:main`.
- **Migration Details:**
  - Invoking `neuralops.benchmarks.cli.main()` emits a `DeprecationWarning` and delegates execution to `cli_v2.main:main`.
  - Update custom automation scripts or wrappers to invoke `neuralops.benchmarks.cli_v2.main:main` directly.
  - The legacy `benchmarks/cli.py` shim is scheduled for removal in `v3.0.0`.

### Quarantined Legacy Benchmark Scripts
- **Action:** 58 legacy standalone benchmark scripts originally located in `neuralops/benchmarks/legacy_root/` and `legacy_scripts/` have been relocated to `neuralops/benchmarks/_legacy/`.
- **Migration Details:** Execute benchmarks through the unified CLI entrypoint:
  ```bash
  neuralops benchmark --config configs/attention.yaml
  ```
- Direct execution of quarantined scripts emits a `DeprecationWarning` indicating migration paths.

---

## 4. Subpackage & Namespace Consolidation

The following namespace migrations decouple internal implementation primitives from clean public APIs:

| Legacy Subpackage | Canonical Replacement (v2.4.0) | Status | Removal Target |
| :--- | :--- | :--- | :--- |
| `neuralops.attentions` | `neuralops.layers` | Deprecated compatibility shim | v3.0.0 |
| `neuralops.compression_engine` | `neuralops.compression` | Deprecated compatibility shim | v3.0.0 |
| `neuralops.core.compression_engine` | `neuralops.compression` | Deprecated compatibility shim | v3.0.0 |
| `neuralops.core.roofline` | `neuralops.core.roofline` (refactored) | Modernized | v3.0.0 |
| `neuralops.training` | `neuralops.trainer` | Deprecated compatibility shim | v3.0.0 |
| `neuralops.core.rope` | `neuralops.core.utils.rope` | Deprecated compatibility shim | v3.0.0 |
| `neuralops.core.gqa_utils` | `neuralops.utils.architecture` | Deprecated compatibility shim | v3.0.0 |
| `neuralops.surgery.universal_calibrator` | `neuralops.trainer.regimes` | Deprecated compatibility shim | v3.0.0 |

---

## 5. PyTorch API & Execution Contracts

### AMP Modernization (Rule 2.1)
- **Forbidden:** `from torch.cuda.amp import autocast, GradScaler`
- **Required:**
  ```python
  from torch.amp import autocast, GradScaler

  with autocast("cuda", dtype=torch.float16):
      outputs = model(inputs)
  scaler = GradScaler("cuda", enabled=True)
  ```
- **Rationale:** PyTorch 2.x deprecates device-specific `torch.cuda.amp` in favor of universal `torch.amp`.

### Safe Deserialization Mandate (Security)
- **Forbidden:** Invoking `torch.load(..., weights_only=False)` or omitting `weights_only`.
- **Required:** All tensor and checkpoint deserialization must explicitly specify `weights_only=True`.

### Hardware Timing & Fail-Closed Status Enums
- **Forbidden:** `time.time()` for GPU duration measurements; raw dictionary return structures `{"status": "PASS"}`.
- **Required:**
  - High-precision CUDA event measurement (`torch.cuda.Event(enable_timing=True)`) with prior L2 cache flushing.
  - Return shapes must conform to `CanonicalRecord` with typed `BenchmarkStatus` enum variants (`SUCCESS`, `FAILED`, `FAIL_NUMERICS`, `SKIPPED`, `HARDWARE_ERROR`).

---

## 6. Test Suite Structure

Test suites have been reorganized into 4 standardized execution tiers:
- `tests/unit/`: Rapid tests (<100ms each, CPU execution only, no network calls).
- `tests/contract/`: Schema, registry integrity, and API contract conformance checks.
- `tests/integration/`: End-to-end multi-stage pipeline workflows on synthetic models.
- `tests/hardware/`: Hardware-dependent GPU kernel benchmarks (`@pytest.mark.gpu`).

Pytest executes with `--import-mode=importlib` to prevent module namespace collisions.
