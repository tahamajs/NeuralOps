# Breaking Changes & Migration Guide — NeuralOps v2.4.0

This document tracks all API modifications, deprecations, renames, and removals introduced in NeuralOps v2.4.0.

---

## 1. CLI Entrypoint & Benchmarks Architecture

### Deprecation of `benchmarks/cli.py` in favor of `cli_v2`
- **Old Path:** `neuralops.benchmarks.cli:main`
- **New Path:** `neuralops.benchmarks.cli_v2.main:main`
- **Impact:** Console script entry points (`neuralops`, `nops`, `neuralops-cli`, `nops-cli`) now execute `benchmarks.cli_v2.main:main`.
- **Migration:**
  - Invoking `neuralops.benchmarks.cli.main()` emits a `DeprecationWarning` and delegates execution to `cli_v2.main:main`.
  - Update any custom invocation wrappers to point directly to `neuralops.benchmarks.cli_v2.main:main`.
  - The legacy `benchmarks/cli.py` module will be permanently removed in `v3.0.0`.

### Quarantined Legacy Benchmark Scripts
- **Action:** 58 historical standalone benchmark scripts originally in `neuralops/benchmarks/legacy_root/` and `legacy_scripts/` have been relocated to `neuralops/benchmarks/_legacy/`.
- **Migration:** All benchmarking operations should be invoked through the unified CLI:
  ```bash
  neuralops benchmark --config configs/attention.yaml
  ```
- Direct execution of quarantined scripts emits a `DeprecationWarning`.

---

## 2. Namespace & Subpackage Migrations

The following subpackage migrations were introduced to decouple internal primitives from external API facades:

| Legacy Subpackage | Canonical Replacement (v2.4.0) | Status | Removal Target |
| :--- | :--- | :--- | :--- |
| `neuralops.attentions` | `neuralops.layers` | Deprecated shim | v3.0.0 |
| `neuralops.compression_engine` | `neuralops.compression` | Deprecated shim | v3.0.0 |
| `neuralops.core.compression_engine` | `neuralops.compression` | Deprecated shim | v3.0.0 |
| `neuralops.core.roofline` | `neuralops.core.roofline` (refactored) | Modernized | v3.0.0 |
| `neuralops.training` | `neuralops.trainer` | Deprecated shim | v3.0.0 |
| `neuralops.core.rope` | `neuralops.core.utils.rope` | Deprecated shim | v3.0.0 |
| `neuralops.core.gqa_utils` | `neuralops.utils.architecture` | Deprecated shim | v3.0.0 |
| `neuralops.surgery.universal_calibrator` | `neuralops.trainer.regimes` | Deprecated shim | v3.0.0 |

---

## 3. PyTorch & Hardware Contract Best Practices

### AMP API Modernization (Rule 2.1)
- **Forbidden:** `from torch.cuda.amp import autocast, GradScaler`
- **Required:**
  ```python
  from torch.amp import autocast, GradScaler

  with autocast("cuda", dtype=torch.float16):
      ...
  scaler = GradScaler("cuda", enabled=True)
  ```
- **Reason:** PyTorch 2.x deprecates `torch.cuda.amp` in favor of device-agnostic `torch.amp`.

### Safe Deserialization Mandate (CVE Protection)
- **Forbidden:** Calling `torch.load(..., weights_only=False)` or omitting `weights_only`.
- **Required:** All model and tensor loading must pass `weights_only=True`.

### Fail-Closed Execution & Status Enums
- Ad-hoc string dictionary outputs (e.g. `{"status": "PASS"}`) are deprecated.
- All benchmark, kernel, and suite runners produce `CanonicalRecord` structures using the strict `BenchmarkStatus` enum.

---

## 4. Test Suite Structure (Rule 5.1)

Test files have been reorganized into 4 standardized execution tiers:
- `tests/unit/`: Fast tests (<100ms each, CPU only, no network).
- `tests/contract/`: Schema, registry integrity, and contract conformance checks.
- `tests/integration/`: Multi-stage workflows, synthetic model transformations.
- `tests/hardware/`: Hardware-dependent tests requiring physical GPUs (`@pytest.mark.gpu`).

Pytest now executes with `--import-mode=importlib` to prevent namespace collisions across deeply nested modules.
