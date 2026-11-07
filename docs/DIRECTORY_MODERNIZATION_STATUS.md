# Directory Modernization Status

**Authoritative Alignment:** NeuralOps v2.0.6  
**Scope:** Status tracking for architectural package refactoring, legacy shims, and test suites.

---

## 1. Subsystem Migration Ledger

| Directory Path | Package Status | Test Suite Status | Migration Target | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `neuralops/operations/` | **Deprecated** | Active in `tests/operations/` | `neuralops/ops/` & `neuralops/pipeline_transform/` | Scheduled for removal in v2.1.0. |
| `neuralops/validator/` | **Merged** | Active in `tests/validator/` | `neuralops/validation/` | Merged at package level; test consolidation planned under F-68. |
| `neuralops/experiment_logging/`| **Merged** | Migrated | `neuralops/tracking/` | Completed. |
| `neuralops/training/` | **Removed** | Migrated | `neuralops/trainer/` | Legacy MetaPathFinder shim eliminated. |
| `neuralops/executor/` | **Deprecated** | Active in `tests/` | `neuralops/pipeline_transform/` | Converted to thin re-export facade with `DeprecationWarning`. |

---

## 2. Granular Details

### `operations/` vs `ops/`
`neuralops/operations/` historically contained model mutation helpers and AST visitors. High-performance CUDA and Triton kernels have been consolidated into `neuralops/ops/`. The remaining transformation routines in `operations/` are mapped to `pipeline_transform/`. The `operations/` directory is scheduled for final removal in v2.1.0 once all legacy adapter calls are transitioned.

### `validator/` vs `validation/`
`neuralops/validator/` was successfully consolidated into `neuralops/validation/`. All static checks, Symplectic guards, and multi-tier gates now execute under `neuralops.validation`. Existing tests in `tests/validator/test_static_and_gates.py` continue to exercise these contracts until F-68 unifies the `tests/` directory tree.
