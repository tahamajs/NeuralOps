# Remaining Code Architecture & Boundary Analysis (Phase 5)

**Date**: 2026-10-08  
**Scope**: All source and test modules outside `core/`, `benchmarks/`, `attentions/`, and `compression_engine/`.  
**Total Target Scope**: 357 files across all remaining packages.

---

## 1. Module Boundary Map & Dependency Flow

The package dependency matrix reveals the following cross-module import paths:
- **`neuralops/utils`** (Layer 1): Imported widely by almost every layer (attentions, compression, tests, trainer).
- **`neuralops/core`** (Layer 0): Imported by 113+ modules (base classes, adapters, registries).
- **`neuralops/data`** (Layer 2): Imported by `trainer`, `evaluation`, and `experiment_utils`.
- **`neuralops/config`** (Layer 3): Imported by `trainer`, `data`, and `experiment_utils`.
- **`neuralops/trainer`** (Layer 6): Imported by `evaluation` and `experiment_utils`.
- **`neuralops/evaluation`** (Layer 7): Imported by `benchmarks` and test suites.
- **`neuralops/runners`**: Orchestration scripts that invoke `benchmarks`, `evaluation`, and `trainer`.
- **`neuralops/experiment_utils`**: High fan-out runner module that imports across `analysis`, `config`, `core`, `data`, `experiment_logging`, `results_saver`, `trainer`, and `utils`.

---

## 2. Layer Violations (Bottom-Up Imports)

Expected architecture layers:
```text
Layer 0: neuralops/core/               (base classes, registry)
Layer 1: neuralops/utils/              (helpers, no upward deps)
Layer 2: neuralops/data/               (data loaders)
Layer 3: neuralops/config/             (configs)
Layer 4: neuralops/attentions/         (uses core)
Layer 5: neuralops/compression_engine/ (uses core)
Layer 6: neuralops/trainer/            (uses core, data, config)
Layer 7: neuralops/evaluation/         (uses everything below)
Layer 8: neuralops/benchmarks/         (uses everything)
Layer 9: neuralops/analysis/           (reads results)
Layer 10: scripts/                     (user-facing)
```

### Detected Violations:
| File:line | Source Layer | Imports From | Violation Description | Severity |
| :--- | :--- | :--- | :--- | :--- |
| [`neuralops/core/_attention_injectors.py:26`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/_attention_injectors.py#L26) | L0 `core` | L1 `utils` | Core imports from utility layer (`neuralops.utils.model_family`) | HIGH |
| [`neuralops/core/multi_mode_benchmark.py:17`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/multi_mode_benchmark.py#L17) | L0 `core` | L8 `benchmarks` | Core module imports directly from benchmarks layer | HIGH |
| [`neuralops/core/registry.py:211`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/registry.py#L211) | L0 `core` | L4 `attentions` | Core registry imports concrete attention modules eagerly | HIGH |
| [`neuralops/core/registry.py:193`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/registry.py#L193) | L0 `core` | L5 `compression_engine`| Core registry imports concrete compressors eagerly | HIGH |
| [`neuralops/attentions/svd/residual_svd_attention.py:18`](file:///Users/tahamajs/Documents/uni/Project/neuralops/attentions/svd/residual_svd_attention.py#L18) | L4 `attentions` | L5 `compression_engine` | Attention layer imports from compression engine | MEDIUM |
| [`neuralops/trainer/train_lra.py:30`](file:///Users/tahamajs/Documents/uni/Project/neuralops/trainer/train_lra.py#L30) | L6 `trainer` | L7 `evaluation` | Trainer imports from evaluation module | MEDIUM |
| [`neuralops/evaluation/run_baseline_comparison.py:43`](file:///Users/tahamajs/Documents/uni/Project/neuralops/evaluation/run_baseline_comparison.py#L43) | L7 `evaluation` | L8 `benchmarks` | Evaluation imports benchmarks | LOW (CLI script) |

---

## 3. Public API Analysis (`__init__.py`)

| Package | Public Exports (`__all__`) | External Usage | Status / Dead Exports |
| :--- | :---: | :---: | :--- |
| `neuralops` (root) | 0 | Root re-exports dynamic alias loaders | Clean |
| `neuralops/core` | 0 | Submodules imported directly (`core.registry`, etc.) | Direct import design |
| `neuralops/config` | 19 exports | Imported in tests and trainers | Clean (monolithic duplication resolved in P3) |
| `neuralops/data` | 26 exports | 8 used in trainers; 18 internal to data suite | Partially dead exports |
| `neuralops/trainer` | 32 exports | 12 used in scripts/runners; 20 only tested | High export surface |
| `neuralops/evaluation` | 14 exports | 4 used in runners; 10 tested internally | High export surface |
| `neuralops/analysis` | 32 exports | 6 used in reporting scripts; 26 unused externally | Dead API surface in `__all__` |
| `neuralops/utils` | 14 exports | Re-exports submodules (`helpers`, `paths`, etc.) | Stable |
| `neuralops/results_saver` | 2 exports | `ResultsSaver`, `SaveConfig` | Actively used |
| `neuralops/tracking` | 7 exports | `ExperimentTracker`, callbacks | Actively used |

---

## 4. Cohesion and Coupling (Fan-In, Fan-Out, Stability)

### Top 10 Most Stable Modules ($I = 0.00$, high fan-in, fan-out = 0)
1. `neuralops.core.adapters` ($I=0.00$, fan-in=113, fan-out=0)
2. `neuralops.core.gqa_utils` ($I=0.00$, fan-in=82, fan-out=0)
3. `neuralops.core.model_utils` ($I=0.00$, fan-in=64, fan-out=0)
4. `neuralops.core.attention_mask` ($I=0.00$, fan-in=55, fan-out=0)
5. `neuralops.core.pruning_utils` ($I=0.00$, fan-in=39, fan-out=0)
6. `neuralops.utils.model_family` ($I=0.00$, fan-in=22, fan-out=0)
7. `neuralops.benchmarks.result` ($I=0.00$, fan-in=12, fan-out=0)
8. `neuralops.utils.lora` ($I=0.00$, fan-in=8, fan-out=0)
9. `neuralops.data.glue_loader` ($I=0.00$, fan-in=6, fan-out=0)
10. `neuralops.utils.paths` ($I=0.00$, fan-in=6, fan-out=0)

### Top 10 Most Unstable Modules ($I = 1.00$, fan-in = 0, high fan-out)
1. `neuralops.tests.test_track2_spectral_ssm_audit` ($I=1.00$, fan-out=26)
2. `neuralops.tests.test_models_and_datasets_matrix` ($I=1.00$, fan-out=13)
3. `neuralops.experiment_utils.runner` ($I=1.00$, fan-out=11)
4. `neuralops.tests.test_svd_audit_suite` ($I=1.00$, fan-out=11)
5. `neuralops.experiment_utils.main` ($I=1.00$, fan-out=11)
6. `neuralops.main` ($I=1.00$, fan-out=10)
7. `neuralops.runners.unified_orchestrator` ($I=1.00$, fan-out=8)
8. `neuralops.evaluation.eval_all_models_datasets` ($I=0.91$, fan-in=1, fan-out=10)

---

## 5. Package-Level Overview

| Package | Files | Total Lines | Avg Lines | Single Purpose Clarity |
| :--- | :---: | :---: | :---: | :--- |
| `neuralops/core` | 48 | 11,008 | 229 | **High**: Central base classes, Triton kernels, registries |
| `neuralops/utils` | 20 | 3,785 | 189 | **High**: System utilities, device helpers, logger, paths |
| `neuralops/data` | 18 | 3,522 | 195 | **High**: Datasets, tokenizers, data collators, GLUE/LRA |
| `neuralops/config` | 10 | 684 | 68 | **High**: Dataclass configuration schemas |
| `neuralops/trainer` | 27 | 4,451 | 164 | **Medium**: Training loops, optimizers, callbacks, LRA |
| `neuralops/evaluation` | 12 | 4,796 | 399 | **Medium**: Model evaluation, harnesses, A/B testing |
| `neuralops/analysis` | 11 | 3,902 | 354 | **High**: Visualization, metrics computation, statistics |
| `neuralops/results_saver`| 7 | 1,282 | 183 | **High**: Serialization, CSV/JSON export |
| `neuralops/tracking` | 4 | 451 | 112 | **High**: MLflow/WandB callbacks |
| `neuralops/experiment_utils`| 10 | 1,747 | 174 | **Low**: High overlap with runners, duplicate files |
| `neuralops/runners` | 7 | 2,347 | 335 | **Medium**: Orchestrator scripts & monitor server |
| `neuralops/tests` | 211 | 28,163 | 133 | **High**: Automated test suites |

---

## 6. Naming Consistency Violations

- **Total violations found**: 110.
- **Key categories**:
  - Internal private classes starting with underscore + PascalCase (e.g., `_AliasLoader`, `_TritonLowRankAutogradFunction`, `_LinearizedGELU`) — standard Python convention for private classes.
  - Suffix casing for acronyms (e.g. `AATC_XKVCache` instead of `AatcXkvCache`).
  - Mathematical helper functions preserving uppercase matrix symbols (e.g., `_dequantize_W`, `_collect_gating_D`, `_factorize_with_JG`, `_V`).

---

## 7. Hidden Dependencies & Risky Imports

1. **Star imports (`from X import *`)**: 22 occurrences (19 in shims/benchmarks delegating to `_legacy/`, 1 in `neuralops/core/compression_engine.py`, 1 in `legacy_ensemble.py`, 1 in `evaluation/metrics.py`).
2. **`sys.path` modifications**: 66 occurrences (predominantly in `neuralops/tests/` and CLI scripts ensuring standalone executability).
3. **Dynamic `importlib.import_module`**: 203 occurrences (used for dynamic lazy loading in `neuralops.core.registry`, `neuralops/__init__.py`, and compressor discovery).

---

## 8. Prioritized Fix Plan

### Priority 1 (CRITICAL)
- **Break upward imports from Core (`core/_attention_injectors.py` → `utils.model_family`)**: Move model family detection into `core/model_utils.py`.
- **Remove upward import in `core/multi_mode_benchmark.py`**: Clean up legacy shim.
- **De-duplicate and clarify `neuralops/experiment_utils`**: Consolidate with `runners/`.

### Priority 2 (IMPORTANT)
- **Clean Star Imports**: Replace star imports in non-shim modules (`evaluation/metrics.py`, `core/compression_engine.py`).
- **Prune dead exports in `neuralops/analysis/__init__.py`**: Keep only the 6 actually consumed functions in `__all__`.

### Priority 3 (NICE TO HAVE)
- **Refactor `sys.path` manipulation in test files**: Replace hardcoded path inserts with standard `pytest` package discovery.
