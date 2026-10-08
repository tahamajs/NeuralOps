# Remaining Code Duplicate Code Audit (Phase 3)

**Date**: 2026-10-08  
**Scope**: All source and test files outside `core/`, `benchmarks/`, `attentions/`, and `compression_engine/`.  
**Total Candidate Files Evaluated**: 357 files (145 source files, 212 test/script files).

---

## 1. Executive Summary

| Category | Count | Primary Areas Involved |
| :--- | :--- | :--- |
| **Identical File Duplicates (100% match)** | 1 pair (2 files, 353 lines each) | `neuralops/experiment_utils/` (`runner.py` vs `main.py`) |
| **Duplicate Class Names (Source Scope)** | 23 classes | `neuralops/config/`, `experiment_utils/`, `tracking/`, `data/` |
| **Duplicate Function Names (Source Scope)** | 71 function names | `neuralops/utils/`, `trainer/`, `experiment_logging/`, `data/` |
| **Identical Function AST Bodies (Source Scope)** | 17 AST patterns | `runner.py` vs `main.py`, `layer_norm_fix.py`, dataset iterators |

---

## 2. Critical Exact File Duplicate (100% Identical)

- [`neuralops/experiment_utils/runner.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/experiment_utils/runner.py) (353 lines)  
- [`neuralops/experiment_utils/main.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/experiment_utils/main.py) (353 lines)  
- **Diff**: 0 lines difference across all 353 lines.  
- **Action**: Deprecate or alias `neuralops/experiment_utils/runner.py` into `main.py` (or vice versa with a deprecation shim).

---

## 3. Class Duplications in Source Files

### 3.1 Monolithic vs Modular Config Duplications
[`neuralops/config/config.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py) defines 18 configuration classes in a single monolithic file that are redundantly redefined in dedicated modular configuration files:

| Class Name | Monolithic Definition | Modular Definition | Recommendation |
| :--- | :--- | :--- | :--- |
| `LoRAConfig` | [`neuralops/config/config.py:16`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L16) | [`neuralops/config/method_config.py:12`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/method_config.py#L12) | Re-export from `method_config.py` |
| `RDropConfig` | [`neuralops/config/config.py:27`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L27) | [`neuralops/config/method_config.py:31`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/method_config.py#L31) | Re-export from `method_config.py` |
| `R3FConfig` | [`neuralops/config/config.py:36`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L36) | [`neuralops/config/method_config.py:46`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/method_config.py#L46) | Re-export from `method_config.py` |
| `FisherConfig` | [`neuralops/config/config.py:46`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L46) | [`neuralops/config/method_config.py:64`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/method_config.py#L64) | Re-export from `method_config.py` |
| `GeLoRAConfig` | [`neuralops/config/config.py:56`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L56) | [`neuralops/config/method_config.py:81`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/method_config.py#L81) | Re-export from `method_config.py` |
| `LLRDConfig` | [`neuralops/config/config.py:68`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L68) | [`neuralops/config/method_config.py:97`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/method_config.py#L97) | Re-export from `method_config.py` |
| `MixoutConfig` | [`neuralops/config/config.py:78`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L78) | [`neuralops/config/method_config.py:114`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/method_config.py#L114) | Re-export from `method_config.py` |
| `DisLoRAConfig` | [`neuralops/config/config.py:88`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L88) | [`neuralops/config/method_config.py:128`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/method_config.py#L128) | Re-export from `method_config.py` |
| `MethodConfig` | [`neuralops/config/config.py:99`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L99) | [`neuralops/config/method_config.py:145`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/method_config.py#L145) | Re-export from `method_config.py` |
| `HardwareConfig` | [`neuralops/config/config.py:133`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L133) | [`neuralops/config/hardware_config.py:11`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/hardware_config.py#L11) | Re-export from `hardware_config.py` |
| `DatasetConfig` | [`neuralops/config/config.py:142`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L142) | [`neuralops/config/experiment_config.py:33`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/experiment_config.py#L33) | Re-export from `experiment_config.py` |
| `ModelConfig` | [`neuralops/config/config.py:156`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L156) | [`neuralops/config/model_config.py:11`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/model_config.py#L11) | Re-export from `model_config.py` |
| `DataConfig` | [`neuralops/config/config.py:171`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L171) | [`neuralops/config/data_config.py:11`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/data_config.py#L11) | Re-export from `data_config.py` |
| `CompressionConfig` | [`neuralops/config/config.py:199`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L199) | [`neuralops/config/compression_config.py:11`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/compression_config.py#L11) | Re-export from `compression_config.py` |
| `SweepConfig` | [`neuralops/config/config.py:214`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L214) | [`neuralops/config/sweep_config.py:11`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/sweep_config.py#L11) | Re-export from `sweep_config.py` |
| `ExperimentConfig` | [`neuralops/config/config.py:228`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py#L228) | [`neuralops/config/experiment_config.py:84`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/experiment_config.py#L84) | Re-export from `experiment_config.py` |

### 3.2 Cross-Package Class Duplications
| Class Name | Locations | Analysis & Plan |
| :--- | :--- | :--- |
| `ResultsSaver` | [`neuralops/results_saver/saver.py:15`](file:///Users/tahamajs/Documents/uni/Project/neuralops/results_saver/saver.py#L15)<br>[`neuralops/utils/results_saver.py:15`](file:///Users/tahamajs/Documents/uni/Project/neuralops/utils/results_saver.py#L15) | Diverged implementations (907 lines diff). Consolidate into `neuralops/results_saver/saver.py`, make `utils/results_saver.py` an alias. |
| `LoRALinear` | [`neuralops/utils/lora.py:14`](file:///Users/tahamajs/Documents/uni/Project/neuralops/utils/lora.py#L14)<br>[`neuralops/trainer/methods/lora.py:15`](file:///Users/tahamajs/Documents/uni/Project/neuralops/trainer/methods/lora.py#L15) | Training vs inference utility duplication (426 lines diff). Consolidate to `neuralops/trainer/methods/lora.py`. |
| `MetricsTracker` | [`neuralops/analysis/metrics.py:15`](file:///Users/tahamajs/Documents/uni/Project/neuralops/analysis/metrics.py#L15)<br>[`neuralops/experiment_logging/tracker.py:20`](file:///Users/tahamajs/Documents/uni/Project/neuralops/experiment_logging/tracker.py#L20) | Different functionality sharing same class name. Rename `analysis/metrics.py:MetricsTracker` to avoid confusion. |
| `TrackerCallback` | [`neuralops/tracking/callbacks.py:14`](file:///Users/tahamajs/Documents/uni/Project/neuralops/tracking/callbacks.py#L14)<br>[`neuralops/experiment_utils/runner.py:328`](file:///Users/tahamajs/Documents/uni/Project/neuralops/experiment_utils/runner.py#L328) | Direct duplication of callback interface across experiment runners. |
| `SyntheticDataset` | [`neuralops/data/prepare_datasets.py:133`](file:///Users/tahamajs/Documents/uni/Project/neuralops/data/prepare_datasets.py#L133)<br>[`neuralops/trainer/train_lra.py:45`](file:///Users/tahamajs/Documents/uni/Project/neuralops/trainer/train_lra.py#L45) | Duplicate PyTorch synthetic dataset definition. Move canonical synthetic dataset to `neuralops/data/`. |

---

## 4. Key Cross-File Function Duplications

| Function Name | Locations | Consolidated Canonical Target |
| :--- | :--- | :--- |
| `count_parameters` | [`neuralops/utils/helpers.py:45`](file:///Users/tahamajs/Documents/uni/Project/neuralops/utils/helpers.py#L45)<br>[`neuralops/trainer/trainer_svd.py:130`](file:///Users/tahamajs/Documents/uni/Project/neuralops/trainer/trainer_svd.py#L130)<br>[`neuralops/trainer/utils.py:405`](file:///Users/tahamajs/Documents/uni/Project/neuralops/trainer/utils.py#L405) | Import canonical from [`neuralops/utils/helpers.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/utils/helpers.py) |
| `get_logger` | [`neuralops/utils/logger.py:30`](file:///Users/tahamajs/Documents/uni/Project/neuralops/utils/logger.py#L30)<br>[`neuralops/experiment_logging/logger.py:194`](file:///Users/tahamajs/Documents/uni/Project/neuralops/experiment_logging/logger.py#L194) | Centralize in [`neuralops/utils/logger.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/utils/logger.py) |
| `parse_args` | [`neuralops/evaluation/run_baseline_comparison.py:97`](file:///Users/tahamajs/Documents/uni/Project/neuralops/evaluation/run_baseline_comparison.py#L97)<br>[`neuralops/evaluation/run_lm_eval_harness.py:59`](file:///Users/tahamajs/Documents/uni/Project/neuralops/evaluation/run_lm_eval_harness.py#L59) | Standard CLI parser for standalone scripts (OK to keep per CLI script). |

---

## 5. Prioritized Consolidation Plan

1. **Priority 1 (Exact File Duplication)**:
   - Make [`neuralops/experiment_utils/runner.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/experiment_utils/runner.py) import and re-export everything from [`neuralops/experiment_utils/main.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/experiment_utils/main.py) with a deprecation notice.
2. **Priority 2 (Configuration Unification)**:
   - Make [`neuralops/config/config.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/config/config.py) re-export classes from modular configs (`method_config.py`, `hardware_config.py`, `model_config.py`, `compression_config.py`, etc.) instead of maintaining parallel duplicate dataclasses.
3. **Priority 3 (Utility Centralization)**:
   - Remove redundant `count_parameters` implementations from `neuralops/trainer/` and import from `neuralops/utils/helpers.py`.
   - Remove duplicated `ResultsSaver` and `LoRALinear` implementations by delegating to canonical modules.
