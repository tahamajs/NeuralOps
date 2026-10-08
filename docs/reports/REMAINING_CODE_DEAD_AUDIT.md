# Remaining Codebase Dead Code Audit

Comprehensive dead code, unused import, and unreferenced file analysis across remaining neuralops subsystems (excluding `attentions/`, `compression_engine/`, `benchmarks/`, and `core/`).

---

## 1. Summary

- **Total Dead / Unused Imports & Variables**: 327 occurrences (mostly in old unit test files).
- **Total Dead Functions**: 80 unreferenced functions in active utility/analysis modules.
- **Total Dead Classes**: 7 unreferenced classes (`FocalLoss`, `DiceLoss`, `PolyLoss`, `ContrastiveLoss`, `LabelSmoothingLoss`, `CheckpointManager`, `_AliasLoader`).
- **Total Unreferenced Standalone Files**: 23 files (ad-hoc debug scripts and unlinked configs).

---

## 2. Directory Breakdown of Dead / Unreferenced Items

| Directory | Unused Imports / Vars (F401/F841) | Dead Functions | Dead Classes | Unreferenced Files |
| :--- | :---: | :---: | :---: | :---: |
| `neuralops/tests/` | 312 | 0 (tests excluded) | 0 | 0 |
| `neuralops/utils/` | 4 | 21 | 1 | 0 |
| `neuralops/trainer/` | 2 | 14 | 5 | 4 |
| `neuralops/analysis/` | 0 | 18 | 0 | 1 |
| `neuralops/results_saver/` | 0 | 8 | 0 | 0 |
| `neuralops/config/` | 0 | 0 | 0 | 4 |
| `neuralops/data/` | 0 | 5 | 0 | 2 |
| `scripts/` | 9 | 14 | 0 | 11 |
| `neuralops/evaluation/` | 0 | 0 | 0 | 1 |

---

## 3. Unreferenced Standalone Files (23 Total)

| File Path | Lines | Purpose / Description | Action Recommendation |
| :--- | :---: | :--- | :--- |
| `scripts/audit_causal_modules.py` | 220 | Causal attention audit script | **Archive** to `scripts/_archive/` |
| `scripts/run_kd_all.py` | 258 | Sequential KD runner wrapper | **Archive** (superseded by new CLI) |
| `scripts/run_kd_reuse_model.py` | 234 | KD model reuse script | **Archive** |
| `scripts/run_niah_all.py` | 201 | Sequential NIAH runner | **Archive** (superseded by RulerAdapter) |
| `scripts/detailed_module_audit.py` | 185 | Causal audit for 115 attention modules | **Archive** |
| `scripts/run_niah_sequential.py` | 96 | Sequential NIAH runner | **Archive** |
| `scripts/test_nan.py` | 110 | Ad-hoc NaN loss debugger | **Archive** |
| `scripts/test_all_causality.py` | 94 | Empirical causality tester | **Archive** |
| `scripts/queue_unified_runs.py` | 62 | Legacy GPU queue runner | **Archive** |
| `scripts/debug_nystrom_model.py` | 54 | Nyström attention debugging | **Archive** |
| `scripts/test_nystrom_direct.py` | 24 | Direct Nyström test | **Archive** |
| `neuralops/analysis/generate_sweep_publication_artifacts.py` | 684 | Publication sweep artifact generation | **Keep** (Active tool for paper plotting) |
| `neuralops/trainer/rl_trainer.py` | 177 | Reinforcement learning trainer stub | **Archive** / delete (unreferenced) |
| `neuralops/trainer/trainer_svd.py` | 132 | Standalone SVD trainer | **Check** (absorb into trainer package) |
| `neuralops/trainer/unsloth_trainer.py` | 82 | Unsloth fast-LoRA trainer wrapper | **Archive** / keep as experimental |
| `neuralops/trainer/schedulers.py` | 65 | Learning rate schedulers | **Keep** / export in `trainer/__init__.py` |
| `neuralops/evaluation/run_targeted_b2_recovery.py` | 99 | Targeted B2 recovery evaluation | **Archive** (superseded by benchmark suite) |
| `neuralops/config/method_config.py` | 61 | Unlinked config classes | **Archive** or export |
| `neuralops/config/data_config.py` | 22 | Unlinked data config | **Archive** or export |
| `neuralops/config/sweep_config.py` | 15 | Unlinked sweep config | **Archive** or export |
| `neuralops/config/hardware_config.py` | 13 | Unlinked hardware config | **Archive** or export |
| `neuralops/data/prepare_cache_once.py` | 17 | Data caching script | **Keep** (useful CLI data prep) |
| `neuralops/data/dataset_discovery.py` | 12 | Scanning dataset directories | **Keep** (utility) |

---

## 4. Dead Classes (7 Total)

| File | Line | Class | Type | Recommendation |
| :--- | :---: | :--- | :--- | :--- |
| `neuralops/trainer/losses.py` | 24 | `FocalLoss` | Loss function | **Export** in `trainer/__init__.py` for public use |
| `neuralops/trainer/losses.py` | 38 | `LabelSmoothingLoss` | Loss function | **Export** in `trainer/__init__.py` |
| `neuralops/trainer/losses.py` | 54 | `DiceLoss` | Loss function | **Export** in `trainer/__init__.py` |
| `neuralops/trainer/losses.py` | 70 | `PolyLoss` | Loss function | **Export** in `trainer/__init__.py` |
| `neuralops/trainer/losses.py` | 83 | `ContrastiveLoss` | Loss function | **Export** in `trainer/__init__.py` |
| `neuralops/utils/checkpoint.py` | 8 | `CheckpointManager` | Utility class | **Keep / Export** |
| `neuralops/__init__.py` | 23 | `_AliasLoader` | Internal loader | **Keep** (used for dynamic backward compat) |

---

## 5. Dead Functions Sample (80 Total)

Major categories of unreferenced functions:
1. **Unused Statistical Corrections** in [`neuralops/analysis/comparison/statistical_tests.py`](../neuralops/analysis/comparison/statistical_tests.py):
   `holm_correction`, `hochberg_correction`, `finner_correction`, `hommel_correction`, `holland_copenhaver_correction`, `mcnemar_test`, `bayesian_correlated_ttest`.
2. **Unused Checkpoint Helpers** in [`neuralops/utils/checkpoint.py`](../neuralops/utils/checkpoint.py):
   `delete_checkpoint`, `get_latest_checkpoint`.
3. **Unused Profiler Helpers** in [`neuralops/utils/profiler.py`](../neuralops/utils/profiler.py):
   `save_report`, `measure_memory_usage`.
4. **Unused Result Saver Backends** in [`neuralops/results_saver/saver.py`](../neuralops/results_saver/saver.py):
   `log_metric_to_sqlite`, `push_to_registries`.

---

## 6. Recommendations & Action Plan

- **Priority 1 (Scripts Cleanup)**: Move 11 unreferenced, ad-hoc one-off test scripts in `scripts/` to `scripts/_archive/`.
- **Priority 2 (Unused Test Imports)**: Strip unused imports in `neuralops/tests/` to prevent warning spam.
- **Priority 3 (Trainer Loss Function Exports)**: Export unreferenced loss functions in `neuralops/trainer/__init__.py` so they are accessible to users rather than being orphaned in `losses.py`.
