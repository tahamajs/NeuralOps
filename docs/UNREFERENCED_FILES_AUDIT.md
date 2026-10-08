# Unreferenced Files Audit Report (Task 5.2 - Report Only)

**Date**: 2026-10-08  
**Scope**: Deep analysis of all 23 unreferenced files identified in Phase 2 across `neuralops/` and `scripts/`.  
**Status**: **REPORT ONLY** — No files have been deleted, moved, or modified.

---

## 1. Summary Classification Table

| # | File Path | Lines | Category | Proposed Action | Rationale |
|---|---|:---:|:---:|:---:|---|
| 1 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 684 | Cat 3 (Utility) | **KEEP** | Active publication plotting & markdown table generator for research papers. |
| 2 | `neuralops/config/data_config.py` | 22 | Cat 4 (Config) | **KEEP** | Modular dataclass to be wired into `config.py` during Session 4. |
| 3 | `neuralops/config/hardware_config.py` | 13 | Cat 4 (Config) | **KEEP** | Modular dataclass to be wired into `config.py` during Session 4. |
| 4 | `neuralops/config/method_config.py` | 61 | Cat 4 (Config) | **KEEP** | Modular dataclass to be wired into `config.py` during Session 4. |
| 5 | `neuralops/config/sweep_config.py` | 15 | Cat 4 (Config) | **KEEP** | Modular dataclass to be wired into `config.py` during Session 4. |
| 6 | `neuralops/data/dataset_discovery.py` | 12 | Cat 3 (Utility) | **KEEP** | Public utility for dataset introspection. |
| 7 | `neuralops/data/prepare_cache_once.py` | 17 | Cat 1 (Script) | **ARCHIVE** | Standalone one-time caching prep script. |
| 8 | `neuralops/evaluation/run_targeted_b2_recovery.py` | 99 | Cat 1 (Script) | **ARCHIVE** | Standalone evaluation run script; superseded by benchmarks CLI. |
| 9 | `neuralops/trainer/rl_trainer.py` | 177 | Cat 3 (Utility) | **KEEP** | Experimental RL / preference optimization trainer with TRL fallback. |
| 10 | `neuralops/trainer/schedulers.py` | 65 | Cat 3 (Utility) | **KEEP** | Learning rate schedulers for custom training loops. |
| 11 | `neuralops/trainer/trainer_svd.py` | 133 | Cat 3 (Utility) | **KEEP** | Standalone SVD trainer utility (re-exports `count_parameters`). |
| 12 | `neuralops/trainer/unsloth_trainer.py` | 82 | Cat 3 (Utility) | **KEEP** | Experimental fast-LoRA trainer with Unsloth support. |
| 13 | `scripts/audit_causal_modules.py` | 220 | Cat 1 (Script) | **ARCHIVE** | One-off static AST audit script used during initial development. |
| 14 | `scripts/debug_nystrom_model.py` | 54 | Cat 1 (Script) | **DELETE** | Disposable throwaway debugging script for Nyström attention. |
| 15 | `scripts/detailed_module_audit.py` | 185 | Cat 1 (Script) | **ARCHIVE** | One-off causal audit script. |
| 16 | `scripts/queue_unified_runs.py` | 62 | Cat 1 (Script) | **ARCHIVE** | Ad-hoc run queue manager; superseded by `unified_orchestrator.py`. |
| 17 | `scripts/run_kd_all.py` | 258 | Cat 2 (Superseded) | **ARCHIVE** | Sequential KD runner; superseded by benchmarks CLI unified sweep. |
| 18 | `scripts/run_kd_reuse_model.py` | 234 | Cat 2 (Superseded) | **ARCHIVE** | Sequential KD runner; superseded by benchmarks CLI. |
| 19 | `scripts/run_niah_all.py` | 201 | Cat 2 (Superseded) | **ARCHIVE** | Sequential NIAH runner; superseded by `RulerAdapter`. |
| 20 | `scripts/run_niah_sequential.py` | 96 | Cat 2 (Superseded) | **ARCHIVE** | Sequential NIAH runner; superseded by `RulerAdapter`. |
| 21 | `scripts/test_all_causality.py` | 94 | Cat 1 (Script) | **ARCHIVE** | One-off empirical causality test script. |
| 22 | `scripts/test_nan.py` | 110 | Cat 1 (Script) | **DELETE** | Throwaway NaN debugging probe. |
| 23 | `scripts/test_nystrom_direct.py` | 24 | Cat 1 (Script) | **DELETE** | Throwaway 24-line Nyström test script. |

---

## 2. Detailed Audit per File

### File 1: `neuralops/analysis/generate_sweep_publication_artifacts.py`
- **Lines**: 684
- **Purpose**: Publication-grade sweep analysis, reporting, and visualization engine.
- **References**: Code: 0 | Configs: 0 | Docs: 3
- **Imported By**: None (standalone CLI analysis entry point)
- **Category**: Cat 3 (Utility)
- **Proposed Action**: **KEEP**
- **Reason**: Active tool used to generate camera-ready artifacts and tables from benchmark sweep directories.

### Files 2–5: Modular Configs (`neuralops/config/{data,hardware,method,sweep}_config.py`)
- **Lines**: 22, 13, 61, 15
- **Purpose**: Modular dataclass configuration schemas.
- **References**: Code: 0 (direct references bypassed by monolithic `config.py`) | Docs: 3–4
- **Category**: Cat 4 (Config)
- **Proposed Action**: **KEEP**
- **Reason**: These modular config files represent the desired architectural target. In Session 4 (Config Unification), `config.py` will re-export these classes, eliminating duplication.

### File 6: `neuralops/data/dataset_discovery.py`
- **Lines**: 12
- **Purpose**: Scans and discovers datasets on local filesystem.
- **References**: Code: 0 | Docs: 2
- **Category**: Cat 3 (Utility)
- **Proposed Action**: **KEEP**
- **Reason**: Useful public utility function for data directory discovery.

### File 7: `neuralops/data/prepare_cache_once.py`
- **Lines**: 17
- **Purpose**: Script to pre-tokenize datasets and cache them to disk.
- **References**: Code: 0 | Docs: 2
- **Category**: Cat 1 (Script)
- **Proposed Action**: **ARCHIVE**
- **Reason**: Standalone preparatory script.

### File 8: `neuralops/evaluation/run_targeted_b2_recovery.py`
- **Lines**: 99
- **Purpose**: Targeted B2 recovery evaluation for high zero-shot logit difference modules.
- **References**: Code: 0 | Docs: 2
- **Category**: Cat 1 (Script)
- **Proposed Action**: **ARCHIVE**
- **Reason**: Ad-hoc recovery experiment script; functionality covered by benchmark runners.

### Files 9–12: Trainer Modules (`rl_trainer.py`, `schedulers.py`, `trainer_svd.py`, `unsloth_trainer.py`)
- **Lines**: 177, 65, 133, 82
- **Purpose**: Specialized training loops (RL/TRL, custom schedulers, SVD benchmarking, fast-LoRA Unsloth).
- **References**: Code: 0 (optional/submodule exports) | Docs: 2–4
- **Category**: Cat 3 (Utility)
- **Proposed Action**: **KEEP**
- **Reason**: Valid, working trainer components providing specialized features and fallbacks.

### Files 13–23: Scripts in `scripts/`
- **Archive Candidates (8 files)**:
  - `scripts/audit_causal_modules.py` (220 lines) — Archive
  - `scripts/detailed_module_audit.py` (185 lines) — Archive
  - `scripts/queue_unified_runs.py` (62 lines) — Archive
  - `scripts/run_kd_all.py` (258 lines) — Archive
  - `scripts/run_kd_reuse_model.py` (234 lines) — Archive
  - `scripts/run_niah_all.py` (201 lines) — Archive
  - `scripts/run_niah_sequential.py` (96 lines) — Archive
  - `scripts/test_all_causality.py` (94 lines) — Archive
- **Delete Candidates (3 throwaway debug files)**:
  - `scripts/test_nan.py` (110 lines) — Throwaway debugging probe
  - `scripts/debug_nystrom_model.py` (54 lines) — Throwaway debugging script
  - `scripts/test_nystrom_direct.py` (24 lines) — Throwaway test script

---

## 3. Proposed Archive Structure

For approved archive files:
```text
scripts/_archive/neuralops_cleanup_2026_10/
├── README.md                          # Explains why scripts were archived & their modern replacements
├── audit_causal_modules.py
├── detailed_module_audit.py
├── queue_unified_runs.py
├── run_kd_all.py
├── run_kd_reuse_model.py
├── run_niah_all.py
├── run_niah_sequential.py
├── test_all_causality.py
├── prepare_cache_once.py
└── run_targeted_b2_recovery.py
```
