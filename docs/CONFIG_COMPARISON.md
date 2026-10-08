# Configuration Comparison & Migration Plan (Session 4)

**Date**: 2026-10-08  
**Scope**: Comparison of monolithic `neuralops/config/config.py` against modular config files.

---

## 1. Executive Summary

| Category | Count | Classes |
|---|:---:|---|
| **Total Monolithic Classes** | 18 | `LoRAConfig`, `RDropConfig`, `R3FConfig`, `FisherConfig`, `GeLoRAConfig`, `LLRDConfig`, `MixoutConfig`, `DisLoRAConfig`, `MethodConfig`, `HardwareConfig`, `DatasetConfig`, `ModelConfig`, `DataConfig`, `TrainingConfig`, `CompressionConfig`, `SweepConfig`, `ExperimentConfig`, `Config` |
| **Total Modular Classes** | 16 | Distributed across `method_config`, `hardware_config`, `model_config`, `data_config`, `sweep_config`, `experiment_config`, `compression_config` |
| **Duplicated Classes** | **16** | 100% field-identical between monolithic and modular files |
| **Only in Monolithic** | **2** | `TrainingConfig` (24 fields) and `Config` (root aggregator) |
| **Only in Modular** | **0** | All modular classes were mirrored in the monolith |
| **External Usage** | 4 classes | `Config` (32 refs), `ExperimentConfig` (9 refs), `DataConfig` (6 refs), `DatasetConfig` (1 ref) |
| **Circular Dependencies** | **0** | Clean directed acyclic tree (`subconfigs` → `MethodConfig` / `Config`) |

---

## 2. Detailed Per-Class Comparison

| Class Name | Monolithic (`config.py`) | Modular File | Field Diff | Proposed Action |
|---|:---:|:---:|:---:|---|
| `LoRAConfig` | L21–L24 (3 fields) | `method_config.py:5` (3 fields) | Identical | **RE-EXPORT** from `method_config.py` |
| `RDropConfig` | L28–L29 (1 field) | `method_config.py:12` (1 field) | Identical | **RE-EXPORT** from `method_config.py` |
| `R3FConfig` | L33–L34 (1 field) | `method_config.py:17` (1 field) | Identical | **RE-EXPORT** from `method_config.py` |
| `FisherConfig` | L38–L41 (3 fields) | `method_config.py:22` (3 fields) | Identical | **RE-EXPORT** from `method_config.py` |
| `GeLoRAConfig` | L45–L48 (3 fields) | `method_config.py:29` (3 fields) | Identical | **RE-EXPORT** from `method_config.py` |
| `LLRDConfig` | L52–L53 (1 field) | `method_config.py:36` (1 field) | Identical | **RE-EXPORT** from `method_config.py` |
| `MixoutConfig` | L57–L58 (1 field) | `method_config.py:41` (1 field) | Identical | **RE-EXPORT** from `method_config.py` |
| `DisLoRAConfig` | L62–L64 (2 fields) | `method_config.py:46` (2 fields) | Identical | **RE-EXPORT** from `method_config.py` |
| `MethodConfig` | L68–L81 (9 fields) | `method_config.py:52` (9 fields) | Identical | **RE-EXPORT** from `method_config.py` |
| `HardwareConfig` | L85–L93 (8 fields) | `hardware_config.py:5` (8 fields) | Identical | **RE-EXPORT** from `hardware_config.py` |
| `DatasetConfig` | L97–L104 (5 fields) | `experiment_config.py:5` (5 fields) | Identical | **RE-EXPORT** from `experiment_config.py` |
| `ModelConfig` | L108–L118 (8 fields) | `model_config.py:5` (8 fields) | Identical | **RE-EXPORT** from `model_config.py` |
| `DataConfig` | L122–L139 (8 fields) | `data_config.py:7` (8 fields) | Identical | **RE-EXPORT** from `data_config.py` |
| `TrainingConfig` | L143–L169 (24 fields) | *None* | Only in monolith | **MOVE** to new `training_config.py` |
| `CompressionConfig`| L173–L253 (68 fields) | `compression_config.py:5` (67 fields)| Monolith has extra `max_grad_norm` | **MERGE** `max_grad_norm` into `compression_config.py` |
| `SweepConfig` | L257–L268 (9 fields) | `sweep_config.py:6` (9 fields) | Identical | **RE-EXPORT** from `sweep_config.py` |
| `ExperimentConfig` | L272–L293 (26 fields)| `experiment_config.py:14` (26 fields)| Identical | **RE-EXPORT** from `experiment_config.py` |
| `Config` | L300–L310 (8 fields) | *None* | Root aggregator | **KEEP** in `config.py` (imports modular classes) |

---

## 3. Execution Plan

1. **Create `neuralops/config/training_config.py`**:
   - Move `TrainingConfig` (24 fields) from `config.py` into `training_config.py`.
2. **Update `neuralops/config/compression_config.py`**:
   - Ensure `max_grad_norm: float = 1.0` is present.
3. **Rewrite `neuralops/config/config.py`**:
   - Re-export all 17 individual configuration classes from their modular files.
   - Retain root `@dataclass class Config` referencing the re-exported modular classes.
   - Expose all 18 classes in `__all__`.
4. **Verification & Tests**:
   - Verify 0 duplicate AST classes across `neuralops/config/*.py`.
   - Run tests: `pytest neuralops/tests/ -k "config"`, `test_imports.py`, `test_public_api.py`.
   - Verify registry intact: 159 attention + 73 compressor.
