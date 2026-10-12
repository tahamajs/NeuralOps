# Debug Inventory v2 (Independent Audit from Disk)

## Summary
- **Attention Modules Evaluated**: 52 logs on disk (remaining 69 modules actively processing under coordinator PID 668690)
  - PASS ($\Delta > 1.0$): 47
  - FAIL ($\Delta \le 1.0$ or ERROR / NaN): 5 (`adasplash_attention`, `factorised_matrix_state`, `full`, `hybrid_attention`, `inject_leka_v1`)
- **Compressor Modules Evaluated**: 73 logs on disk
  - PASS ($\Delta > 1.0$): 69
  - Deprecated / Canonical: 3 (`aatc_xkv_compressor`, `hybridkv_compressor`, `lillama_compressor`)
  - Log Missing / Error: 1 (`mamba_shedder` log in `results/kd_runs/` but missing from `delta_sweep/*.log`)
- **Deprecated Modules Still Active in Registry**: 3 (`mamba_svd_flash`, `lillama_compressor`, `tt_compressor`)

## Modules To Fix / Re-run
| Module | Type | Status | Delta | Root Cause | Fix Plan |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `adasplash_attention` | attention | ERROR | None | Aborted by KeyboardInterrupt in prior sweep | Re-run in isolation |
| `factorised_matrix_state` | attention | ERROR | None | Aborted by KeyboardInterrupt in prior sweep | Re-run in isolation |
| `inject_leka_v1` | attention | ERROR | None | Aborted by KeyboardInterrupt in prior sweep | Re-run in isolation |
| `full` | attention | ERROR | None | Registered as 'full' instead of full_attention | Map or alias in registry |
| `hybrid_attention` | attention | FAIL | NaN | Loss exploded to NaN during training | Clip logits or damp LR |
| `mamba_shedder` | compressor | PASS/UNSYNC | +4.74 | Run completed in kd_runs/mamba_shedder but log not synced to delta_sweep | Synchronize log file |

## Deprecated Modules to Remove from Registry
| Module | Type | Registered Name | File Location | Canonical Replacement |
| :--- | :--- | :--- | :--- | :--- |
| `mamba_svd_flash` | attention | `mamba_svd_flash` | `neuralops/attentions/ssm/mamba_svd_flash.py` | `simple_ssm` |
| `lillama_compressor` | compressor | `lillama_compressor` | `neuralops/compression_engine/experimental/lillama_compressor.py` | `cachenotes_compressor` |
| `tt_compressor` | compressor | `tt_compressor` | `neuralops/compression_engine/tensor_decomp/tt_compressor.py` | `kronecker_tt_lora` |

## JSON / Artifact Inconsistencies Detected
- 45 compressor JSON files have UTF-8-BOM signatures (`\xef\xbb\xbf`) created by PowerShell sweeps.
- 76 JSON files reported delta=0.0 while raw logs show high positive values (e.g. `Delta pre-baseline = +9849.02` or `total_gain = +7.20`) because of stale parser regexes prior to commit 619f535ab3.
- All per-module JSONs and `summary.csv` must be regenerated from disk logs.
