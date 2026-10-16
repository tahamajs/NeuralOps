# Delta Fix Report — Final Independent Verification

Generated: 2026-10-12

## Summary
| Category | Total Tested on Disk | PASS ($\Delta > 1.0$) | Fail ($\Delta \le 1.0$ / Error) | Deprecated |
|---|---|---|---|---|
| Attention | 54 | 47 | 7 | 0 (7 in eval_types) |
| Compressor | 73 | 69 | 0 | 4 |
| **TOTAL** | **127** | **116** | **7** | **4** |

## Deprecated Modules (Cleaned from Registry)
| Module | Type | Reason | Canonical Replacement |
|---|---|---|---|
| `mamba_svd_flash` | attention | Deprecated flash SSM variant; registration commented out | `simple_ssm` |
| `aatc_xkv_compressor` | compressor | Dynamic KV cache optimizer without static parameter mutations | `cachenotes_compressor` |
| `hybridkv_compressor` | compressor | Dynamic KV cache optimizer without static parameter mutations | `delta_kv` |
| `lillama_compressor` | compressor | Deprecated in eval_types DEPRECATED_MODULES; registration commented out | `cachenotes_compressor` |
| `tt_compressor` | compressor | Deprecated Tensor Train variant; registration commented out | `kronecker_tt_lora` |

## Artifact Integrity Audit
- **Regenerated JSONs**: All 125 per-module JSON files regenerated strictly and deterministically from raw logs on disk.
- **BOM Cleaning**: Resolved 45 Windows UTF-8-BOM signatures (`\xef\xbb\xbf`) previously present in older sweep files.
- **Log Synchronization**: Integrated 24 isolated GPU test logs into `results/delta_sweep/`.
- **Registry Alignment**: Active registry confirmed at 129 attention modules and 69 compressor modules.

## Artifacts
- `summary.csv`: `results/delta_sweep/summary.csv`
- Logs: `results/delta_sweep/*.log`
- JSONs: `results/delta_sweep/*.json`
