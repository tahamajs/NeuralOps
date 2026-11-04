# Project Cleanup Matrix

No deletion is authorized by this matrix alone. Each candidate needs evidence from imports, exports, registries, tests, entry points, notebooks, configurations, and documentation.

| Candidate area | Current hypothesis | Evidence required | Decision |
|---|---|---|---|
| `neuralops/attentions/` | Compatibility or alternate API surface alongside `layers/` | Symbol-by-symbol import/export and registry map | Pending |
| `neuralops/compression_engine/` | Compatibility or alternate API surface alongside `compression/` | Symbol-by-symbol implementation and caller map | Pending |
| `neuralops/kernels/` | Public catalog wrappers over core kernel catalogs | Re-export and catalog parity checks | Keep pending verification |
| `neuralops/ops/cuda/` | Backend package tree with support/compatibility directories | Compare filesystem directories to 110/114 catalog counts | Keep; classify subdirectories |
| `scripts/benchmarks/` | Standalone orchestration layer | Map each script to callers and current benchmark manifest | Pending |
| `docs/`, `academic/`, `results/` | Mixed user, historical, and generated material | Provenance, ignore rules, and inbound links | Pending |
| caches and bytecode | Generated local artifacts | Confirm ignore rules and tracked status | Exclude from production structure |

## Decision vocabulary

- **Keep**: active and referenced.
- **Compatibility**: retained for public/backward-compatible imports.
- **Merge candidate**: duplicate behavior with a safe canonical target.
- **Archive candidate**: historical material with no active runtime dependency.
- **Delete candidate**: proven unreferenced and replaceable; requires explicit cleanup step.
- **Unknown**: insufficient evidence; do not change.
