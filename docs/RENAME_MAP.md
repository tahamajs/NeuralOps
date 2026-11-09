# NeuralOps Rename & Legacy Symbol Mapping

This document indexes historical symbol names, module paths, and alias migrations in NeuralOps.

## Package-Level Identifiers

| Old Identifier | Canonical Identifier | Notes |
| :--- | :--- | :--- |
| `SVD_linear_Attention` | `neuralops` | Primary package distribution and import namespace |
| `neuralops ui` (for TUI) | `neuralops tui` | Separated terminal inspector from FastAPI web dashboard |

## Attention Registry Mappings

| Legacy / Deprecated Alias | Canonical Key | Family |
| :--- | :--- | :--- |
| `svd_linear_attn` | `svd_linear` | `svd` |
| `svd_linear_inject` | `svd_linear` | `svd` |
| `cosformer_attention` | `cosformer` | `linear` |
| `chela_slay_attention` | `chela_slay` | `recurrent` |
| `eat_attention` | `eat` | `recurrent` |
| `hgrn2_outer` | `hgrn2_gated_outer` | `gated` |

## Compressor Registry Mappings

| Legacy / Deprecated Alias | Canonical Key | Paradigm |
| :--- | :--- | :--- |
| `asvd_compressor` | `asvd` | `low_rank` |
| `hadamard_svd` | `hadamard_incoherent_svd` | `low_rank` |
| `admm_pruner` | `admm_prune` | `pruning` |

## Legacy Aliases Lookup Table
All backwards-compatibility aliases are formally registered in `neuralops.core.legacy_aliases` and checked against `neuralops/core/deprecated_aliases.json`.

---

## Deprecation Policy Reference
For the complete deprecation schedule, warning lifecycle (minimum one full MINOR cycle), and breaking change requirements, see the [Deprecation Policy & Semantic Versioning Contract](MIGRATION_GUIDE.md#deprecation-policy) in `docs/MIGRATION_GUIDE.md`.
