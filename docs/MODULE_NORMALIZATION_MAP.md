# Module Normalization Map

Mapping rules and alias delegations for normalizing experiment module identifiers to canonical registry keys.

## 1. Automated Normalization Rules

| Pattern | Replacement | Target Canonical Type |
|:---|:---|:---|
| `_attn$` | `_attention` | Attention Mechanism |
| `_class$` | `""` | Class suffix cleanup |
| `_factory$` | `""` | Factory suffix cleanup |
| `_module$` | `""` | Module wrapper cleanup |
| `^inject_` | `""` | Injection wrapper cleanup |
| `^pre_` | `""` | Preflight prefix cleanup |
| `_v[0-9]+$` | `""` | Versioned unifier |

## 2. Special Alias Decisions

- `aegis_v1` -> `hybrid_local_kernel_attention`
- `aegis_v2` -> `hybrid_local_kernel_attention`
- `aegis_v3` -> `hybrid_local_kernel_attention`
- `aegis_v4` -> `hybrid_local_kernel_attention`
- `aegis_v5` -> `split_head_hybrid_attention`
- `aegis_v6` -> `gated_hybrid_kernel_attention`
- `aegis_hybrid` -> `aegis_hybrid`
- `leka_v1` -> `positive_kernel_linear_attention`
- `leka_v2` -> `landmark_local_attention`
- `leka_v3` -> `refined_landmark_attention`
- `fast_softmax` -> `fast_softmax_attention`
- `cur_linear` -> `cur_compressor`

All 641 registered aliases automatically map to canonical names via `neuralops.core.registry`.
