# Duplicate Module Audit

Analysis of candidate duplicate module keys in `ATTENTION_REGISTRY` to determine class identity and consolidation actions.

## 1. Class Equivalence Audit Results

| Candidate Duplicate | Canonical Target | Class Identity | Resolved Class Path | Action |
|:---|:---|:---:|:---|:---|
| `adaptive_svd` | `adaptive_svd_attention` | **SAME** | `neuralops.attentions.svd.adaptive_svd_attention.AdaptiveSVDAttention` | Remove duplicate decorator; add alias |
| `asvd` | `asvd_attention` | **SAME** | `neuralops.attentions.svd.asvd_attention.ASVDAttention` | Remove duplicate decorator; add alias |
| `fms` | `factorised_matrix_state` | **SAME** | `neuralops.attentions.svd.factorised_matrix_state.FactorisedMatrixState` | Remove duplicate decorator; add alias |
| `hgrn2_gated_outer` | `hgrn2_outer` | **SAME** | `neuralops.attentions.gated.hgrn2_gated_outer.HGRN2GatedOuterBlock` | Remove duplicate decorator; add alias |
| `hymba_transx_attn` | `hymba_transx` | **SAME** | `neuralops.attentions.ssm.hymba_transx.HymbaTransXBlock` | Remove duplicate decorator; add alias |
| `nsa` | `nsa_attention` | **SAME** | `neuralops.attentions.sparse.nsa_attention.NativeSparseAttention` | Remove duplicate decorator; add alias |
| `simplessm` | `simple_ssm` | **SAME** | `neuralops.attentions.ssm.selective_ssm.SelectiveSSM` | Remove duplicate decorator; add alias |
| `svd_linear_attn` | `svd_linear_attention` | **SAME** | `neuralops.attentions.svd.svd_linear_attention.SVDLinearAttention` | Remove duplicate decorator; add alias |
| `vig` | `vig_attention` | **SAME** | `neuralops.attentions.modern.vig_attention.ViGAttention` | Remove duplicate decorator; add alias |
| `wgqa` | `wgqa_attention` | **SAME** | `neuralops.attentions.modern.wgqa_attention.WeightedGroupedQueryAttention` | Remove duplicate decorator; add alias |
| `x_attention` | `xattention` | **SAME** | `neuralops.attentions.fast_kernels.xattention.XAttention` | Remove duplicate decorator; add alias |
| `grouped_query_attention` | `gqa` | **SAME** | `neuralops.attentions.modern.gqa_attention.GroupedQueryAttention` | Remove duplicate decorator; add alias |
| `adaptive_sparse_attn` | `adaptive_sparse` | **SAME** | `neuralops.attentions.sparse.adaptive_sparse_attention.AdaptiveSparseAttention` | Remove duplicate decorator; add alias |
| `selective_mamba` | `selective_ssm` | **DIFFERENT** | `neuralops.attentions.ssm.upi_stuffed_mamba.SelectiveMamba` vs `SelectiveSSM` | Keep both; distinct architectures |

## 2. Key Findings

1. **13 of 14 pairs are identical classes**: They simply had redundant `@ATTENTION_REGISTRY.register()` decorators on the same class definition.
2. **`selective_mamba` vs `selective_ssm`**: Point to different implementations (`UPIStuffedMamba.SelectiveMamba` vs `SelectiveSSM`). Both must remain registered in `neuralops.attentions.ssm`.
3. **Consolidation Safety**: All 13 duplicate decorator names will be transitioned to `deprecated_aliases.json` to ensure 100% backward compatibility.
