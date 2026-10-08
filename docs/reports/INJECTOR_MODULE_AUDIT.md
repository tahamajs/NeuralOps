# Injector Module Audit

Audit of injector functions and class aliases in `ATTENTION_REGISTRY`.

## 1. Audit Findings

| Key | Object Type | Target Object | Module | Notes |
|:---|:---:|:---|:---|:---|
| `inject_triton_cosformer` | **CLASS** | `TritonCosformerAttention` | `neuralops.attentions.linear.triton_cosformer` | Deprecated alias pointing to class `TritonCosformerAttention` |
| `triton_cosformer_inject` | **CLASS** | `TritonCosformerAttention` | `neuralops.attentions.linear.triton_cosformer` | Deprecated alias pointing to class `TritonCosformerAttention` |
| `inject_triton_svd_linear_attention` | **FUNCTION** | `inject_triton_svd_linear_attention()` | `neuralops.attentions.svd.triton_svd_linear_attention` | Standalone model monkeypatch/injection function |
| `triton_svd_linear_attention_inject` | **FUNCTION** | `inject_triton_svd_linear_attention()` | `neuralops.attentions.svd.triton_svd_linear_attention` | Redundant alias pointing to the injection function |
| `inject_xattention` | **FUNCTION** | `inject_xattention()` | `neuralops.attentions.fast_kernels.xattention` | Standalone model monkeypatch/injection function |

## 2. Recommendation

1. **Functions in Attention Registry**: Functions `inject_triton_svd_linear_attention` and `inject_xattention` (plus alias `triton_svd_linear_attention_inject`) are procedural injection utilities rather than `nn.Module` attention mechanisms. They must be removed from `ATTENTION_REGISTRY` while retaining their public import availability.
2. **Class Aliases**: `inject_triton_cosformer` and `triton_cosformer_inject` are redundant naming variants of `triton_cosformer` class and should be retired from primary decorators into `deprecated_aliases.json`.
