# Missing Modules Analysis

## Summary

- Attention modules missing: 14
- Compressor modules missing: 0
- Total missing: 14

## Missing Attention Modules

| Module | Family | Category / Reason | Action |
|:---|:---|:---|:---|
| `adaptive_sparse_attn` | `sparse` | New baseline / no prior benchmark | Benchmark needed |
| `fused_svd_kernel_attention` | `svd` | Hardware/Kernel specific (Triton/CUDA required) | Benchmark on GPU |
| `grouped_query_attention` | `modern` | Baseline attention operator | Benchmark needed |
| `hgrn2_gated_outer` | `gated` | Recurrent/gated kernel (specialized dependencies) | Benchmark on GPU |
| `hymba_transx_attn` | `ssm` | New hybrid architecture | Benchmark needed |
| `inject_triton_cosformer` | `linear` | Triton kernel inject wrapper (CUDA required) | Benchmark on GPU |
| `inject_triton_svd_linear_attention` | `svd` | Triton kernel inject wrapper (CUDA required) | Benchmark on GPU |
| `inject_xattention` | `fast_kernels` | Fast kernel inject wrapper | Benchmark needed |
| `selective_mamba` | `ssm` | State Space selective scan | Benchmark on GPU |
| `simplessm` | `ssm` | State Space operator | Benchmark needed |
| `svd_linear_attn` | `svd` | SVD attention variant | Benchmark needed |
| `triton_cosformer_inject` | `linear` | Triton kernel inject wrapper (CUDA required) | Benchmark on GPU |
| `triton_svd_linear_attention_inject` | `svd` | Triton kernel inject wrapper (CUDA required) | Benchmark on GPU |
| `x_attention` | `fast_kernels` | Fast kernel baseline | Benchmark needed |

## Missing Compressor Modules

| Module | Family | Reason | Action |
|:---|:---|:---|:---|
*(None — all 73 compressor modules have benchmark results)*

## Detailed Categorization of Missing Modules

1. **Hardware / Kernel Specific (CUDA / Triton required)**:
   - `fused_svd_kernel_attention`
   - `inject_triton_cosformer`
   - `inject_triton_svd_linear_attention`
   - `triton_cosformer_inject`
   - `triton_svd_linear_attention_inject`
   *Reason*: Require GPU/CUDA environment and Triton compiler to run evaluations.

2. **State Space & Recurrent Operators**:
   - `selective_mamba`
   - `simplessm`
   - `hgrn2_gated_outer`
   - `hymba_transx_attn`
   *Reason*: Require specialized sequence modeling dependencies (e.g. `mamba-ssm`, `causal-conv1d`) or GPU kernels.

3. **New Baselines & Algorithmic Operators**:
   - `adaptive_sparse_attn`
   - `grouped_query_attention`
   - `inject_xattention`
   - `x_attention`
   - `svd_linear_attn`
   *Reason*: Newly registered modules or architectural baselines that have not yet had benchmark suites executed.
