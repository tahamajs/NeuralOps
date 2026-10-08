# Results Deep Analysis

## Coverage
- Total modules: 231 (unique canonical modules across 159 attention and 73 compressor registries, with 1 overlap `svd_linear`)
- Modules with results: 217
- Modules missing results: 14
- Coverage: 93.94% of canonical modules (8.92% across all 231 modules × 63 model/dataset combinations in coverage matrix)

## Missing Modules
- `adaptive_sparse_attn`
- `fused_svd_kernel_attention`
- `grouped_query_attention`
- `hgrn2_gated_outer`
- `hymba_transx_attn`
- `inject_triton_cosformer`
- `inject_triton_svd_linear_attention`
- `inject_xattention`
- `selective_mamba`
- `simplessm`
- `svd_linear_attn`
- `triton_cosformer_inject`
- `triton_svd_linear_attention_inject`
- `x_attention`

## Unique Names in Results
- Unique module names in master summary: 252
- Mapped to canonical: 217
- Unmapped: 35 (including synthetic/test labels like `replaceme`, `dense`, `pruning_utils`, `all`)

## Models and Datasets
- Unique models: 55
- Unique datasets: 12

## Data Quality
- Total rows in master: 16,199
- Valid accuracy: 132
- Missing accuracy: 16,066
- Zero accuracy: 1
- Invalid accuracy: 0

## Content Duplicates
- Duplicate combinations: 791
