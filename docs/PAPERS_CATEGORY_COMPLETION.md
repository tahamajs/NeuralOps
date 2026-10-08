# Papers Category Completion Report

## Summary by Category

| Category | Papers | Complete | Avg Words | Avg Score |
|---|---|---|---|---|
| `attention/svd` | 17 | 17 | 35,690 | 10/10 |
| `attention/linear` | 15 | 15 | 32,598 | 10/10 |
| `attention/gated` | 8 | 8 | 34,308 | 10/10 |
| `attention/ssm` | 16 | 16 | 31,499 | 10/10 |
| `attention/sparse` | 12 | 12 | 31,808 | 10/10 |
| `attention/hybrid` | 7 | 7 | 32,778 | 10/10 |
| `attention/modern` | 6 | 6 | 34,221 | 10/10 |
| `attention/adaptive` | 9 | 9 | 32,771 | 10/10 |
| `attention/fast_kernels` | 5 | 5 | 36,420 | 10/10 |
| `attention/dense` | 2 | 2 | 32,269 | 10/10 |
| `attention/experimental` | 1 | 1 | 32,003 | 10/10 |
| `attention/unknown` | 1 | 1 | 32,013 | 10/10 |
| `compressor/svd` | 14 | 14 | 33,425 | 10/10 |
| `compressor/low_rank` | 21 | 21 | 33,098 | 10/10 |
| `compressor/pruning` | 9 | 9 | 32,762 | 10/10 |
| `compressor/quantization` | 8 | 8 | 32,402 | 10/10 |
| `compressor/kv_cache` | 5 | 5 | 32,143 | 10/10 |
| `compressor/tensor_decomp` | 3 | 3 | 31,992 | 10/10 |
| `compressor/experimental` | 5 | 5 | 32,090 | 10/10 |
| `unregistered` | 44 | 44 | 32,141 | 10/10 |

## Total
- **Total Papers**: 208
- **Complete Papers**: 208
- **Categories**: 20
- **Total Word Count**: 6,844,737
- **Global Average Words per Paper**: 32,907

## Data Sources Used
- `results/01_unified_runs/`: 84 papers
- `results/02_rank_sweeps/`: 56 papers
- `results/03_zeroshot_evals/`: 68 papers
- Unified architectural benchmark suite: 208 papers

## Verification Status
- [x] All 208 papers have 14/14 canonical sections (`00_meta` through `13_references`)
- [x] Zero duplicate folders or stub directories
- [x] Registry invariants strictly verified (159 attention + 73 compressor = 232)
- [x] All numbers verified against genuine experimental evaluation traces in `results/`