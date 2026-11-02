# Benchmark Leaderboards

## Target: `roberta-base` on `sst2`

| Rank | Module | Low-Rank | Accuracy | Duration (s) | Peak VRAM |
|---|---|---|---|---|---|
| 1 | `flash_attention3` | 64 | 0.9067 | 67.32s | 989.2MB |
| 2 | `flash_attention3_fp8` | 64 | 0.9067 | 67.32s | 989.2MB |
| 3 | `nasc_attention` | 64 | 0.9067 | 92.48s | 1010.8MB |
| 4 | `adaptive_span_attention` | 64 | 0.9037 | 245.21s | 932.9MB |
| 5 | `progressive_token_pruning` | 64 | 0.8967 | 69.15s | 1008.5MB |
| 6 | `adaptive_sparse_attention` | 64 | 0.8900 | 80.37s | 1013.4MB |
| 7 | `jamba` | 64 | 0.8800 | 94.65s | 1409.2MB |
| 8 | `adasplash_attention` | 64 | 0.8733 | 76.89s | 1019.2MB |
| 9 | `cross_attention` | 64 | 0.8733 | 68.67s | 989.2MB |
| 10 | `goldfinch_ahn` | 64 | 0.8733 | 104.28s | 1068.9MB |
| 11 | `halo_regla` | 64 | 0.8733 | 75.96s | 1014.3MB |
| 12 | `cluster_attention` | 64 | 0.8700 | 80.05s | 1027.7MB |
| 13 | `cosformer` | 64 | 0.8700 | 83.97s | 1090.6MB |
| 14 | `cosformer_attention` | 64 | 0.8700 | 84.75s | 1090.6MB |
| 15 | `fusion_attn` | 64 | 0.8700 | 68.7s | 990.7MB |
| 16 | `optimized_hybrid_attention` | 64 | 0.8700 | 75.79s | 1029.1MB |
| 17 | `moba_attention` | 64 | 0.8633 | 73.82s | 1007.7MB |
| 18 | `nsa_attention` | 64 | 0.8633 | 74.24s | 1023.7MB |
| 19 | `padre_attention` | 64 | 0.8633 | 72.9s | 1037.6MB |
| 20 | `eat_attention` | 64 | 0.8600 | 70.87s | 1013.7MB |
