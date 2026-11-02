# Benchmark Leaderboards

## Target: `roberta-base` on `sst2`

| Rank | Module | Low-Rank | Accuracy | Duration (s) | Peak VRAM |
|---|---|---|---|---|---|
| 1 | `adaptive_span_attention` | 64 | 0.9067 | 104.8s | 979.7MB |
| 2 | `baseline` | 64 | 0.9067 | 28.63s | 1039.2MB |
| 3 | `fusion_attn` | 64 | 0.9067 | 34.96s | 998.2MB |
| 4 | `mixattention_moska` | 64 | 0.8933 | 28.11s | 994.3MB |
| 5 | `gqa_attention` | 64 | 0.8900 | 27.57s | 1007.2MB |
| 6 | `lora_attention` | 64 | 0.8900 | 31.64s | 1008.2MB |
| 7 | `moba_attention` | 64 | 0.8867 | 33.73s | 999.3MB |
| 8 | `nasc_attention` | 64 | 0.8800 | 49.46s | 1026.4MB |
| 9 | `adasplash_attention` | 64 | 0.8767 | 30.03s | 1025.5MB |
| 10 | `cluster_attention` | 64 | 0.8733 | 29.15s | 1024.3MB |
| 11 | `cosformer` | 64 | 0.8700 | 35.34s | 1090.9MB |
| 12 | `cosformer_attention` | 64 | 0.8700 | 33.83s | 1090.9MB |
| 13 | `template_attention` | 64 | 0.8700 | 30.43s | 999.6MB |
| 14 | `token_compressed_attention` | 64 | 0.8667 | 33.45s | 999.1MB |
| 15 | `nsa_attention` | 64 | 0.8633 | 27.68s | 1014.7MB |
| 16 | `progressive_token_pruning` | 64 | 0.8600 | 27.96s | 1015.6MB |
| 17 | `jamba` | 64 | 0.8533 | 40.5s | 1413.7MB |
| 18 | `asvd_attention` | 64 | 0.8467 | 30.42s | 1041.8MB |
| 19 | `padre_attention` | 64 | 0.8200 | 30.38s | 1044.6MB |
| 20 | `chela_slay` | 64 | 0.8067 | 45.68s | 1111.3MB |
