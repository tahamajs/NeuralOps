# Benchmark Leaderboards

## Target: `roberta-base` on `sst2`

| Rank | Module | Low-Rank | Accuracy | Duration (s) | Peak VRAM |
|---|---|---|---|---|---|
| 1 | `moe_deltakv_paged` | 64 | 0.8233 | 65.81s | 1001.2MB |
| 2 | `native_hybrid_attention` | 64 | 0.8067 | 44.14s | 989.0MB |
| 3 | `lola_attention` | 64 | 0.6500 | 34.75s | 878.3MB |
| 4 | `halo_regla` | 64 | 0.5167 | 45.47s | 978.4MB |
| 5 | `dynamic_conv_attention` | 64 | 0.4833 | 56.19s | 1117.2MB |
