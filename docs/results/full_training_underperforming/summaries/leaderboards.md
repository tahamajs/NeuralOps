# Benchmark Leaderboards

## Target: `gpt2` on `sst2`

| Rank | Module | Low-Rank | Accuracy | Duration (s) | Peak VRAM |
|---|---|---|---|---|---|
| 1 | `aegis_v5` | 64 | 0.7875 | 47.75s | 859.3MB |
| 2 | `split_head_hybrid` | 64 | 0.7875 | 57.47s | 859.3MB |
| 3 | `astro_quantizer` | 64 | 0.7375 | 38.5s | 999.8MB |
| 4 | `fusion_attn` | 64 | 0.7250 | 36.84s | 904.2MB |
| 5 | `triton_cosformer` | 64 | 0.6250 | 43.03s | 991.7MB |
| 6 | `qcqa_attention` | 64 | 0.6125 | 38.93s | 902.7MB |
| 7 | `chela_slay` | 64 | 0.6125 | 45.7s | 900.0MB |
| 8 | `padre_attention` | 64 | 0.6000 | 39.6s | 884.7MB |
| 9 | `optimized_hybrid_attention` | 64 | 0.6000 | 37.21s | 889.4MB |
| 10 | `rpc_attention` | 64 | 0.5875 | 45.13s | 799.3MB |
| 11 | `griffin_ssd` | 64 | 0.5375 | 39.29s | 933.4MB |
| 12 | `zeros_attention` | 64 | 0.5250 | 49.06s | 979.9MB |
| 13 | `enlighten_compressor` | 64 | 0.5250 | 40.5s | 1027.8MB |
| 14 | `hgrn2_gated_outer` | 64 | 0.5125 | 62.13s | 1068.2MB |
| 15 | `hgrn2_outer` | 64 | 0.5125 | 60.68s | 1068.2MB |
| 16 | `grecodec_compressor` | 64 | 0.5125 | 27.29s | 446.8MB |
| 17 | `kda_attention` | 64 | 0.5000 | 48.14s | 894.3MB |
| 18 | `aegis_hybrid` | 64 | 0.5000 | 52.66s | 825.3MB |
| 19 | `tri_branch_hybrid` | 64 | 0.5000 | 53.21s | 825.3MB |
| 20 | `mgie_budget` | 64 | 0.5000 | 31.34s | 446.8MB |
