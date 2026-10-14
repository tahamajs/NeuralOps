# Benchmark Leaderboards

## Target: `roberta-base` on `sst2`

| Rank | Module | Low-Rank | Accuracy | Duration (s) | Peak VRAM |
|---|---|---|---|---|---|
| 1 | `sink_attention` | 64 | 0.8300 | 27.0s | 947.4MB |
| 2 | `hgrn2_gated_outer` | 64 | 0.8100 | 79.83s | 1093.5MB |
| 3 | `hgrn2_outer` | 64 | 0.8100 | 81.48s | 1093.5MB |
| 4 | `dynamic_conv_attention` | 64 | 0.5200 | 40.16s | 1011.4MB |
| 5 | `mamba_block` | 64 | 0.5200 | 254.58s | 1290.6MB |
| 6 | `selective_ssm` | 64 | 0.5200 | 245.1s | 1359.0MB |
| 7 | `simple_ssm` | 64 | 0.5200 | 260.18s | 1359.0MB |
| 8 | `ttt_linear_attention` | 64 | 0.5200 | 83.45s | 1056.6MB |
| 9 | `hamming_attention` | 64 | 0.4800 | 26.63s | 957.0MB |
| 10 | `svd_mamba` | 64 | 0.4800 | 80.68s | 1397.0MB |
