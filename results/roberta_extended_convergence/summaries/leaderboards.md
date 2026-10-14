# Benchmark Leaderboards

## Target: `textattack/roberta-base-SST-2` on `sst2`

| Rank | Module | Low-Rank | Accuracy | Duration (s) | Peak VRAM |
|---|---|---|---|---|---|
| 1 | `gated_delta_net` | 64 | 0.7600 | 551.29s | 1217.9MB |
| 2 | `hgrn2_gated_outer` | 64 | 0.7500 | 128.72s | 1013.9MB |
| 3 | `gated_delta_net` | 64 | 0.6950 | 691.87s | 1198.1MB |
| 4 | `selective_ssm` | 64 | 0.5050 | 67.58s | 1440.3MB |
| 5 | `selective_ssm` | 64 | 0.4950 | 77.53s | 1440.3MB |
| 6 | `gla_attention` | 64 | 0.4950 | 33.42s | 1180.1MB |
| 7 | `mamba_block` | 64 | 0.4950 | 74.23s | 1456.9MB |
| 8 | `svd_mamba` | 64 | 0.4950 | 187.39s | 1358.9MB |
