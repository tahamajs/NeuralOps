# Benchmark Leaderboards

## Target: `textattack/roberta-base-SST-2` on `sst2`

| Rank | Module | Low-Rank | Accuracy | Duration (s) | Peak VRAM |
|---|---|---|---|---|---|
| 1 | `gated_delta_net` | 64 | 0.7600 | 551.29s | 1217.9MB |
| 2 | `selective_ssm` | 64 | 0.5050 | 67.58s | 1440.3MB |
| 3 | `selective_ssm` | 64 | 0.4950 | 77.53s | 1440.3MB |
