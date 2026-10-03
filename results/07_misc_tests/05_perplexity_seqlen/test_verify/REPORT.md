# GPT-2 Decoder Sweep Report

- Baseline PPL: **19.15**
- Modules evaluated: 2

| Module | Status | PPL | PPL Ratio | Comp. | Inj. Layers | Comp. Layers | Latency (ms/tok) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| full | PASS | 19.148 | 1.0 | 1.0 | 0 | 0 | 0.3885 |
| full_attention | PASS | 19.1478 | 1.0 | 0.9906 | 12 | 0 | 0.169 |