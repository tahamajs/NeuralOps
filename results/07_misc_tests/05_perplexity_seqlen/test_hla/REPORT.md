# GPT-2 Decoder Sweep Report

- Baseline PPL: **23.01**
- Modules evaluated: 2

| Module | Status | PPL | PPL Ratio | Comp. | Inj. Layers | Comp. Layers | Latency (ms/tok) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| full | PASS | 23.011 | 1.0 | 1.0 | 0 | 0 | 0.1339 |
| hla | FAIL_PPL_DIVERGED | 3096.9222 | 134.5844 | 0.9908 | 12 | 0 | 0.2304 |