# GPT-2 Decoder Sweep Report

- Baseline PPL: **21.71**
- Modules evaluated: 2

| Module | Status | PPL | PPL Ratio | Comp. | Inj. Layers | Comp. Layers | Latency (ms/tok) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| full | PASS | 21.7132 | 1.0 | 1.0 | 0 | 0 | 0.0714 |
| expert_choice_moe | PASS | 21.7132 | 1.0 | 0.9373 | 12 | 0 | 0.1853 |