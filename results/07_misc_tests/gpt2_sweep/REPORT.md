# GPT-2 Decoder Sweep Report

- Baseline PPL: **37.71**
- Modules evaluated: 2

| Module | Status | PPL | PPL Ratio | Comp. | Inj. Layers | Comp. Layers | Latency (ms/tok) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| full | PASS | 37.7141 | 1.0 | 1.0 | 0 | 0 | 0.8649 |
| triton_cosformer | FAIL_PPL_DIVERGED | 27603.8611 | 731.9242 | 1.0 | 12 | 0 | 0.8224 |