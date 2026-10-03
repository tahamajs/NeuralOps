# GPT-2 Decoder Sweep Report

- Baseline PPL: **24.86**
- Modules evaluated: 3

| Module | Status | PPL | PPL Ratio | Comp. | Inj. Layers | Comp. Layers | Latency (ms/tok) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| full | PASS | 24.8557 | 1.0 | 1.0 | 0 | 0 | 0.1077 |
| triton_cosformer | FAIL_PPL_DIVERGED | 4156.9102 | 167.2417 | 1.0 | 12 | 0 | 0.2055 |
| triton_svd_linear_attention | FAIL_PPL_DIVERGED | 578.1608 | 23.2607 | 1.1658 | 12 | 0 | 0.1777 |