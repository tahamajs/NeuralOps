# GPT-2 Decoder Sweep Report

- Baseline PPL: **23.73**
- Modules evaluated: 5

| Module | Status | PPL | PPL Ratio | Comp. | Inj. Layers | Comp. Layers | Latency (ms/tok) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| full | PASS | 23.7263 | 1.0 | 1.0 | 0 | 0 | 0.1868 |
| gated_delta_net | FAIL_PPL_DIVERGED | 650.1731 | 27.4031 | 0.7702 | 12 | 0 | 37.8186 |
| hla | FAIL_PPL_DIVERGED | 2878.0402 | 121.3017 | 0.9908 | 12 | 0 | 0.2345 |
| phantom | FAIL_PPL_DIVERGED | 1155.2642 | 48.6913 | 1.2197 | 12 | 0 | 0.549 |
| vspec | FAIL_PPL_DIVERGED | 592.3634 | 24.9665 | 1.1886 | 12 | 0 | 0.3716 |