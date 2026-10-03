# GPT-2 Decoder Sweep Report

- Baseline PPL: **19.15**
- Modules evaluated: 12

| Module | Status | PPL | PPL Ratio | Comp. | Inj. Layers | Comp. Layers | Latency (ms/tok) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| full | PASS | 19.148 | 1.0 | 1.0 | 0 | 0 | 0.3451 |
| block_transformer_hiba | FAIL_PPL_DIVERGED | 944.549 | 49.3289 | 0.9906 | 12 | 0 | 0.9059 |
| chela_slay | FAIL_PPL_DIVERGED | 3130.5486 | 163.4922 | 0.989 | 12 | 0 | 0.639 |
| cluster | PASS | 19.1478 | 1.0 | 0.9906 | 12 | 0 | 0.2441 |
| flash3 | PASS | 19.148 | 1.0 | 0.9906 | 12 | 0 | 0.1376 |
| fusion_attn | FAIL_PPL_DIVERGED | 828.9545 | 43.292 | 0.989 | 12 | 0 | 0.6426 |
| gated_linear_attn | FAIL_PPL_DIVERGED | 14022.3471 | 732.3139 | 0.9351 | 12 | 0 | 0.8138 |
| hamming | FAIL_PPL_DIVERGED | 2361.6855 | 123.3385 | 0.9906 | 12 | 0 | 0.3206 |
| hla | PASS | -1.0 | -0.0522 | 0.9908 | 12 | 0 | 190.1157 |
| jamba | FAIL_PPL_DIVERGED | 1052.1325 | 54.9474 | 0.5465 | 12 | 0 | 3.6477 |
| qcqa | FAIL_PPL_DIVERGED | 740.5466 | 38.6749 | 1.073 | 12 | 0 | 0.2182 |
| routing | FAIL_PPL_DIVERGED | 513.2293 | 26.8033 | 1.2197 | 12 | 0 | 1.3722 |