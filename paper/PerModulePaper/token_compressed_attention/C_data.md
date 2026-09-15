# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Params change | +{{D_PARAM_PCT}}% |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| KV length (baseline) | 128 |
| KV length (TCA) | {{D_KVLEN}} |
| Attention FLOP reduction | {{FLOP_PCT}}% |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Score net params | 49,985 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Accuracy | {{R_ACC}}% |
| KV length (TCA) | {{R_KVLEN}} |
| Peak VRAM | {{R_VRAM}} MB |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Accuracy | {{G_ACC}}% |
| KV length (TCA) | {{G_KVLEN}} |
| Peak VRAM | {{G_VRAM}} MB |

## C.4 ρ Ablation

| ρ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.9 | {{D_R09}} | {{R_R09}} | {{G_R09}} |
| 0.7 | {{D_R07}} | {{R_R07}} | {{G_R07}} |
| 0.5 | {{D_R05}} | {{R_R05}} | {{G_R05}} |
| 0.3 | {{D_R03}} | {{R_R03}} | {{G_R03}} |
| 0.1 | {{D_R01}} | {{R_R01}} | {{G_R01}} |

## C.5 Score Net Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| norm + score_net | {{D_NL}} | {{R_NL}} | {{G_NL}} |
| norm only | {{D_N}} | {{R_N}} | {{G_N}} |
| score_net only | {{D_L}} | {{R_L}} | {{G_L}} |
| random | {{D_R}} | {{R_R}} | {{G_R}} |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
...
50    {{D_S50}}
```

## C.7 Reproduction Commands

```bash
python run_all_4_token_compressed.py --compression_ratio 0.5
```

---
