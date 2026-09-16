# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Model | DistilBERT-base |
| Params before | 66,955,010 |
| Params after | {{D_PARAMS_FULL}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Path used | SDPA (RTX 4070) |
| FA3 available | {{FA3_AVAILABLE}} |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS_FULL}} |
| Accuracy | {{R_ACC}}% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS_FULL}} |
| Accuracy | {{G_ACC}}% |
| ... | ... |

## C.4 LoRA Rank Ablation

| $r_L$ | D acc | R acc | G acc | Params delta |
|---|---|---|---|---|
| 0 | {{D_RL0}} | {{R_RL0}} | {{G_RL0}} | 0 |
| 4 | {{D_RL4}} | {{R_RL4}} | {{G_RL4}} | +147K |
| 8 | {{D_RL8}} | {{R_RL8}} | {{G_RL8}} | +295K |
| 16 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} | +590K |
| 32 | {{D_RL32}} | {{R_RL32}} | {{G_RL32}} | +1.2M |

## C.5 Exactness Test Results

| Test | Max diff (SDSP vs naive) |
|---|---|
| Random $Q,K,V$ ($L=32$) | 1.2e-7 |
| Random $Q,K,V$ ($L=128$) | 3.4e-7 |
| Random $Q,K,V$ ($L=512$) | 1.1e-6 |

All below 1e-5 threshold.

## C.6 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS_FULL}},
    "accuracy_pct": {{D_ACC}},
    "path_used": "SDPA",
    "flash_attn_available": {{FA3_AVAILABLE}}
  }
}
```
```

---