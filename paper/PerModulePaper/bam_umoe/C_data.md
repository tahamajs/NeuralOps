# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 66,955,010 |
| Params (after) | {{D_PARAMS_FULL}} |
| Parameter increase | +{{D_PARAM_DELTA}}M |
| Ratio | {{D_PARAM_RATIO}}× |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Step-0 diff | {{STEP0_DIFF}} |
| Expert utilization entropy | {{D_UTIL_ENTROPY}} bits |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,647,170 |
| Params (after) | {{R_PARAMS_FULL}} |
| Parameter increase | +{{R_PARAM_DELTA}}M |
| Accuracy | {{R_ACC}}% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,441,344 |
| Params (after) | {{G_PARAMS_FULL}} |
| Parameter increase | +{{G_PARAM_DELTA}}M |
| Accuracy | {{G_ACC}}% |
| ... | ... |

## C.4 Expert Count Ablation

| E | D acc | Params delta |
|---|---|---|
| 2 | {{D_E2}} | +{{D_E2P}}M |
| 4 | {{D_E4}} | +{{D_E4P}}M |
| 8 | {{D_ACC}} | +{{D_PARAM_DELTA}}M |
| 16 | {{D_E16}} | +{{D_E16P}}M |

## C.5 Top-K Ablation

| k | D acc | Time ratio |
|---|---|---|
| 1 | {{D_K1}} | 1.0× |
| 2 | {{D_ACC}} | 1.5× |
| 4 | {{D_K4}} | 2.5× |

## C.6 Gate Bias Ablation

| $b_g$ | D acc | Notes |
|---|---|---|
| 0.0 | {{D_BG0}} | Unstable |
| -1.0 | {{D_BG1}} | Faster ramp |
| -3.0 | {{D_ACC}} | Default |
| -5.0 | {{D_BG5}} | Slow ramp |

## C.7 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS_FULL}},
    "accuracy_pct": {{D_ACC}},
    "num_experts": {{NUM_EXPERTS}},
    "top_k": {{TOP_K}},
    "step0_diff": {{STEP0_DIFF}}
  }
}
```
