# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 66,955,010 |
| Params (after) | 293000000 |
| Parameter increase | +226M |
| Ratio | 4.4× |
| Accuracy | 90.02% |
| F1 | 0.9043 |
| Peak VRAM | 770 MB MB |
| Inference time | 4.5 s |
| Step-0 diff | 1.2e-6 |
| Expert utilization entropy | 2.95 bits |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,647,170 |
| Params (after) | 570000000 |
| Parameter increase | +445M |
| Accuracy | 84.63% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,441,344 |
| Params (after) | 570000000 |
| Parameter increase | +445M |
| Accuracy | 81.42% |
| ... | ... |

## C.4 Expert Count Ablation

| E | D acc | Params delta |
|---|---|---|
| 2 | 89.0 | +56M |
| 4 | 89.3 | +113M |
| 8 | 90.02 | +226M |
| 16 | 89.5 | +452M |

## C.5 Top-K Ablation

| k | D acc | Time ratio |
|---|---|---|
| 1 | 88.5 | 1.0× |
| 2 | 90.02 | 1.5× |
| 4 | 89.3 | 2.5× |

## C.6 Gate Bias Ablation

| $b_g$ | D acc | Notes |
|---|---|---|
| 0.0 | 84.5 | Unstable |
| -1.0 | 88.8 | Faster ramp |
| -3.0 | 90.02 | Default |
| -5.0 | 88.5 | Slow ramp |

## C.7 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 293000000,
    "accuracy_pct": 90.02,
    "num_experts": 8,
    "top_k": 2,
    "step0_diff": 1.2e-6
  }
}
```

---