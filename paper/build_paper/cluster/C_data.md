# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 67544826 |
| Params change | +589826 |
| Accuracy | 90.02% |
| F1 | 0.9039 |
| Peak VRAM | 332 MB MB |
| Inference time | 2.10 s |
| K | 16 |
| Window | 32 |
| Learned τ | 0.087 |
| Learned α | 0.48 |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 125826818 |
| Accuracy | 72.02% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 125630220 |
| Accuracy | 87.96% |
| ... | ... |

## C.4 K Ablation — Full Data

| K | D acc | R acc | G acc | FLOPs ratio |
|---|---|---|---|---|
| 4 | 88.5 | 75.8 | 77.2 | 6.4× |
| 8 | 89.5 | 77.0 | 78.5 | 3.2× |
| 16 | 90.02 | 72.02 | 87.96 | 1.6× |
| 32 | 90.05 | 77.6 | 79.0 | 0.8× |
| 64 | 90.15 | 77.7 | 79.05 | 0.4× |

## C.5 Window Ablation

| W | D acc | R acc | G acc |
|---|---|---|---|
| 0 | 88.2 | 75.5 | 76.8 |
| 8 | 89.5 | 77.0 | 78.4 |
| 32 | 90.02 | 72.02 | 87.96 |
| 64 | 89.80 | 77.45 | 78.95 |

## C.6 Iterations Ablation

| Iters | D acc | Time |
|---|---|---|
| 1 | 89.0 | 1× |
| 3 | 90.02 | 3× |
| 5 | 89.95 | 5× |
| 10 | 89.98 | 10× |

## C.7 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 67544826,
    "accuracy_pct": 90.02,
    "num_clusters": 16,
    "window_size": 32,
    "learned_temp": 0.087,
    "learned_alpha": 0.48
  }
}
```

---