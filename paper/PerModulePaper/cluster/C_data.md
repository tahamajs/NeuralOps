# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS_FULL}} |
| Params change | +{{D_PARAM_DELTA}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| K | {{NUM_CLUSTERS}} |
| Window | {{WINDOW_SIZE}} |
| Learned τ | {{D_TAU_L}} |
| Learned α | {{D_A_L}} |

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

## C.4 K Ablation — Full Data

| K | D acc | R acc | G acc | FLOPs ratio |
|---|---|---|---|---|
| 4 | {{D_K4}} | {{R_K4}} | {{G_K4}} | 6.4× |
| 8 | {{D_K8}} | {{R_K8}} | {{G_K8}} | 3.2× |
| 16 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} | 1.6× |
| 32 | {{D_K32}} | {{R_K32}} | {{G_K32}} | 0.8× |
| 64 | {{D_K64}} | {{R_K64}} | {{G_K64}} | 0.4× |

## C.5 Window Ablation

| W | D acc | R acc | G acc |
|---|---|---|---|
| 0 | {{D_W0}} | {{R_W0}} | {{G_W0}} |
| 8 | {{D_W8}} | {{R_W8}} | {{G_W8}} |
| 32 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| 64 | {{D_W64}} | {{R_W64}} | {{G_W64}} |

## C.6 Iterations Ablation

| Iters | D acc | Time |
|---|---|---|
| 1 | {{D_IT1}} | 1× |
| 3 | {{D_ACC}} | 3× |
| 5 | {{D_IT5}} | 5× |
| 10 | {{D_IT10}} | 10× |

## C.7 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS_FULL}},
    "accuracy_pct": {{D_ACC}},
    "num_clusters": {{NUM_CLUSTERS}},
    "window_size": {{WINDOW_SIZE}},
    "learned_temp": {{D_TAU_L}},
    "learned_alpha": {{D_A_L}}
  }
}
```
