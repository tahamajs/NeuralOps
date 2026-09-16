# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS_FULL}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Path used | Short (L=128) |
| Block size | {{BLOCK_SIZE}} |
| Top-K | {{TOP_K}} |

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

## C.4 Block Size Ablation (Simulated L=512)

| $B$ | D acc | Time ratio |
|---|---|---|
| 32 | {{D_B32}} | 0.5× |
| 64 | {{D_B64}} | 1.0× |
| 128 | {{D_B128}} | 2.0× |

## C.5 Top-K Ablation (Simulated L=512)

| $K$ | D acc | Time ratio |
|---|---|---|
| 2 | {{D_K2}} | 0.5× |
| 4 | {{D_K4}} | 1.0× |
| 8 | {{D_K8}} | 2.0× |

## C.6 LoRA Rank Ablation

| $r_L$ | D acc | R acc | G acc |
|---|---|---|---|
| 0 | {{D_RL0}} | {{R_RL0}} | {{G_RL0}} |
| 4 | {{D_RL4}} | {{R_RL4}} | {{G_RL4}} |
| 16 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| 32 | {{D_RL32}} | {{R_RL32}} | {{G_RL32}} |

## C.7 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS_FULL}},
    "accuracy_pct": {{D_ACC}},
    "block_size": {{BLOCK_SIZE}},
    "top_k": {{TOP_K}},
    "path_used": "short",
    "sequence_length": 128
  }
}
```
```

---