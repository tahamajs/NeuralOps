# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS_FULL}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Path | Galerkin |
| Rank $r$ | {{RANK}} |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS_FULL}} |
| Accuracy | {{R_ACC}}% |
| Path | Galerkin |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS_FULL}} |
| Accuracy | {{G_ACC}}% |
| Path | **Fallback** |

## C.4 Rank Ablation (Encoder)

| $r$ | D acc | R acc |
|---|---|---|
| 16 | {{D_R16}} | {{R_R16}} |
| 32 | {{D_R32}} | {{R_R32}} |
| 64 | {{D_ACC}} | {{R_ACC}} |
| 128 | {{D_R128}} | {{R_R128}} |

## C.5 LoRA Rank Ablation

| $r_L$ | D acc | R acc | G acc |
|---|---|---|---|
| 0 | {{D_RL0}} | {{R_RL0}} | {{G_RL0}} |
| 16 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| 32 | {{D_RL32}} | {{R_RL32}} | {{G_RL32}} |

## C.6 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS_FULL}},
    "accuracy_pct": {{D_ACC}},
    "rank": {{RANK}},
    "path": "galerkin"
  },
  "gpt2": {
    "params_before": 124441344,
    "params_after": {{G_PARAMS_FULL}},
    "accuracy_pct": {{G_ACC}},
    "path": "fallback"
  }
}
```
