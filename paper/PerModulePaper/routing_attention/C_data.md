<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Loss | {{D_LOSS}} |
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Peak VRAM | {{R_VRAM}} MB |
| Inference time | {{R_TIME}} s |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Peak VRAM | {{G_VRAM}} MB |
| Inference time | {{G_TIME}} s |

## C.4 Cluster Count Ablation

| K | DistilBERT | RoBERTa | GPT-2 | Time |
|---|---|---|---|---|
| 4 | {{D_K4}} | {{R_K4}} | {{G_K4}} | {{T_K4}} |
| 8 | {{D_K8}} | {{R_K8}} | {{G_K8}} | {{T_K8}} |
| 16 | {{D_K16}} | {{R_K16}} | {{G_K16}} | {{T_K16}} |
| 32 | {{D_K32}} | {{R_K32}} | {{G_K32}} | {{T_K32}} |

## C.5 k-Means Iterations Ablation

| $n_{\text{iters}}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | {{D_I1}} | {{R_I1}} | {{G_I1}} |
| 2 | {{D_I2}} | {{R_I2}} | {{G_I2}} |
| 3 | {{D_I3}} | {{R_I3}} | {{G_I3}} |
| 5 | {{D_I5}} | {{R_I5}} | {{G_I5}} |

## C.6 Anchor Ablation

| Anchor | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Yes | {{D_ANCHOR}} | {{R_ANCHOR}} | {{G_ANCHOR}} |
| No | {{D_NOANCHOR}} | {{R_NOANCHOR}} | {{G_NOANCHOR}} |

## C.7 Self-Token Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Self-token | {{D_SELF}} | {{R_SELF}} | {{G_SELF}} |
| No self-token | {{D_NOSELF}} | {{R_NOSELF}} | {{G_NOSELF}} |

## C.8 Q+K vs Q Ablation

| Feature | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q+K | {{D_QK}} | {{R_QK}} | {{G_QK}} |
| Q only | {{D_Q}} | {{R_Q}} | {{G_Q}} |

## C.9 Training Loss Curves

### DistilBERT

```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
...
50    {{D_S50}}
```

### RoBERTa

```
Step  Loss
1     {{R_S1}}
...
50    {{R_S50}}
```

### GPT-2

```
Step  Loss
1     {{G_S1}}
...
50    {{G_S50}}
```

## C.10 Raw JSON

```json
{
  "distilbert": {
    "accuracy": {{D_ACC}},
    "params_after": {{D_PARAMS}},
    "num_clusters": {{NUM_CLUSTERS}},
    "peak_vram_mb": {{D_VRAM}}
  },
  ...
}
```

## C.11 Reproduction

```bash
python run_all_4_routing.py --models distilbert,roberta,gpt2
```

## C.12 Cluster Size Distribution

### DistilBERT, $K=16$

| Cluster | Avg size | Max size | Min size |
|---|---|---|---|
| 0 | {{D_C0_AVG}} | {{D_C0_MAX}} | {{D_C0_MIN}} |
| 1 | {{D_C1_AVG}} | {{D_C1_MAX}} | {{D_C1_MIN}} |
| ... | ... | ... | ... |
```

---