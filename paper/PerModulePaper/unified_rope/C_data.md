<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66,955,010 |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Loss | {{D_LOSS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| RoPE base | 10000 |
| Rotary dim | {{D_ROTARY_DIM}} |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124,647,170 |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Peak VRAM | {{R_VRAM}} MB |
| Nested | True |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124,441,344 |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Peak VRAM | {{G_VRAM}} MB |
| Conv1D | True |
| Causal | True |

## C.4 RoPE Correctness

**Norm preservation:**

| Model | Before | After | Error |
|---|---|---|---|
| DistilBERT | {{D_Q_NORM_BEFORE}} | {{D_Q_NORM_AFTER}} | {{D_Q_NORM_ERR}} |
| RoBERTa | {{R_Q_NORM_BEFORE}} | {{R_Q_NORM_AFTER}} | {{R_Q_NORM_ERR}} |
| GPT-2 | {{G_Q_NORM_BEFORE}} | {{G_Q_NORM_AFTER}} | {{G_Q_NORM_ERR}} |

**Relative position:**

| Model | Score variance across absolute positions |
|---|---|
| DistilBERT | {{D_SCORE_VAR}} |
| RoBERTa | {{R_SCORE_VAR}} |
| GPT-2 | {{G_SCORE_VAR}} |

## C.5 RoPE Base Ablation

| Base | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 5000 | {{D_BASE5K}} | {{R_BASE5K}} | {{G_BASE5K}} |
| 10000 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| 100000 | {{D_BASE100K}} | {{R_BASE100K}} | {{G_BASE100K}} |

## C.6 Rotary Dim Ablation

| Rot dim | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_ROT32}} | {{R_ROT32}} | {{G_ROT32}} |
| 64 | {{D_ROT64}} | {{R_ROT64}} | {{G_ROT64}} |
| Full | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |

## C.7 Training Loss Curves

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
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

## C.8 Raw JSON

```json
{
  "distilbert": {
    "params": 66955010,
    "accuracy_pct": {{D_ACC}},
    "f1": {{D_F1}},
    "rope_base": 10000,
    "rotary_dim": {{D_ROTARY_DIM}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

---
