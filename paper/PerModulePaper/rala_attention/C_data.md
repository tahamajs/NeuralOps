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
| Tokens/sec | {{D_TPS}} |
| Alpha final | {{D_ALPHA_FINAL}} |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Peak VRAM | {{R_VRAM}} MB |
| Inference time | {{R_TIME}} s |
| Alpha final | {{R_ALPHA_FINAL}} |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Peak VRAM | {{G_VRAM}} MB |
| Inference time | {{G_TIME}} s |
| Alpha final | {{G_ALPHA_FINAL}} |

## C.4 Alpha Trajectory

### DistilBERT

| Step | Alpha |
|---|---|
| 0 | 0.0000 |
| 10 | {{D_A10}} |
| 25 | {{D_A25}} |
| 50 | {{D_A50}} |

### RoBERTa

| Step | Alpha |
|---|---|
| 0 | 0.0000 |
| 10 | {{R_A10}} |
| 25 | {{R_A25}} |
| 50 | {{R_A50}} |

### GPT-2

| Step | Alpha |
|---|---|
| 0 | 0.0000 |
| 50 | {{G_A50}} |

## C.5 Rank Ablation (DistilBERT)

| Rank | Acc | Extra params | Time |
|---|---|---|---|
| 1 | {{D_R1}} | 6,150 | {{D_R1_T}} |
| 2 | {{D_R2}} | 12,294 | {{D_R2_T}} |
| 4 | {{D_R4}} | 24,582 | {{D_R4_T}} |
| 8 | {{D_R8}} | 49,158 | {{D_R8_T}} |
| 16 | {{D_R16}} | 98,310 | {{D_R16_T}} |

## C.6 Kernel Ablation

| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| softplus | {{D_SOFT}} | {{R_SOFT}} | {{G_SOFT}} |
| elu+1 | {{D_ELU}} | {{R_ELU}} | {{G_ELU}} |
| ReLU | {{D_RELU}} | {{R_RELU}} | {{G_RELU}} |

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
    "accuracy": {{D_ACC}},
    "params_after": {{D_PARAMS}},
    "alpha_final": {{D_ALPHA_FINAL}},
    "peak_vram_mb": {{D_VRAM}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.9 Per-Layer Alpha

| Layer | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_L0_A}} | {{R_L0_A}} | {{G_L0_A}} |
| 1 | {{D_L1_A}} | {{R_L1_A}} | {{G_L1_A}} |
| ... | ... | ... | ... |

---