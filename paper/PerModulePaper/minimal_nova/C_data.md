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
| Params after | {{D_PARAMS}} |
| Ratio | {{D_RATIO}}× |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Loss | {{D_LOSS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Compressed layers | {{D_COMP}} |
| Avg rank | {{D_AVG_RANK}} |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Ratio | {{R_RATIO}}× |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Peak VRAM | {{R_VRAM}} MB |
| Compressed layers | {{R_COMP}} |
| Avg rank | {{R_AVG_RANK}} |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Ratio | {{G_RATIO}}× |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Peak VRAM | {{G_VRAM}} MB |
| Compressed layers | {{G_COMP}} |
| Avg rank | {{G_AVG_RANK}} |

## C.4 Effective Rank Distribution (DistilBERT)

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | {{D_L0_RQ}} | {{D_L0_RK}} | {{D_L0_RV}} | {{D_L0_RO}} |
| 1 | {{D_L1_RQ}} | {{D_L1_RK}} | {{D_L1_RV}} | {{D_L1_RO}} |
| 2 | {{D_L2_RQ}} | {{D_L2_RK}} | {{D_L2_RV}} | {{D_L2_RO}} |
| 3 | {{D_L3_RQ}} | {{D_L3_RK}} | {{D_L3_RV}} | {{D_L3_RO}} |
| 4 | {{D_L4_RQ}} | {{D_L4_RK}} | {{D_L4_RV}} | {{D_L4_RO}} |
| 5 | {{D_L5_RQ}} | {{D_L5_RK}} | {{D_L5_RV}} | {{D_L5_RO}} |

## C.5 Effective Rank Distribution (RoBERTa)

(12 layers × 4 projections)

## C.6 Effective Rank Distribution (GPT-2)

(12 layers × 4 projections)

## C.7 Reconstruction Error

### DistilBERT
| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | {{D_L0_EQ}} | {{D_L0_EK}} | {{D_L0_EV}} | {{D_L0_EO}} |
| ... | ... | ... | ... | ... |

## C.8 Rank Cap Ablation

| $r_\text{max}$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 32 | {{D_R32}} | {{D_RATIO32}}× | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{D_RATIO64}}× | {{R_R64}} | {{G_R64}} |
| 128 | {{D_R128}} | {{D_RATIO128}}× | {{R_R128}} | {{G_R128}} |
| 192 | {{D_R192}} | {{D_RATIO192}}× | {{R_R192}} | {{G_R192}} |
| 384 | {{D_R384}} | {{D_RATIO384}}× | {{R_R384}} | {{G_R384}} |

## C.9 Energy Threshold Ablation

| $\eta$ | DistilBERT | Avg rank | RoBERTa | Avg rank | GPT-2 |
|---|---|---|---|---|---|
| 0.90 | {{D_E90}} | {{D_AR90}} | {{R_E90}} | {{R_AR90}} | {{G_E90}} |
| 0.95 | {{D_E95}} | {{D_AR95}} | {{R_E95}} | {{R_AR95}} | {{G_E95}} |
| 0.99 | {{D_E99}} | {{D_AR99}} | {{R_E99}} | {{R_AR99}} | {{G_E99}} |

## C.10 LoRA Rank Ablation

| LoRA rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## C.11 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## C.12 Training Loss Curves

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

## C.13 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS}},
    "ratio": {{D_RATIO}},
    "accuracy_pct": {{D_ACC}},
    "f1": {{D_F1}},
    "compressed_layers": {{D_COMP}},
    "avg_rank": {{D_AVG_RANK}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

---