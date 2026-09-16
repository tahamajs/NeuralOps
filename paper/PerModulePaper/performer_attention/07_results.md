<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Accuracy on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Params Δ | Time | Status |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 1.9 s | PASS |
| DistilBERT | **Performer** | **{{D_ACC}}** | **{{D_DELTA}}** | {{D_F1}} | +{{D_PARAM_PCT}}% | {{D_TIME}} | {{D_STATUS}} |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 4.1 s | PASS |
| RoBERTa | **Performer** | **{{R_ACC}}** | **{{R_DELTA}}** | {{R_F1}} | +{{R_PARAM_PCT}}% | {{R_TIME}} | {{R_STATUS}} |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 4.2 s | PASS |
| GPT-2 | **Performer** | **{{G_ACC}}** | **{{G_DELTA}}** | {{G_F1}} | +{{G_PARAM_PCT}}% | {{G_TIME}} | {{G_STATUS}} |

**Key observations:**

1. DistilBERT: {{D_OBS}}
2. RoBERTa: {{R_OBS}}
3. GPT-2: {{G_OBS}}

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | Performer | Δ Abs | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_ABS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_ABS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_ABS}} | +{{G_PARAM_PCT}}% |

**Non-LoRA additions:** $\omega$ ($F \times d$), `output_scale` (4 params).

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Time (base) | Time (Performer) | Ratio | VRAM (base) | VRAM (Performer) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× | 183 MB | {{D_VRAM}} MB |
| RoBERTa | 4.1 s | {{R_TIME}} s | {{R_TIME_R}}× | 712 MB | {{R_VRAM}} MB |
| GPT-2 | 4.2 s | {{G_TIME}} s | {{G_TIME_R}}× | 793 MB | {{G_VRAM}} MB |

## 7.4 Ablation: Number of Features

**Table 7.4: Accuracy vs $F$.**

| $F$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_F32}} | {{R_F32}} | {{G_F32}} |
| 64 | {{D_F64}} | {{R_F64}} | {{G_F64}} |
| 128 | {{D_F128}} | {{R_F128}} | {{G_F128}} |
| 256 | {{D_F256}} | {{R_F256}} | {{G_F256}} |

**Observation.** {{FEATURES_OBS}}

## 7.5 Ablation: Kernel Type

**Table 7.5: FAVOR+ vs elu vs relu.**

| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| `favor_plus` | {{D_FAVOR}} | {{R_FAVOR}} | {{G_FAVOR}} |
| `elu` | {{D_ELU}} | {{R_ELU}} | {{G_ELU}} |
| `relu` | {{D_RELU}} | {{R_RELU}} | {{G_RELU}} |

**Observation.** {{KERNEL_OBS}}

## 7.6 Ablation: LoRA Rank

**Table 7.6: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_R4}} | {{R_R4}} | {{G_R4}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |

## 7.7 Ablation: Recovery Steps

**Table 7.7: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.8 Comparison to Prior Modules

**Table 7.8: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| cosformer | {{COS_D}} | {{COS_R}} | {{COS_G}} | +5.7% |
| orion | {{ORION_D}} | {{ORION_R}} | {{ORION_G}} | **-37%** |
| a3 | {{A3_D}} | {{A3_R}} | {{A3_G}} | **-60%** |
| **performer** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **+{{PARAM_PCT}}%** |

## 7.9 Training Loss Curves

**Figure 7.1** shows recovery loss.

## 7.10 Summary

- **DistilBERT:** {{D_ACC}}% (Δ = {{D_DELTA}})
- **RoBERTa:** {{R_ACC}}% (Δ = {{R_DELTA}})
- **GPT-2:** {{G_ACC}}% (Δ = {{G_DELTA}})
```

---