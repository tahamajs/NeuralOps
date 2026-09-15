# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Attn Params | Ratio |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 2.36M | 1.00× |
| DistilBERT | **VSpec++** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_ATTN}} | **{{D_RATIO}}×** |
| RoBERTa | full | 78.56% | — | 0.8124 | 2.36M | 1.00× |
| RoBERTa | **VSpec++** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_ATTN}} | **{{R_RATIO}}×** |
| GPT-2 | full | 79.36% | — | 0.7978 | 2.36M | 1.00× |
| GPT-2 | **VSpec++** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_ATTN}} | **{{G_RATIO}}×** |

## 7.2 Parameter Analysis

**Table 7.2: Total parameters.**

| Model | Baseline | Ours | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | {{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | {{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | {{G_PARAM_PCT}}% |

## 7.3 Efficiency

**Table 7.3: Inference time and VRAM.**

| Model | Baseline Time | Ours | VRAM |
|---|---|---|---|
| DistilBERT | 1.90 s | {{D_TIME}} s | {{D_VRAM}} MB |
| RoBERTa | 4.10 s | {{R_TIME}} s | {{R_VRAM}} MB |
| GPT-2 | 4.20 s | {{G_TIME}} s | {{G_VRAM}} MB |

## 7.4 Ablation: Token Router

**Table 7.4: Effect of token routing.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With router (ρ=0.75) | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| No router (full L) | {{D_NOROUTE}} | {{R_NOROUTE}} | {{G_NOROUTE}} |
| Router ρ=0.5 | {{D_RHO50}} | {{R_RHO50}} | {{G_RHO50}} |

## 7.5 Ablation: Spectral Gate

**Table 7.5: Effect of the variational spectral gate.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With spectral gate | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| Without (g=1) | {{D_NOGATE}} | {{R_NOGATE}} | {{G_NOGATE}} |
| Deterministic gate (no ε) | {{D_DET}} | {{R_DET}} | {{G_DET}} |

## 7.6 Ablation: Head Gate

**Table 7.6: Effect of the head gate.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With head gate | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| Without (h=1) | {{D_NOHEAD}} | {{R_NOHEAD}} | {{G_NOHEAD}} |

## 7.7 Ablation: SVD Rank

**Table 7.7: Accuracy vs SVD rank.**

| Rank | DistilBERT | Ratio | RoBERTa | Ratio |
|---|---|---|---|---|
| 32 | {{D_R32}} | {{D_R32_RATIO}}× | {{R_R32}} | {{R_R32_RATIO}}× |
| 64 | {{D_R64}} | {{D_R64_RATIO}}× | {{R_R64}} | {{R_R64_RATIO}}× |
| 128 | {{D_ACC}} | {{D_RATIO}}× | {{R_ACC}} | {{R_RATIO}}× |
| 192 | {{D_R192}} | {{D_R192_RATIO}}× | {{R_R192}} | {{R_R192_RATIO}}× |
| 256 | {{D_R256}} | {{D_R256_RATIO}}× | {{R_R256}} | {{R_R256_RATIO}}× |

## 7.8 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA rank.**

| LoRA rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## 7.9 Ablation: Recovery Steps

**Table 7.9: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.10 Gate Statistics

**Table 7.10: Mean gate values after training.**

| Model | Spectral gate $g$ | Head gate $h$ | Router retention |
|---|---|---|---|
| DistilBERT | {{D_G_MEAN}} | {{D_H_MEAN}} | {{D_RHO_ACTUAL}} |
| RoBERTa | {{R_G_MEAN}} | {{R_H_MEAN}} | {{R_RHO_ACTUAL}} |
| GPT-2 | {{G_G_MEAN}} | {{G_H_MEAN}} | {{G_RHO_ACTUAL}} |

## 7.11 Summary

- **Compression:** attention parameters reduce by {{D_RATIO}}×.
- **Accuracy:** drops {{D_DELTA}}%–{{G_DELTA}}%.
- **All three gates contribute** to accuracy.
- **Variational sampling helps** by {{VAR_GAIN}}% over deterministic.

---
