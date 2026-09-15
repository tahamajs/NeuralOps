# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | Trainable |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 0 |
| DistilBERT | **HymbaTransX** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}} | {{D_TRAINABLE}} |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 0 |
| RoBERTa | **HymbaTransX** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}} | {{R_TRAINABLE}} |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 0 |
| GPT-2 | **HymbaTransX** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}} | {{G_TRAINABLE}} |

## 7.2 Parameter Analysis

**Table 7.2: Total parameter counts.**

| Model | Baseline | Ours | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_PCT}}% |

**We state explicitly:** HymbaTransX increases parameter count.

## 7.3 Efficiency Profile

**Table 7.3: Inference time and VRAM.**

| Model | Baseline Time | Ours | Baseline VRAM | Ours |
|---|---|---|---|---|
| DistilBERT | 1.90 s | {{D_TIME}} s | 183 MB | {{D_VRAM}} MB |
| RoBERTa | 4.10 s | {{R_TIME}} s | 712 MB | {{R_VRAM}} MB |
| GPT-2 | 4.20 s | {{G_TIME}} s | 793 MB | {{G_VRAM}} MB |

## 7.4 Ablation: Differential Attention

**Table 7.4: Effect of differential attention.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Standard MHSA (single Q-K) | {{D_MHSA}} | {{R_MHSA}} | {{G_MHSA}} |
| Differential (2 Q-K, λ=0.2) | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| Differential (2 Q-K, λ=0.8) | {{D_L08}} | {{R_L08}} | {{G_L08}} |

## 7.5 Ablation: SSM Fusion

**Table 7.5: Effect of fusion.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Attn only (α=1) | {{D_ATTN_ONLY}} | {{R_ATTN_ONLY}} | {{G_ATTN_ONLY}} |
| SSM only (α=0) | {{D_SSM_ONLY}} | {{R_SSM_ONLY}} | {{G_SSM_ONLY}} |
| Fixed α=0.5 | {{D_FIXED}} | {{R_FIXED}} | {{G_FIXED}} |
| **Learned α** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |

## 7.6 Learned α Values

**Table 7.6: Final α per model.**

| Model | α initial | α final | Interpretation |
|---|---|---|---|
| DistilBERT | 0.27 | {{D_ALPHA}} | {{D_ALPHA_INTERP}} |
| RoBERTa | 0.27 | {{R_ALPHA}} | {{R_ALPHA_INTERP}} |
| GPT-2 | 0.27 | {{G_ALPHA}} | {{G_ALPHA_INTERP}} |

## 7.7 Ablation: LoRA Rank

**Table 7.7: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |

## 7.8 Ablation: Recovery Steps

**Table 7.8: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.9 Training Loss

**Figure 7.1** shows recovery loss. {{TRAINING_DESC}}

## 7.10 Summary

- **Accuracy:** drops {{D_DELTA}}% to {{G_DELTA}}%.
- **Parameters:** increase {{D_PARAM_PCT}}% to {{G_PARAM_PCT}}%.
- **Differential attention** helps by {{DIFF_GAIN}}%.
- **SSM fusion** helps by {{SSM_GAIN}}%.

---
