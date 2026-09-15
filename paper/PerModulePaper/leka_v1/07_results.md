# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Attn Params | Ratio |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 2.36M | 1.00× |
| DistilBERT | **LEKAv1** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_ATTN}} | {{D_RATIO}}× |
| RoBERTa | full | 78.56% | — | 0.8124 | 2.36M | 1.00× |
| RoBERTa | **LEKAv1** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_ATTN}} | {{R_RATIO}}× |
| GPT-2 | full | 79.36% | — | 0.7978 | 2.36M | 1.00× |
| GPT-2 | **LEKAv1** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_ATTN}} | {{G_RATIO}}× |

## 7.2 Parameter Analysis

**Table 7.2: Total parameters.**

| Model | Baseline | Ours | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_PCT}}% |

## 7.3 Inference Speed vs Sequence Length

**Table 7.3: Inference time (seconds), DistilBERT.**

| L | Baseline | LEKAv1 | Speedup |
|---|---|---|---|
| 128 | 1.90 | {{D_T128}} | {{S128}}× |
| 256 | 3.50 | {{D_T256}} | {{S256}}× |
| 512 | 12.0 | {{D_T512}} | {{S512}}× |
| 1024 | 45.0 | {{D_T1024}} | {{S1024}}× |
| 2048 | 180.0 | {{D_T2048}} | {{S2048}}× |

**Breakeven:** $L^* = {{BREAKEVEN}}$.

## 7.4 Ablation: Feature Dimension

**Table 7.4: Accuracy vs feature dimension.**

| F | DistilBERT | RoBERTa | GPT-2 | Attn Params |
|---|---|---|---|---|
| 64 | {{D_F64}} | {{R_F64}} | {{G_F64}} | {{D_F64_PARAMS}} |
| 128 | {{D_F128}} | {{R_F128}} | {{G_F128}} | {{D_F128_PARAMS}} |
| 256 | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | {{D_ATTN}} |
| 512 | {{D_F512}} | {{R_F512}} | {{G_F512}} | {{D_F512_PARAMS}} |

## 7.5 Ablation: Preservation Gate

**Table 7.5: Effect of preservation gate.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With gate | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| No gate (α=1) | {{D_NOGATE}} | {{R_NOGATE}} | {{G_NOGATE}} |
| Δ | {{GATE_GAIN_D}} | {{GATE_GAIN_R}} | {{GATE_GAIN_G}} |

## 7.6 Ablation: Head-wise LayerNorm

**Table 7.6: Effect of pre-projection LayerNorm.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With norm | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| No norm | {{D_NONORM}} | {{R_NONORM}} | {{G_NONORM}} |

## 7.7 Ablation: QR Orthogonalization

**Table 7.7: Effect of QR init.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| QR projection | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| Random (no QR) | {{D_NOQR}} | {{R_NOQR}} | {{G_NOQR}} |

## 7.8 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |

## 7.9 Ablation: Recovery Steps

**Table 7.9: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.10 Gate Statistics

**Table 7.10: Final α and output_scale.**

| Model | α final | output_scale |
|---|---|---|
| DistilBERT | {{D_ALPHA}} | {{D_OSCALE}} |
| RoBERTa | {{R_ALPHA}} | {{R_OSCALE}} |
| GPT-2 | {{G_ALPHA}} | {{G_OSCALE}} |

## 7.11 Summary

- **Accuracy:** drops {{D_DELTA}}%–{{G_DELTA}}%.
- **Linear scaling:** LEKAv1 is faster at L > {{BREAKEVEN}}.
- **Preservation gate:** contributes {{GATE_GAIN_D}}%.
- **QR projection:** contributes {{QR_GAIN}}%.
- **Parameter increase:** {{D_PARAM_PCT}}%.

---
