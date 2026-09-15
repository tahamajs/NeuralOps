# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Attn Params | Landmark Ratio |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 2.36M | — |
| DistilBERT | **LEKAv2** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_ATTN}} | {{D_LR}} |
| RoBERTa | full | 78.56% | — | 0.8124 | 2.36M | — |
| RoBERTa | **LEKAv2** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_ATTN}} | {{R_LR}} |
| GPT-2 | full | 79.36% | — | 0.7978 | 2.36M | — |
| GPT-2 | **LEKAv2** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_ATTN}} | {{G_LR}} |

## 7.2 Parameter Analysis

**Table 7.2: Total parameter counts.**

| Model | Baseline | Ours | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_PCT}}% |

## 7.3 Inference Time vs Sequence Length

**Table 7.3: Inference time (seconds), DistilBERT.**

| L | Baseline | LEKAv2 | Speedup |
|---|---|---|---|
| 128 | 1.90 | {{D_T128}} | {{S128}}× |
| 256 | 3.50 | {{D_T256}} | {{S256}}× |
| 512 | 12.0 | {{D_T512}} | {{S512}}× |
| 1024 | 45.0 | {{D_T1024}} | {{S1024}}× |
| 2048 | 180.0 | {{D_T2048}} | {{S2048}}× |

## 7.4 Ablation: Landmark Ratio ρ

**Table 7.4: Accuracy vs ρ.**

| ρ | DistilBERT | Landmark count | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.25 | {{D_RHO25}} | {{D_RHO25_L}} | {{R_RHO25}} | {{G_RHO25}} |
| 0.50 | **{{D_ACC}}** | {{D_LR}} | **{{R_ACC}}** | **{{G_ACC}}** |
| 0.75 | {{D_RHO75}} | {{D_RHO75_L}} | {{R_RHO75}} | {{G_RHO75}} |
| 1.00 | {{D_RHO100}} | 128 | {{R_RHO100}} | {{G_RHO100}} |

## 7.5 Ablation: Feature Dim F

**Table 7.5: Accuracy vs feature dimension.**

| F | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 64 | {{D_F64}} | {{R_F64}} | {{G_F64}} |
| 128 | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| 256 | {{D_F256}} | {{R_F256}} | {{G_F256}} |
| 512 | {{D_F512}} | {{R_F512}} | {{G_F512}} |

## 7.6 Ablation: Landmark Distillation

**Table 7.6: Effect of learned landmark selection.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With scorer | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| Random landmarks | {{D_RAND}} | {{R_RAND}} | {{G_RAND}} |
| No landmarks (full) | {{D_FULL}} | {{R_FULL}} | {{G_FULL}} |
| Δ from learned | {{LANDMARK_GAIN}} | {{R_LANDMARK_GAIN}} | {{G_LANDMARK_GAIN}} |

## 7.7 Ablation: Local Window

**Table 7.7: Effect of local window.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With local (W={{LOCAL_WINDOW}}) | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| No local (global only) | {{D_NOLOCAL}} | {{R_NOLOCAL}} | {{G_NOLOCAL}} |
| W=32 | {{D_W32}} | {{R_W32}} | {{G_W32}} |
| W=256 | {{D_W256}} | {{R_W256}} | {{G_W256}} |

## 7.8 Ablation: Softpick vs Softmax

**Table 7.8: Effect of softpick.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Softpick | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| Standard softmax | {{D_SM}} | {{R_SM}} | {{G_SM}} |

## 7.9 Ablation: Dual-Scale RFF

**Table 7.9: Effect of dual-scale.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Dual-scale (F + F/2) | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| Single-scale (F) | {{D_SINGLE}} | {{R_SINGLE}} | {{G_SINGLE}} |

## 7.10 Ablation: LoRA Rank

**Table 7.10: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## 7.11 Ablation: Recovery Steps

**Table 7.11: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.12 Gate Statistics

**Table 7.12: Local gate values after training.**

| Model | Mean $g_h$ | Std | Min | Max |
|---|---|---|---|---|
| DistilBERT | {{D_G_MEAN}} | {{D_G_STD}} | {{D_G_MIN}} | {{D_G_MAX}} |
| RoBERTa | {{R_G_MEAN}} | {{R_G_STD}} | {{R_G_MIN}} | {{R_G_MAX}} |
| GPT-2 | {{G_G_MEAN}} | {{G_G_STD}} | {{G_G_MIN}} | {{G_G_MAX}} |

## 7.13 Summary

- **Accuracy:** drops {{D_DELTA}}%–{{G_DELTA}}%.
- **Landmark distillation:** contributes {{LANDMARK_GAIN}}%.
- **Local window:** contributes {{LOCAL_GAIN}}%.
- **Dual-scale:** contributes {{DUAL_GAIN}}%.
- **Softpick:** contributes {{SOFTPICK_GAIN}}%.
- **Parameters:** increase {{D_PARAM_PCT}}%.

---
