# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | Trainable |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 0 |
| DistilBERT | **JambaInfo** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}} | {{D_TRAINABLE}} |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 0 |
| RoBERTa | **JambaInfo** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}} | {{R_TRAINABLE}} |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 0 |
| GPT-2 | **JambaInfo** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}} | {{G_TRAINABLE}} |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | Ours | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_PCT}}% |

## 7.3 Bottleneck Dimension Ablation

**Table 7.3: Accuracy vs bottleneck dim.**

| $b$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 64 | {{D_B64}} | {{R_B64}} | {{G_B64}} |
| 128 | {{D_B128}} | {{R_B128}} | {{G_B128}} |
| 256 | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| 512 | {{D_B512}} | {{R_B512}} | {{G_B512}} |

## 7.4 Ablation: No Bottleneck

**Table 7.4: Effect of the bottleneck.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With bottleneck | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| No bottleneck (full x into SSM) | {{D_NOBN}} | {{R_NOBN}} | {{G_NOBN}} |
| Δ | {{BN_GAIN}} | {{R_BN_GAIN}} | {{G_BN_GAIN}} |

## 7.5 Ablation: α Initialization

**Table 7.5: Accuracy vs α init.**

| w₀ | α₀ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| -4 | 0.018 | {{D_WM4}} | {{R_WM4}} | {{G_WM4}} |
| -2 | 0.12 | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| 0 | 0.5 | {{D_W0}} | {{R_W0}} | {{G_W0}} |
| +2 | 0.88 | {{D_WP2}} | {{R_WP2}} | {{G_WP2}} |

## 7.6 Ablation: Differential Attention

**Table 7.6: Effect of differential attention.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Standard (single Q-K) | {{D_STD}} | {{R_STD}} | {{G_STD}} |
| Differential | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |

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

## 7.9 Learned α

**Table 7.9: Final α values.**

| Model | α₀ | α_final |
|---|---|---|
| DistilBERT | 0.12 | {{D_ALPHA}} |
| RoBERTa | 0.12 | {{R_ALPHA}} |
| GPT-2 | 0.12 | {{G_ALPHA}} |

## 7.10 Summary

- **Accuracy:** drops {{D_DELTA}}% to {{G_DELTA}}%.
- **Bottleneck helps** by {{BN_GAIN}}%.
- **Differential helps** by {{DIFF_GAIN}}%.
- **Parameters:** increase {{D_PARAM_PCT}}%.
```

---