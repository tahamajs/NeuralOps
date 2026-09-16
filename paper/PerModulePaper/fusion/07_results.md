# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy (L = 128, short path).**

| Model | Module | Acc | Δ | F1 | Params | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 183 MB |
| DistilBERT | **FUSIONATTN** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}}M | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 712 MB |
| RoBERTa | **FUSIONATTN** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}}M | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 793 MB |
| GPT-2 | **FUSIONATTN** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}}M | {{G_VRAM}} MB |

## 7.2 Parameter Analysis

| Model | Baseline | FUSIONATTN | Δ |
|---|---|---|---|
| DistilBERT | 66.9M | {{D_PARAMS}}M | +{{D_PARAM_DELTA}}M |
| RoBERTa | 124.6M | {{R_PARAMS}}M | +{{R_PARAM_DELTA}}M |
| GPT-2 | 124.4M | {{G_PARAMS}}M | +{{G_PARAM_DELTA}}M |

**FUSIONATTN increases parameter count by ~{{PARAM_CHANGE}}%.**

## 7.3 Attention Path Analysis

**Table 7.3: Path taken at different sequence lengths.**

| $L$ | Path | Note |
|---|---|---|
| 32 | Short | Trivial |
| 64 | Short | |
| **128** | **Short (all models)** | **Test length** |
| 256 | Long | Not tested |
| 512 | Long | Not tested |

**⚠️ Critical observation.** At $L = 128$, all experiments use the 
**short path**. Sparsity is not exercised.

## 7.4 Ablation: Block Size

**Table 7.4 (L=512 only, not tested at L=128).**

| $B$ | Acc | Time |
|---|---|---|
| 32 | {{D_B32}} | {{D_T32}} |
| 64 | {{D_B64}} | {{D_T64}} |
| 128 | {{D_B128}} | {{D_T128}} |

## 7.5 Ablation: Top-K

**Table 7.5 (L=512 only).**

| $K$ | Acc | Time |
|---|---|---|
| 2 | {{D_K2}} | {{D_TK2}} |
| 4 | {{D_K4}} | {{D_TK4}} |
| 8 | {{D_K8}} | {{D_TK8}} |

## 7.6 Ablation: LoRA Rank

| $r_L$ | D acc | R acc | G acc |
|---|---|---|---|
| 0 | {{D_RL0}} | {{R_RL0}} | {{G_RL0}} |
| 4 | {{D_RL4}} | {{R_RL4}} | {{G_RL4}} |
| **16** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| 32 | {{D_RL32}} | {{R_RL32}} | {{G_RL32}} |

## 7.7 Comparison with Companion Methods

| Method | D acc | Params Δ | Sparse? | Path at L=128 |
|---|---|---|---|---|
| full | 90.71% | 0 | No | — |
| `adasplash` | 90.25% | +0.89% | Yes (α-entmax) | — |
| `nha` | 90.25% | +0.89% | No | — |
| **`fusion_attn`** | **{{D_ACC}}** | **+{{D_PARAM_DELTA_PCT}}%** | **Block** | **Short** |

## 7.8 Summary

- **Accuracy:** matches baseline (short path).
- **Parameters:** +{{PARAM_CHANGE}}%.
- **Sparsity:** not exercised at L=128.
- **Long path:** untested.

---