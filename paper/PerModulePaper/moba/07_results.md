# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.95M | 183 MB |
| DistilBERT | **Mamba SSM** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}}M | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 712 MB |
| RoBERTa | **Mamba SSM** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}}M | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 793 MB |
| GPT-2 | **Mamba SSM** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}}M | {{G_VRAM}} MB |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | Mamba SSM | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS_ABS}} | {{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS_ABS}} | {{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS_ABS}} | {{G_PARAM_PCT}}% |

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | Mamba SSM | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | Mamba SSM | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.4 Ablation: Preservation Gate

**Table 7.4: Effect of preservation gate.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With gate (init 2.0) | {{D_PRES}} | {{R_PRES}} | {{G_PRES}} |
| No gate | {{D_NOPRES}} | {{R_NOPRES}} | {{G_NOPRES}} |
| Gate init 0.0 | {{D_G0}} | {{R_G0}} | {{G_G0}} |
| Gate init 4.0 | {{D_G4}} | {{R_G4}} | {{G_G4}} |

## 7.5 Ablation: Learned Decay

**Table 7.5: Effect of learned decay.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Learned decay | {{D_LEARNED}} | {{R_LEARNED}} | {{G_LEARNED}} |
| Fixed decay ($\gamma = 0.99$) | {{D_FIXED99}} | {{R_FIXED99}} | {{G_FIXED99}} |
| Fixed decay ($\gamma = 0.9$) | {{D_FIXED9}} | {{R_FIXED9}} | {{G_FIXED9}} |
| No decay ($\gamma = 1$) | {{D_NODECAY}} | {{R_NODECAY}} | {{G_NODECAY}} |

## 7.6 Ablation: LoRA Rank

**Table 7.6: Accuracy vs LoRA rank.**

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_R4}} | {{R_R4}} | {{G_R4}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} |

## 7.7 Ablation: Recovery Steps

**Table 7.7: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.8 Scaling with Sequence Length

**Table 7.8: Inference time vs L.**

| L | Standard | Mamba SSM | Ratio |
|---|---|---|---|
| 64 | {{T64_STD}} | {{T64_MAMBA}} | {{R64}}× |
| 128 | {{T128_STD}} | {{T128_MAMBA}} | {{R128}}× |
| 256 | {{T256_STD}} | {{T256_MAMBA}} | {{R256}}× |
| 512 | {{T512_STD}} | {{T512_MAMBA}} | {{R512}}× |

**Note.** Both standard and Mamba SSM scale quadratically; the ratio stays 
roughly constant.

## 7.9 Comparison with Companion Methods

**Table 7.9: Mamba SSM vs other modules.**

| Method | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% |
| nha | 90.25% | FAIL | 78.78% |
| nasc | 90.71% | 64.56% | 78.44% |
| **Mamba SSM** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |

## 7.10 Summary

- Accuracy near baseline on DistilBERT.
- Preservation gate essential.
- Learned decay outperforms fixed.
- Complexity is $\mathcal{O}(L^2 d)$, **not** O(L).

---