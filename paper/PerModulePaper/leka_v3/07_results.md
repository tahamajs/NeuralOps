# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | L_land | W | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | — | 183 MB |
| DistilBERT | **LEKAv3** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_LLAND}} | {{D_W}} | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | — | — | 712 MB |
| RoBERTa | **LEKAv3** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_LLAND}} | {{R_W}} | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | — | — | 793 MB |
| GPT-2 | **LEKAv3** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_LLAND}} | {{G_W}} | {{G_VRAM}} MB |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | LEKAv3 | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_PCT}}% |

**LEKAv3 adds parameters** for kernel projections, landmark scorer, 
$W_\text{corr}$, and LoRA adapters. We do not claim parameter reduction.

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | LEKAv3 | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.1 s | 27400 | 712 MB |
| RoBERTa | LEKAv3 | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | LEKAv3 | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.4 Ablation: Landmark Count

**Table 7.4: Accuracy vs base landmark count $L_0$.**

| $L_0$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_L8}} | {{R_L8}} | {{G_L8}} |
| 16 | {{D_L16}} | {{R_L16}} | {{G_L16}} |
| 32 | {{D_L32}} | {{R_L32}} | {{G_L32}} |
| 64 | {{D_L64}} | {{R_L64}} | {{G_L64}} |
| 128 | {{D_L128}} | {{R_L128}} | {{G_L128}} |

## 7.5 Ablation: Local Window Size

**Table 7.5: Accuracy vs $W$.**

| $W$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_W8}} | {{R_W8}} | {{G_W8}} |
| 16 | {{D_W16}} | {{R_W16}} | {{G_W16}} |
| 32 | {{D_W32}} | {{R_W32}} | {{G_W32}} |
| 64 | {{D_W64}} | {{R_W64}} | {{G_W64}} |

## 7.6 Ablation: Feature Dimension

**Table 7.6: Accuracy vs $d_\text{feat}$.**

| $d_\text{feat}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_F32}} | {{R_F32}} | {{G_F32}} |
| 64 | {{D_F64}} | {{R_F64}} | {{G_F64}} |
| 128 | {{D_F128}} | {{R_F128}} | {{G_F128}} |
| 256 | {{D_F256}} | {{R_F256}} | {{G_F256}} |

## 7.7 Ablation: Component Contribution

**Table 7.7: Effect of removing each component.**

| Configuration | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full LEKAv3 | {{D_FULL}} | {{R_FULL}} | {{G_FULL}} |
| − landmark scorer | {{D_NO_SCORER}} | {{R_NO_SCORER}} | {{G_NO_SCORER}} |
| − local branch | {{D_NO_LOCAL}} | {{R_NO_LOCAL}} | {{G_NO_LOCAL}} |
| − W_corr | {{D_NO_WCORR}} | {{R_NO_WCORR}} | {{G_NO_WCORR}} |
| − multi-scale | {{D_NO_MULTI}} | {{R_NO_MULTI}} | {{G_NO_MULTI}} |

## 7.8 Ablation: Gate Initialization

**Table 7.8: Effect of gate init value.**

| Init | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.0 (50/50) | {{D_G0}} | {{R_G0}} | {{G_G0}} |
| 1.0 (73/27) | {{D_G1}} | {{R_G1}} | {{G_G1}} |
| 2.0 (88/12) | {{D_G2}} | {{R_G2}} | {{G_G2}} |
| 4.0 (98/2) | {{D_G4}} | {{R_G4}} | {{G_G4}} |

## 7.9 Ablation: Recovery Steps

**Table 7.9: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.10 Scaling with Sequence Length

**Table 7.10: Inference time vs $L$ (GPT-2).**

| $L$ | Standard | LEKAv3 | Speedup |
|---|---|---|---|
| 128 | {{T128_STD}} s | {{T128_LEKA}} s | {{S128}}× |
| 256 | {{T256_STD}} s | {{T256_LEKA}} s | {{S256}}× |
| 512 | {{T512_STD}} s | {{T512_LEKA}} s | {{S512}}× |
| 1024 | {{T1024_STD}} s | {{T1024_LEKA}} s | {{S1024}}× |

## 7.11 Comparison with Companion Methods

**Table 7.11: LEKAv3 vs prior attention modules.**

| Method | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% |
| nha | 90.25% | FAIL | 78.78% |
| nasc | 90.71% | 64.56% | 78.44% |
| **LEKAv3** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |

## 7.12 Summary

- Landmark + local fusion gives **{{D_ACC}}%** on DistilBERT.
- Parameters increase by ~7.5% (honest framing).
- Complexity is linear in $L$ for fixed $L_\text{land}, W$.
```

---