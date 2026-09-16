# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Order | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 183 MB |
| DistilBERT | **PADRe** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_ORDER}} | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 712 MB |
| RoBERTa | **PADRe** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_ORDER}} | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 793 MB |
| GPT-2 | **PADRe** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_ORDER}} | {{G_VRAM}} MB |

## 7.2 Order Ablation

**Table 7.2: Accuracy vs polynomial order.**

| Order | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | {{D_O1}} | {{R_O1}} | {{G_O1}} |
| 2 | {{D_O2}} | {{R_O2}} | {{G_O2}} |
| 4 | {{D_O4}} | {{R_O4}} | {{G_O4}} |
| 8 | {{D_O8}} | {{R_O8}} | {{G_O8}} |

## 7.3 Temperature Ablation

**Table 7.3: Accuracy vs temperature.**

| $T$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | {{D_T05}} | {{R_T05}} | {{G_T05}} |
| 1.0 | {{D_T10}} | {{R_T10}} | {{G_T10}} |
| 2.0 | {{D_T20}} | {{R_T20}} | {{G_T20}} |

## 7.4 Effect of Max-Centering

**Table 7.4: Without vs with centering.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With centering | {{D_CENTER}} | {{R_CENTER}} | {{G_CENTER}} |
| Without centering | {{D_NOCENTER}} | {{R_NOCENTER}} | {{G_NOCENTER}} |

## 7.5 Injection Success

**Table 7.5: Layers successfully injected.**

| Model | Total layers | Injected | Skipped |
|---|---|---|---|
| DistilBERT | 6 | {{D_INJ}} | {{D_SKIP}} |
| RoBERTa | 12 | {{R_INJ}} | {{R_SKIP}} |
| GPT-2 | 12 | {{G_INJ}} | {{G_SKIP}} |

## 7.6 Output Equivalence

**Table 7.6: Output equivalence error per model.**

| Model | Mean error | Max error | Threshold |
|---|---|---|---|
| DistilBERT | {{D_EQ_MEAN}} | {{D_EQ_MAX}} | {{MAX_ERROR}} |
| RoBERTa | {{R_EQ_MEAN}} | {{R_EQ_MAX}} | {{MAX_ERROR}} |
| GPT-2 | {{G_EQ_MEAN}} | {{G_EQ_MAX}} | {{MAX_ERROR}} |

## 7.7 Efficiency

**Table 7.7: Inference time and VRAM.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | PADRe | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | PADRe | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.8 Recovery Steps

**Table 7.8: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.9 Summary

- PADRe maintains accuracy within {{D_DELTA}}% on DistilBERT.
- Order 2 is a good default.
- Max-centering is essential.
- Kernelization not yet implemented; runtime unchanged.
```

---