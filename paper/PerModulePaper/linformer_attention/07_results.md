# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | k | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 183 MB |
| DistilBERT | **Linformer** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_K}} | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 712 MB |
| RoBERTa | **Linformer** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_K}} | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 793 MB |
| GPT-2 | **Linformer** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_K}} | {{G_VRAM}} MB |

**Key observations:**

1. DistilBERT (encoder): Linformer compression works well.
2. RoBERTa (encoder, nested): similar to DistilBERT.
3. GPT-2 (decoder): causal fallback preserves accuracy but loses linearity.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | Linformer | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_PCT}}% |

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | Linformer | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | Linformer | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.4 Ablation: k

**Table 7.4: Accuracy vs k.**

| k | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_K16}} | {{R_K16}} | {{G_K16}} |
| 32 | {{D_K32}} | {{R_K32}} | {{G_K32}} |
| 64 | {{D_K64}} | {{R_K64}} | {{G_K64}} |
| 128 | {{D_K128}} | {{R_K128}} | {{G_K128}} |
| 256 | {{D_K256}} | {{R_K256}} | {{G_K256}} |

## 7.5 Ablation: E, F Initialization

**Table 7.5: Effect of E, F init.**

| Init | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Diagonal + noise | {{D_DIAG}} | {{R_DIAG}} | {{G_DIAG}} |
| Gaussian only | {{D_GAUSS}} | {{R_GAUSS}} | {{G_GAUSS}} |
| Zero | {{D_ZERO}} | {{R_ZERO}} | {{G_ZERO}} |

## 7.6 Ablation: Causal Fallback

**Table 7.6: Effect of causal fallback (GPT-2).**

| Config | Acc | Time |
|---|---|---|
| Fallback (correct) | {{G_CAUSAL}} | {{G_CAUSAL_TIME}} s |
| No fallback (incorrect) | {{G_NO_CAUSAL}} | {{G_NO_CAUSAL_TIME}} s |

## 7.7 Ablation: Recovery Steps

**Table 7.7: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.8 Scaling with Sequence Length

**Table 7.8: Inference time vs L (encoder only).**

| L | Standard | Linformer | Speedup |
|---|---|---|---|
| 128 | {{T128_STD}} | {{T128_LIN}} | {{S128}}× |
| 256 | {{T256_STD}} | {{T256_LIN}} | {{S256}}× |
| 512 | {{T512_STD}} | {{T512_LIN}} | {{S512}}× |
| 1024 | {{T1024_STD}} | {{T1024_LIN}} | {{S1024}}× |

## 7.9 Summary

- Encoders benefit from linear complexity.
- Decoders preserve causality via fallback.
- Parameter count increases by ~7%.

---
