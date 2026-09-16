# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.95M | 183 MB |
| DistilBERT | **AEGIS-V5** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}}M | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 712 MB |
| RoBERTa | **AEGIS-V5** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}}M | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 793 MB |
| GPT-2 | **AEGIS-V5** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}}M | {{G_VRAM}} MB |

**Key observations:**

1. DistilBERT: {{D_OBS}}
2. RoBERTa: {{R_OBS}}
3. GPT-2: {{G_OBS}}

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | AEGIS-V5 | Change | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS_ABS}} | {{D_PARAM_PCT}}% | {{D_NOTE}} |
| RoBERTa | 124,647,170 | {{R_PARAMS_ABS}} | {{R_PARAM_PCT}}% | {{R_NOTE}} |
| GPT-2 | 124,441,344 | {{G_PARAMS_ABS}} | {{G_PARAM_PCT}}% | {{G_NOTE}} |

**Important.** With `max_rank=384` (default), SVD provides no parameter 
reduction. The parameter change is dominated by LoRA + RFF.

## 7.3 SVD Rank Analysis

**Table 7.3: Effective SVD ranks per layer.**

| Model | Layer | Q rank | K rank | V rank | O rank |
|---|---|---|---|---|---|
| DistilBERT | 0 | {{D_L0_Q}} | {{D_L0_K}} | {{D_L0_V}} | {{D_L0_O}} |
| DistilBERT | 3 | {{D_L3_Q}} | {{D_L3_K}} | {{D_L3_V}} | {{D_L3_O}} |
| ... | ... | ... | ... | ... | ... |

## 7.4 Efficiency Profile

**Table 7.4: Inference efficiency.**

| Model | Config | Time | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | AEGIS-V5 | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | AEGIS-V5 | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.5 Ablation: SVD Rank / Energy

**Table 7.5: Accuracy vs max_rank.**

| max_rank | DistilBERT | Params | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 32 | {{D_R32}} | {{D_R32_P}}M | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{D_R64_P}}M | {{R_R64}} | {{G_R64}} |
| 128 | {{D_R128}} | {{D_R128_P}}M | {{R_R128}} | {{G_R128}} |
| 256 | {{D_R256}} | {{D_R256_P}}M | {{R_R256}} | {{G_R256}} |
| 384 | {{D_R384}} | {{D_R384_P}}M | {{R_R384}} | {{G_R384}} |

## 7.6 Ablation: Head Split

**Table 7.6: Accuracy vs local/global head split.**

| Split | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| All local | {{D_ALL_LOCAL}} | {{R_ALL_LOCAL}} | {{G_ALL_LOCAL}} |
| 8/4 (default) | {{D_SPLIT84}} | {{R_SPLIT84}} | {{G_SPLIT84}} |
| 6/6 | {{D_SPLIT66}} | {{R_SPLIT66}} | {{G_SPLIT66}} |
| 4/8 | {{D_SPLIT48}} | {{R_SPLIT48}} | {{G_SPLIT48}} |
| All global | {{D_ALL_GLOBAL}} | {{R_ALL_GLOBAL}} | {{G_ALL_GLOBAL}} |

## 7.7 Ablation: Window Size

**Table 7.7: Accuracy vs local window $W$.**

| $W$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_W16}} | {{R_W16}} | {{G_W16}} |
| 32 | {{D_W32}} | {{R_W32}} | {{G_W32}} |
| 64 | {{D_W64}} | {{R_W64}} | {{G_W64}} |
| 128 | {{D_W128}} | {{R_W128}} | {{G_W128}} |

## 7.8 Ablation: RFF Features

**Table 7.8: Accuracy vs RFF feature dim $F$.**

| $F$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_F32}} | {{R_F32}} | {{G_F32}} |
| 64 | {{D_F64}} | {{R_F64}} | {{G_F64}} |
| 128 | {{D_F128}} | {{R_F128}} | {{G_F128}} |
| 256 | {{D_F256}} | {{R_F256}} | {{G_F256}} |

## 7.9 Ablation: Softpick Fix

**Table 7.9: Impact of softpick fix (V4 vs V5).**

| Version | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| V4 (broken softpick) | {{D_V4}} | {{R_V4}} | {{G_V4}} |
| **V5 (fixed)** | **{{D_V5}}** | **{{R_V5}}** | **{{G_V5}}** |

## 7.10 Ablation: Recovery Steps

**Table 7.10: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.11 Scaling with Sequence Length

**Table 7.11: Inference time vs L.**

| L | Standard (enc) | AEGIS-V5 (enc) | Speedup |
|---|---|---|---|
| 128 | {{T128_STD}} | {{T128_AEG}} | {{S128}}× |
| 256 | {{T256_STD}} | {{T256_AEG}} | {{S256}}× |
| 512 | {{T512_STD}} | {{T512_AEG}} | {{S512}}× |
| 1024 | {{T1024_STD}} | {{T1024_AEG}} | {{S1024}}× |

## 7.12 Summary

- With `max_rank=384`, no parameter reduction (baseline-equivalent).
- With `max_rank=128`, ~{{COMPRESSION}}× compression.
- Decoder fallback preserves accuracy but not complexity.
```

---