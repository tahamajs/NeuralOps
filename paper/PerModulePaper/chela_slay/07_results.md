# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | Trainable | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 0 | 183 MB |
| DistilBERT | **CHELA-SLAY** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}} | {{D_TRAINABLE}} | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 0 | 712 MB |
| RoBERTa | **CHELA-SLAY** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}} | {{R_TRAINABLE}} | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 0 | 793 MB |
| GPT-2 | **CHELA-SLAY** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}} | {{G_TRAINABLE}} | {{G_VRAM}} MB |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | CHELA-SLAY | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_PCT}}% |

## 7.3 Inference Time

**Table 7.3: Inference time (seconds).**

| Model | Baseline | CHELA-SLAY | Ratio |
|---|---|---|---|
| DistilBERT (L=128) | 1.90 | {{D_TIME}} | **{{D_TIME_RATIO}}×** |
| RoBERTa (L=128) | 4.10 | {{R_TIME}} | **{{R_TIME_RATIO}}×** |
| GPT-2 (L=128) | 4.20 | {{G_TIME}} | **{{G_TIME_RATIO}}×** |

**Note.** CHELA-SLAY is **slower** because both streams are computed.

## 7.4 Peak VRAM

**Table 7.4: Peak VRAM (MB).**

| Model | Baseline | CHELA-SLAY | Ratio |
|---|---|---|---|
| DistilBERT | 183 | {{D_VRAM}} | {{D_VRAM_RATIO}}× |
| RoBERTa | 712 | {{R_VRAM}} | {{R_VRAM_RATIO}}× |
| GPT-2 | 793 | {{G_VRAM}} | {{G_VRAM_RATIO}}× |

**The causal cumsum is the major memory sink:** see Section 4.7 for the 
analytic formula.

## 7.5 Ablation: Feature Dim $F$

**Table 7.5: Accuracy vs feature dimension.**

| $F$ | DistilBERT | RoBERTa | GPT-2 | Cumsum memory (MB) |
|---|---|---|---|---|
| 32 | {{D_F32}} | {{R_F32}} | {{G_F32}} | {{CMEM_F32}} |
| 64 | {{D_F64}} | {{R_F64}} | {{G_F64}} | {{CMEM_F64}} |
| 128 | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | {{CMEM_F128}} |
| 256 | {{D_F256}} | {{R_F256}} | {{G_F256}} | {{CMEM_F256}} |

## 7.6 Ablation: Gate Bias

**Table 7.6: Accuracy vs gate initialization bias.**

| $b_g$ | σ(b_g) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.0 | 0.50 | {{D_GB0}} | {{R_GB0}} | {{G_GB0}} |
| 2.0 | 0.88 | {{D_GB2}} | {{R_GB2}} | {{G_GB2}} |
| **5.0** | **{{GATE_INIT}}** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| 10.0 | ~1.0 | {{D_GB10}} | {{R_GB10}} | {{G_GB10}} |

## 7.7 Ablation: Dense-Only vs Linear-Only vs Hybrid

**Table 7.7: Stream contribution.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Dense only | {{D_DENSE_ONLY}} | {{R_DENSE_ONLY}} | {{G_DENSE_ONLY}} |
| Linear only | {{D_LINEAR_ONLY}} | {{R_LINEAR_ONLY}} | {{G_LINEAR_ONLY}} |
| **Hybrid** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |

## 7.8 Ablation: Spherical Normalization

**Table 7.8: Effect of spherical normalization on the linear stream.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With spherical | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| Without | {{D_NO_SPH}} | {{R_NO_SPH}} | {{G_NO_SPH}} |

## 7.9 Ablation: Yat-Kernel vs ELU+1

**Table 7.9: Feature map comparison.**

| Feature map | Dim | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| Yat-Kernel | $2F = {{FEATURE_DIM_2}} $ | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| ELU+1 | $d_h$ | {{D_ELU}} | {{R_ELU}} | {{G_ELU}} |

## 7.10 Gate Statistics

**Figure 7.1** shows the learned gate distribution.

| Model | Mean g | Std g |
|---|---|---|
| DistilBERT | {{D_GATE_MEAN}} | {{D_GATE_STD}} |
| RoBERTa | {{R_GATE_MEAN}} | {{R_GATE_STD}} |
| GPT-2 | {{G_GATE_MEAN}} | {{G_GATE_STD}} |

**Observation.** The gate remains near 1 (dense-dominant) even after training, 
suggesting the model does not heavily rely on the linear stream for 
$L = 128$.

## 7.11 Sequence-Length Scaling

**Table 7.11: Inference time vs L (GPT-2).**

| L | Baseline | CHELA-SLAY | Speedup |
|---|---|---|---|
| 128 | {{B128}} s | {{C128}} s | {{S128}}× |
| 256 | {{B256}} s | {{C256}} s | {{S256}}× |
| 512 | {{B512}} s | {{C512}} s | {{S512}}× |
| 1024 | {{B1024}} s | {{C1024}} s | {{S1024}}× |
| 2048 | {{B2048}} s | {{C2048}} s | {{S2048}}× |

**Breakeven:** $L^* = {{BREAKEVEN}}$.

## 7.12 Summary

- **Accuracy:** {{D_DELTA}}% to {{G_DELTA}}% change.
- **Parameters:** +{{D_PARAM_PCT}}%.
- **Speed:** slower at L=128 (both streams computed).
- **Memory:** cumsum dominates peak VRAM.
- **Gate:** learned near-dense (mean {{D_GATE_MEAN}}).

---