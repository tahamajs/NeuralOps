# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | Δ Params | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.95M | — | 183 MB |
| DistilBERT | **REAP-MoE** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}} | {{D_PARAM_CHANGE}} | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | — | 712 MB |
| RoBERTa | **REAP-MoE** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}} | {{R_PARAM_CHANGE}} | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | — | 793 MB |
| GPT-2 | **REAP-MoE** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}} | {{G_PARAM_CHANGE}} | {{G_VRAM}} MB |

**Observations:**

1. **Accuracy:** {{D_ACC}}% on DistilBERT, {{R_ACC}}% on RoBERTa, {{G_ACC}}% on GPT-2.
2. **Parameter change:** {{D_PARAM_CHANGE}} on DistilBERT (honest reporting).
3. **VRAM:** {{D_VRAM}} MB.

## 7.2 Parameter Analysis

**Table 7.2: Per-layer parameter breakdown (DistilBERT).**

| Component | Original | REAP-MoE |
|---|---|---|
| Attention | 2.36M | 2.36M (unchanged) |
| FFN (removed) | 2.36M | 0 (replaced) |
| MoE experts (8) | 0 | ~7.1M |
| Router | 0 | ~5.9K |
| LayerNorm (2) | 0 | 3K |
| **Total** | **4.72M** | **~9.47M** |
| **Ratio** | 1.00× | **2.01×** |

## 7.3 Expert Utilization

**Table 7.3: Expert routing distribution.**

| Model | Expert 0 | Expert 1 | Expert 2 | Expert 3 | Max/Min |
|---|---|---|---|---|---|
| DistilBERT | {{D_E0}}% | {{D_E1}}% | {{D_E2}}% | {{D_E3}}% | {{D_MAXMIN}} |
| RoBERTa | {{R_E0}}% | {{R_E1}}% | {{R_E2}}% | {{R_E3}}% | {{R_MAXMIN}} |
| GPT-2 | {{G_E0}}% | {{G_E1}}% | {{G_E2}}% | {{G_E3}}% | {{G_MAXMIN}} |

**Observation.** Utilization is balanced within **{{D_MAXMIN}}×** on DistilBERT.

## 7.4 Efficiency Profile

**Table 7.4: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | REAP-MoE | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |

## 7.5 Ablation: Number of Experts

**Table 7.5: Accuracy vs $E$.**

| $E$ | DistilBERT | Params | Ratio |
|---|---|---|---|
| 2 | {{D_E2_ACC}} | {{D_E2_PARAMS}} | {{D_E2_RATIO}}× |
| 4 | {{D_E4_ACC}} | {{D_E4_PARAMS}} | {{D_E4_RATIO}}× |
| 8 | {{D_E8_ACC}} | {{D_E8_PARAMS}} | {{D_E8_RATIO}}× |
| 16 | {{D_E16_ACC}} | {{D_E16_PARAMS}} | {{D_E16_RATIO}}× |

## 7.6 Ablation: Top-$k$

**Table 7.6: Accuracy vs $k$.**

| $k$ | DistilBERT | FLOPs | Accuracy |
|---|---|---|---|
| 1 | {{D_K1_ACC}} | 1× | {{D_K1_ACC}}% |
| 2 | {{D_K2_ACC}} | 2× | {{D_K2_ACC}}% |
| 4 | {{D_K4_ACC}} | 4× | {{D_K4_ACC}}% |

## 7.7 Ablation: SVD Rank

**Table 7.7: Accuracy vs expert SVD rank.**

| Rank | DistilBERT | Expert params |
|---|---|---|
| $d/8$ | {{D_R8_ACC}} | {{D_R8_PARAMS}} |
| $d/4$ | {{D_R4_ACC}} | {{D_R4_PARAMS}} |
| $d/2$ | {{D_R2_ACC}} | {{D_R2_PARAMS}} |
| $d$ | {{D_RD_ACC}} | {{D_RD_PARAMS}} |

## 7.8 Ablation: Load-Balancing Loss

**Table 7.8: Effect of aux loss weight.**

| $\lambda_\text{aux}$ | DistilBERT | Max/Min utilization |
|---|---|---|
| 0.0 | {{D_L0_ACC}} | {{D_L0_MAXMIN}} |
| 0.01 | {{D_L001_ACC}} | {{D_L001_MAXMIN}} |
| 0.1 | {{D_L01_ACC}} | {{D_L01_MAXMIN}} |
| 1.0 | {{D_L1_ACC}} | {{D_L1_MAXMIN}} |

## 7.9 Ablation: Noise Injection

**Table 7.9: Effect of noise std $\sigma$.**

| $\sigma$ | DistilBERT | Router entropy |
|---|---|---|
| 0.0 | {{D_S0_ACC}} | {{D_S0_ENT}} |
| 0.01 | {{D_S001_ACC}} | {{D_S001_ENT}} |
| 0.05 | {{D_S005_ACC}} | {{D_S005_ENT}} |
| 0.1 | {{D_S01_ACC}} | {{D_S01_ENT}} |

## 7.10 Ablation: Component Contributions

**Table 7.10: Component ablation.**

| Config | DistilBERT | Δ |
|---|---|---|
| Full REAP-MoE | {{D_FULL}} | — |
| No SVD experts | {{D_NOSVD}} | {{D_NOSVD_D}} |
| No aux loss | {{D_NOAUX}} | {{D_NOAUX_D}} |
| No noise | {{D_NONOISE}} | {{D_NONOISE_D}} |

## 7.11 Summary

- **Accuracy:** {{D_ACC}}% on DistilBERT
- **Parameter change:** {{D_PARAM_CHANGE}} (increase due to MoE)
- **Expert balance:** {{D_MAXMIN}}× max/min utilization
- **VRAM:** {{D_VRAM}} MB
```

---