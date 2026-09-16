# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | Trainable | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 0 | 183 MB |
| DistilBERT | **EC-MoE** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}} | {{D_TRAINABLE}} | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 0 | 712 MB |
| RoBERTa | **EC-MoE** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}} | {{R_TRAINABLE}} | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 0 | 793 MB |
| GPT-2 | **EC-MoE** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}} | {{G_TRAINABLE}} | {{G_VRAM}} MB |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | EC-MoE | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_PCT}}% |

**We state explicitly:** parameter count **increases**.

## 7.3 Expert Utilization

**Table 7.3: Expert utilization (should be perfectly balanced).**

| Expert | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{UTIL_D_E0}}% | {{UTIL_R_E0}}% | {{UTIL_G_E0}}% |
| 1 | {{UTIL_D_E1}}% | {{UTIL_R_E1}}% | {{UTIL_G_E1}}% |
| 2 | {{UTIL_D_E2}}% | {{UTIL_R_E2}}% | {{UTIL_G_E2}}% |
| 3 | {{UTIL_D_E3}}% | {{UTIL_R_E3}}% | {{UTIL_G_E3}}% |
| 4 | {{UTIL_D_E4}}% | {{UTIL_R_E4}}% | {{UTIL_G_E4}}% |
| 5 | {{UTIL_D_E5}}% | {{UTIL_R_E5}}% | {{UTIL_G_E5}}% |
| 6 | {{UTIL_D_E6}}% | {{UTIL_R_E6}}% | {{UTIL_G_E6}}% |
| 7 | {{UTIL_D_E7}}% | {{UTIL_R_E7}}% | {{UTIL_G_E7}}% |
| **Std** | **{{UTIL_D_STD}}** | **{{UTIL_R_STD}}** | **{{UTIL_G_STD}}** |

**Observation.** Utilization is **exactly balanced** (std ≈ 0) — this is a 
direct consequence of expert-choice routing.

## 7.4 Efficiency Profile

**Table 7.4: Inference efficiency.**

| Model | Baseline Time | EC-MoE | VRAM |
|---|---|---|---|
| DistilBERT | 1.90 s | {{D_TIME}} s | {{D_VRAM}} MB |
| RoBERTa | 4.10 s | {{R_TIME}} s | {{R_VRAM}} MB |
| GPT-2 | 4.20 s | {{G_TIME}} s | {{G_VRAM}} MB |

## 7.5 Ablation: Number of Experts

**Table 7.5: Accuracy vs experts.**

| $E$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_E4}} | {{R_E4}} | {{G_E4}} |
| 8 | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| 16 | {{D_E16}} | {{R_E16}} | {{G_E16}} |

## 7.6 Ablation: Capacity Factor

**Table 7.6: Accuracy vs capacity factor.**

| CF | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | {{D_CF05}} | {{R_CF05}} | {{G_CF05}} |
| 1.0 | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| 1.5 | {{D_CF15}} | {{R_CF15}} | {{G_CF15}} |
| 2.0 | {{D_CF20}} | {{R_CF20}} | {{G_CF20}} |

## 7.7 Ablation: MoE Gate Initialization

**Table 7.7: Accuracy vs moe_init_gate.**

| $b_g$ | σ(b_g) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| -8.0 | 0.0003 | {{D_G_M8}} | {{R_G_M8}} | {{G_G_M8}} |
| -4.0 | {{MOE_GATE_VALUE}} | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| -2.0 | 0.12 | {{D_G_M2}} | {{R_G_M2}} | {{G_G_M2}} |
| 0.0 | 0.50 | {{D_G_0}} | {{R_G_0}} | {{G_G_0}} |

## 7.8 Ablation: Zero-Init

**Table 7.8: Effect of zero-init.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With zero-init | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| Without | {{D_NO_ZI}} | {{R_NO_ZI}} | {{G_NO_ZI}} |

## 7.9 Comparison to Token-Choice MoE

**Table 7.9: Expert-choice vs token-choice (BAM-UMoE).**

| Method | DistilBERT | RoBERTa | GPT-2 | Expert balance |
|---|---|---|---|---|
| BAM-UMoE (token-choice) | {{BAM_D}} | {{BAM_R}} | {{BAM_G}} | {{BAM_STD}} |
| **EC-MoE (expert-choice)** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **~0** |

## 7.10 Training Loss

**Figure 7.1** shows recovery loss vs step. {{TRAINING_DESC}}

## 7.11 Gate Evolution

**Figure 7.2** shows the learned gate $\gamma$ over training.

| Step | DistilBERT γ | RoBERTa γ | GPT-2 γ |
|---|---|---|---|
| 0 | {{GAMMA_STEP0}} | {{GAMMA_STEP0}} | {{GAMMA_STEP0}} |
| 10 | {{GAMMA_STEP10}} | {{GAMMA_STEP10}} | {{GAMMA_STEP10}} |
| 25 | {{GAMMA_STEP25}} | {{GAMMA_STEP25}} | {{GAMMA_STEP25}} |
| 50 | {{GAMMA_STEP50}} | {{GAMMA_STEP50}} | {{GAMMA_STEP50}} |

## 7.12 Summary

- **Accuracy:** {{D_DELTA}}% to {{G_DELTA}}% change.
- **Parameters:** +{{D_PARAM_PCT}}%.
- **Expert utilization:** perfectly balanced (std ≈ 0).
- **Gate:** ramps from {{MOE_GATE_VALUE}} to {{GAMMA_FINAL}}.
```

---