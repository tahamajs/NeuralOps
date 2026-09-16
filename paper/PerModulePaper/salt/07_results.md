<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy.**

| Model | Module | Acc | Δ | F1 | Params Δ | Storage |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 134 MB |
| DistilBERT | **salt** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | **{{D_PARAM_RED}}×** | {{D_STORAGE}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 249 MB |
| RoBERTa | **salt** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | **{{R_PARAM_RED}}×** | {{R_STORAGE}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 249 MB |
| GPT-2 | **salt** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | **{{G_PARAM_RED}}×** | {{G_STORAGE}} MB |

**Key observations:**

1. **Parameter reduction:** {{D_PARAM_RED}}× (DistilBERT), {{R_PARAM_RED}}× 
   (RoBERTa), {{G_PARAM_RED}}× (GPT-2).
2. **Accuracy preservation:** within {{MAX_DELTA}}%.
3. **Storage reduction:** ~{{STORAGE_REDUCTION}}×.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After SALT | Reduction |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | **{{D_PARAM_RED}}×** |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | **{{R_PARAM_RED}}×** |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | **{{G_PARAM_RED}}×** |

**Per-layer breakdown (768×768 projection, r=16):**

| Component | Params |
|---|---|
| Full W | 589,824 |
| SALT total | 49,936 |
| **Reduction** | **11.8×** |

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Baseline | SALT | Ratio |
|---|---|---|---|
| DistilBERT | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× |
| RoBERTa | 4.1 s | {{R_TIME}} s | {{R_TIME_R}}× |
| GPT-2 | 4.2 s | {{G_TIME}} s | {{G_TIME_R}}× |

**Note.** SALT is **slightly slower** at inference due to full matrix 
reconstruction at each forward pass.

## 7.4 Ablation: Rank

**Table 7.4: Effect of rank.**

| Rank | DistilBERT | Params (per layer) | Reduction |
|---|---|---|---|
| 4 | {{D_R4}} | 12,336 | 47.8× |
| 8 | {{D_R8}} | 24,656 | 23.9× |
| 16 | {{D_R16}} | 49,296 | 12.0× |
| 32 | {{D_R32}} | 98,576 | 5.98× |
| 64 | {{D_R64}} | 197,136 | 2.99× |

## 7.5 Ablation: Gamma

**Table 7.5: Effect of trainable γ.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| γ trainable | {{D_G}} | {{R_G}} | {{G_G}} |
| γ fixed = 1 | {{D_NOG}} | {{R_NOG}} | {{G_NOG}} |
| Δ | {{D_G_DELTA}} | {{R_G_DELTA}} | {{G_G_DELTA}} |

## 7.6 Ablation: Residual AB

**Table 7.6: Effect of trainable residual.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| AB trainable | {{D_AB}} | {{R_AB}} | {{G_AB}} |
| AB = 0 (frozen) | {{D_NOAB}} | {{R_NOAB}} | {{G_NOAB}} |
| Δ | {{D_AB_DELTA}} | {{R_AB_DELTA}} | {{G_AB_DELTA}} |

## 7.7 Ablation: Recovery Steps

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.8 Comparison to Prior Modules

**Table 7.8: Cross-module comparison.**

| Module | DistilBERT | Params Δ | Storage Δ |
|---|---|---|---|
| full | 90.71% | 0% | 0% |
| nha | 90.25% | +0.88% | +0.88% |
| a3 | {{A3_ACC}} | **-80%** | **-80%** |
| **salt** | **{{D_ACC}}** | **-92%** | **-92%** |

## 7.9 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: {{D_LOSS_TRAJ}}
- RoBERTa: {{R_LOSS_TRAJ}}
- GPT-2: {{G_LOSS_TRAJ}}

## 7.10 Summary

- **Parameter reduction: {{D_PARAM_RED}}×–{{G_PARAM_RED}}×** ✓
- **Storage reduction: ~{{STORAGE_REDUCTION}}×** ✓
- **Accuracy preserved within {{MAX_DELTA}}%** ✓
- **Inference memory: unchanged** ✗
- **Inference latency: {{D_TIME_R}}× (slightly slower)** ✗
- Verdict: **{{VERDICT}}**

---