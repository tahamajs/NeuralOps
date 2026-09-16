<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy.**

| Model | Module | Acc | Δ | F1 | Params Δ | Time |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 1.9 s |
| DistilBERT | **RALA** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | +{{D_PARAM_PCT}}% | {{D_TIME}} |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 4.1 s |
| RoBERTa | **RALA** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | +{{R_PARAM_PCT}}% | {{R_TIME}} |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 4.2 s |
| GPT-2 | **RALA** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | +{{G_PARAM_PCT}}% | {{G_TIME}} |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After | Δ | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_ABS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_ABS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_ABS}} | +{{G_PARAM_PCT}}% |

**Note.** RALA is **not** a compression method. Parameter count increases 
because of LoRA adapters and the augmentation matrices.

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Baseline | RALA | Ratio |
|---|---|---|---|
| DistilBERT | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× |
| RoBERTa | 4.1 s | {{R_TIME}} s | {{R_TIME_R}}× |
| GPT-2 | 4.2 s | {{G_TIME}} s | {{G_TIME_R}}× |

**Note.** The slowdown is dominated by the sequential causal prefix scan.

## 7.4 Ablation: Augmentation Rank

**Table 7.4: Effect of augmentation rank $r$ (DistilBERT).**

| Rank | Acc | Extra params |
|---|---|---|
| 1 | {{D_R1}} | 6,150 |
| 2 | {{D_R2}} | 12,294 |
| 4 | {{D_R4}} | 24,582 |
| 8 | {{D_R8}} | 49,158 |
| 16 | {{D_R16}} | 98,310 |

## 7.5 Ablation: Alpha Init

**Table 7.5: Effect of $\alpha$ initialization.**

| $\alpha_0$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.0 | {{D_A0}} | {{R_A0}} | {{G_A0}} |
| 0.1 | {{D_A1}} | {{R_A1}} | {{G_A1}} |
| 1.0 | {{D_A10}} | {{R_A10}} | {{G_A10}} |

## 7.6 Ablation: Kernel Choice

**Table 7.6: Effect of kernel feature map.**

| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| softplus | {{D_SOFT}} | {{R_SOFT}} | {{G_SOFT}} |
| elu+1 | {{D_ELU}} | {{R_ELU}} | {{G_ELU}} |
| ReLU | {{D_RELU}} | {{R_RELU}} | {{G_RELU}} |

## 7.7 Ablation: Recovery Steps

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.8 Comparison to Prior Modules

**Table 7.8: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| nha | 90.25% | FAIL | 78.78% | +0.88% |
| nasc | 90.71% | 64.56% | 78.44% | +0.94% |
| **rala** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | +{{D_PARAM_PCT}}% |

## 7.9 Loss Curves

**Figure 7.1** shows recovery loss for RALA.

- DistilBERT: {{D_LOSS_TRAJ}}
- RoBERTa: {{R_LOSS_TRAJ}}
- GPT-2: {{G_LOSS_TRAJ}}

## 7.10 Summary

- RALA is a **linear attention replacement**, not a compressor.
- Accuracy drop: {{D_DELTA}}%, {{R_DELTA}}%, {{G_DELTA}}%.
- Parameter increase: +{{D_PARAM_PCT}}%.
- Inference time: {{D_TIME_R}}× (due to sequential prefix scan).
- Verdict: **{{VERDICT}}**
```

---