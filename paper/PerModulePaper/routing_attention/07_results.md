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
| DistilBERT | **routing** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | +{{D_PARAM_PCT}}% | {{D_TIME}} |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 4.1 s |
| RoBERTa | **routing** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | +{{R_PARAM_PCT}}% | {{R_TIME}} |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 4.2 s |
| GPT-2 | **routing** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | +{{G_PARAM_PCT}}% | {{G_TIME}} |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After | Δ | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_ABS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_ABS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_ABS}} | +{{G_PARAM_PCT}}% |

**Note.** Routing Attention increases parameter count via LoRA adapters. 
It is **not** a compression method.

## 7.3 Efficiency Profile

**Table 7.3: Inference time.**

| Model | Baseline | Routing | Ratio |
|---|---|---|---|
| DistilBERT | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× |
| RoBERTa | 4.1 s | {{R_TIME}} s | {{R_TIME_R}}× |
| GPT-2 | 4.2 s | {{G_TIME}} s | {{G_TIME_R}}× |

**Note.** Routing is **slower** than baseline, because the cluster loop 
recomputes $K$ full attention matrices.

## 7.4 Ablation: Number of Clusters

**Table 7.4: Effect of cluster count (DistilBERT).**

| $K$ | Acc | Time |
|---|---|---|
| 4 | {{D_K4}} | {{D_K4_T}} |
| 8 | {{D_K8}} | {{D_K8_T}} |
| 16 | {{D_K16}} | {{D_K16_T}} |
| 32 | {{D_K32}} | {{D_K32_T}} |

**Observation.** {{K_OBS}}

## 7.5 Ablation: k-means Iterations

**Table 7.5: Effect of $n_{\text{iters}}$.**

| $n_{\text{iters}}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | {{D_I1}} | {{R_I1}} | {{G_I1}} |
| 2 | {{D_I2}} | {{R_I2}} | {{G_I2}} |
| 3 | {{D_I3}} | {{R_I3}} | {{G_I3}} |
| 5 | {{D_I5}} | {{R_I5}} | {{G_I5}} |

## 7.6 Ablation: Token-0 Anchor

**Table 7.6: Effect of token-0 anchor.**

| Anchor | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Yes | {{D_ANCHOR}} | {{R_ANCHOR}} | {{G_ANCHOR}} |
| No | {{D_NOANCHOR}} | {{R_NOANCHOR}} | {{G_NOANCHOR}} |

## 7.7 Ablation: Self-Token Preservation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Self-token yes | {{D_SELF}} | {{R_SELF}} | {{G_SELF}} |
| Self-token no | {{D_NOSELF}} | {{R_NOSELF}} | {{G_NOSELF}} |

## 7.8 Ablation: Q+K vs Q only

| Feature | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q+K (ours) | {{D_QK}} | {{R_QK}} | {{G_QK}} |
| Q only | {{D_Q}} | {{R_Q}} | {{G_Q}} |

## 7.9 Comparison to Prior Modules

**Table 7.9: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| nha | 90.25% | FAIL | 78.78% | +0.88% |
| nasc | 90.71% | 64.56% | 78.44% | +0.94% |
| rala | 90.71% | — | 79.36% | +0.90% |
| **routing** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | +{{D_PARAM_PCT}}% |

## 7.10 Loss Curves

**Figure 7.1** shows recovery loss.

## 7.11 Summary

- Accuracy: preserved within {{D_DELTA}}%–{{G_DELTA}}%.
- Parameters: +{{D_PARAM_PCT}}%.
- Inference time: {{D_TIME_R}}× **slower**.
- Verdict: **{{VERDICT}}**

---