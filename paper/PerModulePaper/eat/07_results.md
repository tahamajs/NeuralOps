<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Compressed | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 183 MB | 1.9 s |
| DistilBERT | **EAT** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_COMP}} | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 712 MB | 4.1 s |
| RoBERTa | **EAT** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_COMP}} | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 793 MB | 4.2 s |
| GPT-2 | **EAT** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_COMP}} | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After EAT | Δ | Δ % |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_DELTA}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_DELTA}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_DELTA}} | +{{G_PARAM_PCT}}% |

**Interpretation.** EAT **does not reduce** parameters. The increase 
corresponds exactly to LoRA adapters added to four attention projections 
per layer.

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | EAT | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | EAT | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | EAT | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.4 Ablation: LoRA Rank

**Table 7.4: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 | Param increase |
|---|---|---|---|---|
| 4 | {{D_R4}} | {{R_R4}} | {{G_R4}} | +{{D_R4_PARAM}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} | +{{D_R8_PARAM}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} | +{{D_R16_PARAM}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} | +{{D_R32_PARAM}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} | +{{D_R64_PARAM}} |

## 7.5 Ablation: Recovery Steps

**Table 7.5: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.6 Ablation: Pruning

**Table 7.6: Effect of progressive token pruning at inference.**

| Config | DistilBERT | Inference time | RoBERTa | Inference time |
|---|---|---|---|---|
| EAT without pruning | {{D_NOPRUNE}} | {{D_NOPRUNE_TIME}} s | {{R_NOPRUNE}} | {{R_NOPRUNE_TIME}} s |
| EAT with pruning (r=0.5) | {{D_PRUNE50}} | {{D_PRUNE50_TIME}} s | {{R_PRUNE50}} | {{R_PRUNE50_TIME}} s |
| EAT with pruning (r=0.7) | {{D_PRUNE70}} | {{D_PRUNE70_TIME}} s | {{R_PRUNE70}} | {{R_PRUNE70_TIME}} s |

**Observation.** Pruning reduces inference time at the cost of accuracy. 
The optimal pruning ratio depends on the task.

## 7.7 Comparison with Companion Methods

**Table 7.7: EAT vs prior modules.**

| Method | DistilBERT | RoBERTa | GPT-2 | Params |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 1.00× |
| nha | 90.25% | FAIL | 78.78% | 1.01× |
| a3 (r=64) | {{A3_D}} | {{A3_R}} | {{A3_G}} | 0.20× |
| aasvd (r=288) | {{AASVD_D}} | {{AASVD_R}} | {{AASVD_G}} | 0.75× |
| **EAT** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **1.01×** |

## 7.8 Failure Cases

**Table 7.8: Modules that failed.**

| Model | Status | Accuracy | Reason |
|---|---|---|---|
| DistilBERT | {{D_STATUS}} | {{D_ACC}} | {{D_REASON}} |
| RoBERTa | {{R_STATUS}} | {{R_ACC}} | {{R_REASON}} |
| GPT-2 | {{G_STATUS}} | {{G_ACC}} | {{G_REASON}} |

## 7.9 Summary

| Metric | Value |
|---|---|
| Accuracy drop (best) | {{BEST_DELTA}}% |
| Parameter increase | +{{PARAM_CHANGE}}% |
| Inference overhead | {{INFERENCE_RATIO}}× |
| Pruning speedup (r=0.5) | {{PRUNE_SPEEDUP}}× |

---