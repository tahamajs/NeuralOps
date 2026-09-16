<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Compressed | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 183 MB | 1.9 s |
| DistilBERT | **Infini** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_COMP}} | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 712 MB | 4.1 s |
| RoBERTa | **Infini** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_COMP}} | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 793 MB | 4.2 s |
| GPT-2 | **Infini** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_COMP}} | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After Infini | Δ | Δ % |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_DELTA}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_DELTA}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_DELTA}} | +{{G_PARAM_PCT}}% |

**Breakdown (DistilBERT, per layer):**

| Component | Per layer | 6 layers |
|---|---|---|
| Memory $M$ | 98,304 | 589,824 |
| Gate $\beta$ | 1 | 6 |
| LoRA (4 projections) | 98,304 | 589,824 |
| **Total** | **196,609** | **1,179,654** |

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | Infini | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | Infini | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | Infini | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.4 Ablation: Compression Dimension $d_c$

**Table 7.4: Accuracy vs compression dim.**

| $d_c$ | DistilBERT | RoBERTa | GPT-2 | Memory Δ |
|---|---|---|---|---|
| 32 | {{D_C32}} | {{R_C32}} | {{G_C32}} | +{{M32}} |
| 64 | {{D_C64}} | {{R_C64}} | {{G_C64}} | +{{M64}} |
| 128 | {{D_C128}} | {{R_C128}} | {{G_C128}} | +{{M128}} |
| 256 | {{D_C256}} | {{R_C256}} | {{G_C256}} | +{{M256}} |

**Observation.** {{DC_OBS}}

## 7.5 Ablation: Local Window $W$

**Table 7.5: Accuracy vs window size.**

| $W$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_W32}} | {{R_W32}} | {{G_W32}} |
| 64 | {{D_W64}} | {{R_W64}} | {{G_W64}} |
| 128 | {{D_W128}} | {{R_W128}} | {{G_W128}} |
| 256 | {{D_W256}} | {{R_W256}} | {{G_W256}} |

## 7.6 Ablation: $\beta$ Initialization

**Table 7.6: Accuracy vs initial β.**

| β init | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| -10 | {{D_B10}} | {{R_B10}} | {{G_B10}} |
| -5 | {{D_B5}} | {{R_B5}} | {{G_B5}} |
| -2 | {{D_B2}} | {{R_B2}} | {{G_B2}} |
| 0 | {{D_B0}} | {{R_B0}} | {{G_B0}} |

**Observation.** {{BETA_OBS}}

## 7.7 Ablation: LoRA Rank

**Table 7.7: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} |

## 7.8 Comparison with Companion Methods

**Table 7.8: Infini vs prior modules.**

| Method | DistilBERT | RoBERTa | GPT-2 | Params |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 1.00× |
| nha | 90.25% | FAIL | 78.78% | 1.01× |
| eat | 90.25% | 0.00% | 78.78% | 1.01× |
| **infini** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **{{PARAM_RATIO}}×** |

## 7.9 Failure Cases

**Table 7.9: Failures.**

| Model | Status | Accuracy | Reason |
|---|---|---|---|
| DistilBERT | {{D_STATUS}} | {{D_ACC}} | {{D_REASON}} |
| RoBERTa | {{R_STATUS}} | {{R_ACC}} | {{R_REASON}} |
| GPT-2 | {{G_STATUS}} | {{G_ACC}} | {{G_REASON}} |

## 7.10 Summary

| Metric | Value |
|---|---|
| Accuracy drop (best) | {{BEST_DELTA}}% |
| Parameter increase | +{{PARAM_CHANGE}}% |
| Memory Δ | +{{MEMORY_DELTA}} |
| Inference ratio | {{INFERENCE_RATIO}}× |
```

---