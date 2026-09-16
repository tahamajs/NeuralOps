<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params Δ % | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 183 MB | 1.9 s |
| DistilBERT | **OHA** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | +{{D_PARAM_PCT}}% | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 712 MB | 4.1 s |
| RoBERTa | **OHA** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | +{{R_PARAM_PCT}}% | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 793 MB | 4.2 s |
| GPT-2 | **OHA** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | +{{G_PARAM_PCT}}% | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After OHA | Δ | Δ % |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_DELTA}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_DELTA}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_DELTA}} | +{{G_PARAM_PCT}}% |

**Breakdown (DistilBERT, per layer):**

| Component | Per layer | 6 layers |
|---|---|---|
| $W_k^\text{comp}$ | 24,576 | 147,456 |
| $W_v^\text{comp}$ | 24,576 | 147,456 |
| Gate | 9,228 | 55,368 |
| LoRA (4) | 98,304 | 589,824 |
| **Total** | **156,684** | **940,104** |

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | OHA | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | OHA | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | OHA | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.4 Ablation: Compression Rank

**Table 7.4: Accuracy vs compression rank.**

| $d_c$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_C8}} | {{R_C8}} | {{G_C8}} |
| 16 | {{D_C16}} | {{R_C16}} | {{G_C16}} |
| 32 | {{D_C32}} | {{R_C32}} | {{G_C32}} |
| 64 | {{D_C64}} | {{R_C64}} | {{G_C64}} |

## 7.5 Ablation: Window Size

**Table 7.5: Accuracy vs window size.**

| $W$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_W16}} | {{R_W16}} | {{G_W16}} |
| 32 | {{D_W32}} | {{R_W32}} | {{G_W32}} |
| 64 | {{D_W64}} | {{R_W64}} | {{G_W64}} |
| 128 | {{D_W128}} | {{R_W128}} | {{G_W128}} |

## 7.6 Ablation: Gate Bias

**Table 7.6: Accuracy vs gate bias init.**

| bias | σ(bias) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.0 | 0.500 | {{D_GB0}} | {{R_GB0}} | {{G_GB0}} |
| 1.0 | 0.731 | {{D_GB1}} | {{R_GB1}} | {{G_GB1}} |
| 2.0 | 0.881 | {{D_GB2}} | {{R_GB2}} | {{G_GB2}} |
| 3.0 | 0.953 | {{D_GB3}} | {{R_GB3}} | {{G_GB3}} |

## 7.7 Ablation: LoRA Rank

**Table 7.7: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} |

## 7.8 Comparison with Companion Methods

**Table 7.8: OHA vs prior modules.**

| Method | DistilBERT | RoBERTa | GPT-2 | Params |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 1.00× |
| nha | 90.25% | FAIL | 78.78% | 1.01× |
| infini | 89.5% | 76.5% | 77.5% | 1.02× |
| **oha** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **{{PARAM_RATIO}}×** |

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
| Inference ratio (L=128) | {{INFERENCE_RATIO}}× |
| Long-context benefit (L=2048) | {{LONG_CTX_SPEEDUP}}× |

---