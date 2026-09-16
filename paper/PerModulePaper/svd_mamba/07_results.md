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
| DistilBERT | **SVD-Mamba** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | +{{D_PARAM_PCT}}% | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 712 MB | 4.1 s |
| RoBERTa | **SVD-Mamba** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | +{{R_PARAM_PCT}}% | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 793 MB | 4.2 s |
| GPT-2 | **SVD-Mamba** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | +{{G_PARAM_PCT}}% | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After SVD-Mamba | Δ | Δ % |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_DELTA}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_DELTA}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_DELTA}} | +{{G_PARAM_PCT}}% |

**Per-layer breakdown (DistilBERT):**

| Component | Params |
|---|---|
| Q/K/V/O SVD projections | ~600K |
| SSM in_proj | ~1.2M |
| SSM x_proj, dt_proj | ~500K |
| SSM out_proj | ~600K |
| Conv1d | ~3K |
| Global V, Global out | ~50K |
| Gating | ~590K |
| LoRA | ~100K |
| **Total per layer** | **~3.6M** |

**Baseline** per layer: ~1.77M. **Increase:** ~2×.

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | SVD-Mamba | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | SVD-Mamba | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | SVD-Mamba | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.4 Ablation: SSM State Dimension

**Table 7.4: Accuracy vs $d_\text{state}$.**

| $d_\text{state}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_DS8}} | {{R_DS8}} | {{G_DS8}} |
| 16 | {{D_DS16}} | {{R_DS16}} | {{G_DS16}} |
| 32 | {{D_DS32}} | {{R_DS32}} | {{G_DS32}} |
| 64 | {{D_DS64}} | {{R_DS64}} | {{G_DS64}} |

## 7.5 Ablation: Convolution Kernel

**Table 7.5: Accuracy vs $d_\text{conv}$.**

| $d_\text{conv}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 2 | {{D_DC2}} | {{R_DC2}} | {{G_DC2}} |
| 4 | {{D_DC4}} | {{R_DC4}} | {{G_DC4}} |
| 8 | {{D_DC8}} | {{R_DC8}} | {{G_DC8}} |

## 7.6 Ablation: Global Rank

**Table 7.6: Accuracy vs $r_g$.**

| $r_g$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_GR4}} | {{R_GR4}} | {{G_GR4}} |
| 8 | {{D_GR8}} | {{R_GR8}} | {{G_GR8}} |
| 16 | {{D_GR16}} | {{R_GR16}} | {{G_GR16}} |

## 7.7 Ablation: Gate Bias

**Table 7.6: Accuracy vs initial gate bias.**

| bias | $\sigma(b)$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| -10 | 4.5e-5 | {{D_GB10}} | {{R_GB10}} | {{G_GB10}} |
| -7 | 9.1e-4 | {{D_GB7}} | {{R_GB7}} | {{G_GB7}} |
| -5 | 6.7e-3 | {{D_GB5}} | {{R_GB5}} | {{G_GB5}} |
| 0 | 0.5 | {{D_GB0}} | {{R_GB0}} | {{G_GB0}} |

## 7.8 Gate Evolution During Training

**Table 7.8: Mean $\gamma$ over recovery steps.**

| Step | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 0.0009 | 0.0009 | 0.0009 |
| 10 | {{D_G10}} | {{R_G10}} | {{G_G10}} |
| 25 | {{D_G25}} | {{R_G25}} | {{G_G25}} |
| 50 | {{D_G50}} | {{R_G50}} | {{G_G50}} |

## 7.9 Comparison with Companion Methods

**Table 7.9: SVD-Mamba vs prior modules.**

| Method | DistilBERT | RoBERTa | GPT-2 | Params |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 1.00× |
| aasvd (r=64) | 88.5% | 75.5% | 76.8% | 0.20× |
| nha | 90.25% | FAIL | 78.78% | 1.01× |
| oha | 89.2% | 76.5% | 77.8% | 1.02× |
| **svd_mamba** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **{{PARAM_RATIO}}×** |

## 7.10 Failure Cases

**Table 7.10: Failures.**

| Model | Status | Accuracy | Reason |
|---|---|---|---|
| DistilBERT | {{D_STATUS}} | {{D_ACC}} | {{D_REASON}} |
| RoBERTa | {{R_STATUS}} | {{R_ACC}} | {{R_REASON}} |
| GPT-2 | {{G_STATUS}} | {{G_ACC}} | {{G_REASON}} |

## 7.11 Summary

| Metric | Value |
|---|---|
| Accuracy drop (best) | {{BEST_DELTA}}% |
| Parameter increase | +{{PARAM_CHANGE}}% |
| Inference ratio | {{INFERENCE_RATIO}}× |
| Gate at end of training | {{GATE_FINAL}} |
```

---