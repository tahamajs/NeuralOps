<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Comp. | Ratio | VRAM | Time |
|---|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 1.00× | 183 MB | 1.9 s |
| DistilBERT | **A3** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_COMP}} | **{{D_RATIO}}×** | {{D_VRAM}} | {{D_TIME}} |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 1.00× | 712 MB | 4.1 s |
| RoBERTa | **A3** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_COMP}} | **{{R_RATIO}}×** | {{R_VRAM}} | {{R_TIME}} |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 1.00× | 793 MB | 4.2 s |
| GPT-2 | **A3** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_COMP}} | **{{G_RATIO}}×** | {{G_VRAM}} | {{G_TIME}} |

**Key observations:**

1. Parameter reduction: **{{D_RATIO}}×** (DistilBERT), **{{R_RATIO}}×** (RoBERTa), **{{G_RATIO}}×** (GPT-2).
2. Accuracy drop: **{{D_DELTA}}%**, **{{R_DELTA}}%**, **{{G_DELTA}}%**.
3. Peak VRAM reduced by **{{VRAM_REDUCTION}}%** on all models.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After A3 | Reduction | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | **{{D_PARAM_RED}}×** | {{D_NOTE}} |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | **{{R_PARAM_RED}}×** | {{R_NOTE}} |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | **{{G_PARAM_RED}}×** | {{G_NOTE}} |

**Per-layer breakdown (DistilBERT):**

| Layer | Original | A3 | Ratio |
|---|---|---|---|
| `q_lin` (768×768) | 589,824 | {{D_Q_A3}} | {{D_Q_RATIO}}× |
| `k_lin` (768×768) | 589,824 | {{D_K_A3}} | {{D_K_RATIO}}× |
| `v_lin` (768×768) | 589,824 | {{D_V_A3}} | {{D_V_RATIO}}× |
| `out_lin` (768×768) | 589,824 | {{D_O_A3}} | {{D_O_RATIO}}× |
| `lin1` (768×3072) | 2,359,296 | {{D_L1_A3}} | {{D_L1_RATIO}}× |
| `lin2` (3072×768) | 2,359,296 | {{D_L2_A3}} | {{D_L2_RATIO}}× |

## 7.3 Reconstruction Error Distribution

**Table 7.3: Frobenius reconstruction errors by layer type.**

| Layer type | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Attention Q | {{D_REC_Q}} | {{R_REC_Q}} | {{G_REC_Q}} |
| Attention K | {{D_REC_K}} | {{R_REC_K}} | {{G_REC_K}} |
| Attention V | {{D_REC_V}} | {{R_REC_V}} | {{G_REC_V}} |
| Attention O | {{D_REC_O}} | {{R_REC_O}} | {{G_REC_O}} |
| FFN in | {{D_REC_FFI}} | {{R_REC_FFI}} | {{G_REC_FFI}} |
| FFN out | {{D_REC_FFO}} | {{R_REC_FFO}} | {{G_REC_FFO}} |
| **Mean** | **{{D_REC_MEAN}}** | **{{R_REC_MEAN}}** | **{{G_REC_MEAN}}** |

All values below threshold $\tau = {{MAX_ERROR}}$.

## 7.4 Fallback Analysis

**Table 7.4: Layers compressed vs fallback.**

| Model | Compressed | Fallback | Fallback % |
|---|---|---|---|
| DistilBERT | {{D_COMP}} | {{D_FALLBACK}} | {{D_FALLBACK_PCT}}% |
| RoBERTa | {{R_COMP}} | {{R_FALLBACK}} | {{R_FALLBACK_PCT}}% |
| GPT-2 | {{G_COMP}} | {{G_FALLBACK}} | {{G_FALLBACK_PCT}}% |

## 7.5 Inference Efficiency

**Table 7.5: Inference time and VRAM.**

| Model | Config | Time | VRAM | Tokens/s |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 183 MB | 59000 |
| DistilBERT | A3 | {{D_TIME}} s | {{D_VRAM}} MB | {{D_TPS}} |
| RoBERTa | full | 4.1 s | 712 MB | 27400 |
| RoBERTa | A3 | {{R_TIME}} s | {{R_VRAM}} MB | {{R_TPS}} |
| GPT-2 | full | 4.2 s | 793 MB | 26700 |
| GPT-2 | A3 | {{G_TIME}} s | {{G_VRAM}} MB | {{G_TPS}} |

## 7.6 Ablation: Rank

**Table 7.6: Accuracy vs compression rank.**

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 32 | {{D_R32}} | {{D_RATIO32}}× | {{R_R32}} | {{R_RATIO32}}× | {{G_R32}} | {{G_RATIO32}}× |
| 64 | {{D_R64}} | {{D_RATIO64}}× | {{R_R64}} | {{R_RATIO64}}× | {{G_R64}} | {{G_RATIO64}}× |
| 128 | {{D_R128}} | {{D_RATIO128}}× | {{R_R128}} | {{R_RATIO128}}× | {{G_R128}} | {{G_RATIO128}}× |
| 192 | {{D_R192}} | {{D_RATIO192}}× | {{R_R192}} | {{R_RATIO192}}× | {{G_R192}} | {{G_RATIO192}}× |
| 384 | {{D_R384}} | {{D_RATIO384}}× | {{R_R384}} | {{R_RATIO384}}× | {{G_R384}} | {{G_RATIO384}}× |

**Observation.** {{RANK_OBS}}

## 7.7 Ablation: Fisher Weighting

**Table 7.7: Effect of Fisher weighting.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| A3 (with Fisher) | {{D_FISHER}} | {{R_FISHER}} | {{G_FISHER}} |
| A3 (no Fisher) | {{D_NOFISH}} | {{R_NOFISH}} | {{G_NOFISH}} |
| Δ | {{D_FISHER_DELTA}} | {{R_FISHER_DELTA}} | {{G_FISHER_DELTA}} |

**Observation.** {{FISHER_OBS}}

## 7.8 Ablation: LayerNorm Recalibration

**Table 7.8: Effect of LN recalibration.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| A3 (with LN recal) | {{D_LN}} | {{R_LN}} | {{G_LN}} |
| A3 (no LN recal) | {{D_NOLN}} | {{R_NOLN}} | {{G_NOLN}} |
| Δ | {{D_LN_DELTA}} | {{R_LN_DELTA}} | {{G_LN_DELTA}} |

**Observation.** {{LN_OBS}}

## 7.9 Ablation: Recovery Steps

**Table 7.9: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.10 Training Loss Curves

**Figure 7.1** shows loss vs step for all three models.

- DistilBERT: {{D_LOSS_TRAJ}}
- RoBERTa: {{R_LOSS_TRAJ}}
- GPT-2: {{G_LOSS_TRAJ}}

## 7.11 Compression Time

**Table 7.11: Wall-clock compression time.**

| Model | Calibration | SVD | Validation | Total |
|---|---|---|---|---|
| DistilBERT | {{D_CALIB}} s | {{D_SVD}} s | {{D_VAL}} s | **{{D_TOTAL}} s** |
| RoBERTa | {{R_CALIB}} s | {{R_SVD}} s | {{R_VAL}} s | **{{R_TOTAL}} s** |
| GPT-2 | {{G_CALIB}} s | {{G_SVD}} s | {{G_VAL}} s | **{{G_TOTAL}} s** |

## 7.12 Summary

- A3 achieves **{{D_RATIO}}×–{{G_RATIO}}×** parameter reduction.
- Accuracy drop is **{{D_DELTA}}%–{{G_DELTA}}%**.
- All three models pass the 56% quality gate.
- Fisher weighting contributes **{{FISHER_CONTRIB}}%**.
- LayerNorm recalibration contributes **{{LN_CONTRIB}}%**.
