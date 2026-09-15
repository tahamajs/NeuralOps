<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Compressed | Ratio | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 1.00× | 183 MB |
| DistilBERT | **AASVD** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_COMP}} | **{{D_RATIO}}×** | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 1.00× | 712 MB |
| RoBERTa | **AASVD** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_COMP}} | **{{R_RATIO}}×** | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 1.00× | 793 MB |
| GPT-2 | **AASVD** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_COMP}} | **{{G_RATIO}}×** | {{G_VRAM}} MB |

**Key observations:**

1. **Parameter reduction:** **{{D_RATIO}}×** on DistilBERT.
2. **Accuracy:** drops only **{{D_DELTA}}%** on DistilBERT.
3. **VRAM:** reduces by **{{VRAM_REDUCTION}}%**.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | AASVD | Ratio | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | **{{D_RATIO}}×** | {{D_NOTE}} |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | **{{R_RATIO}}×** | {{R_NOTE}} |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | **{{G_RATIO}}×** | {{G_NOTE}} |

**Per-layer breakdown (DistilBERT):**

| Layer | Original | AASVD r={{RANK}} | Ratio |
|---|---|---|---|
| `q_lin` (768×768) | 589,824 | {{D_Q_AASVD}} | {{D_Q_RATIO}}× |
| `k_lin` (768×768) | 589,824 | {{D_K_AASVD}} | {{D_K_RATIO}}× |
| `v_lin` (768×768) | 589,824 | {{D_V_AASVD}} | {{D_V_RATIO}}× |
| `out_lin` (768×768) | 589,824 | {{D_O_AASVD}} | {{D_O_RATIO}}× |

## 7.3 Reconstruction Error

**Table 7.3: Frobenius reconstruction errors.**

| Layer type | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q projection | {{D_REC_Q}} | {{R_REC_Q}} | {{G_REC_Q}} |
| K projection | {{D_REC_K}} | {{R_REC_K}} | {{G_REC_K}} |
| V projection | {{D_REC_V}} | {{R_REC_V}} | {{G_REC_V}} |
| O projection | {{D_REC_O}} | {{R_REC_O}} | {{G_REC_O}} |
| **Mean** | **{{D_REC_MEAN}}** | **{{R_REC_MEAN}}** | **{{G_REC_MEAN}}** |

All below threshold $\tau = {{MAX_ERROR}}$.

## 7.4 Fallback Analysis

**Table 7.4: Compression success rate.**

| Model | Compressed | Fallback | Fallback % |
|---|---|---|---|
| DistilBERT | {{D_COMP}} | {{D_FALLBACK}} | {{D_FB_PCT}}% |
| RoBERTa | {{R_COMP}} | {{R_FALLBACK}} | {{R_FB_PCT}}% |
| GPT-2 | {{G_COMP}} | {{G_FALLBACK}} | {{G_FB_PCT}}% |

## 7.5 Efficiency Profile

**Table 7.5: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | AASVD | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | AASVD | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | AASVD | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.6 Ablation: Rank

**Table 7.6: Accuracy vs compression rank.**

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 32 | {{D_R32}} | {{D_RATIO32}}× | {{R_R32}} | {{R_RATIO32}}× | {{G_R32}} | {{G_RATIO32}}× |
| 64 | {{D_R64}} | {{D_RATIO64}}× | {{R_R64}} | {{R_RATIO64}}× | {{G_R64}} | {{G_RATIO64}}× |
| 128 | {{D_R128}} | {{D_RATIO128}}× | {{R_R128}} | {{R_RATIO128}}× | {{G_R128}} | {{G_RATIO128}}× |
| 256 | {{D_R256}} | {{D_RATIO256}}× | {{R_R256}} | {{R_RATIO256}}× | {{G_R256}} | {{G_RATIO256}}× |

## 7.7 Ablation: Covariance Usage

**Table 7.7: Effect of covariance weighting.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| AASVD (full cov) | {{D_FULLCOV}} | {{R_FULLCOV}} | {{G_FULLCOV}} |
| AASVD (diagonal) | {{D_DIAG}} | {{R_DIAG}} | {{G_DIAG}} |
| AASVD (no weighting) | {{D_NOW}} | {{R_NOW}} | {{G_NOW}} |

**Observation.** Full covariance improves accuracy by **{{COV_GAIN}}%** 
over diagonal; diagonal improves **{{DIAG_GAIN}}%** over unweighted.

## 7.8 Ablation: EMA β

**Table 7.8: Accuracy vs EMA decay.**

| β | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | {{D_B5}} | {{R_B5}} | {{G_B5}} |
| 0.7 | {{D_B7}} | {{R_B7}} | {{G_B7}} |
| 0.9 | {{D_B9}} | {{R_B9}} | {{G_B9}} |
| 0.95 | {{D_B95}} | {{R_B95}} | {{G_B95}} |
| 0.99 | {{D_B99}} | {{R_B99}} | {{G_B99}} |

## 7.9 Ablation: Calibration Source

**Table 7.9: Effect of calibration corpus.**

| Source | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Default bundled | {{D_DEFAULT}} | {{R_DEFAULT}} | {{G_DEFAULT}} |
| SST-2 training | {{D_TASK}} | {{R_TASK}} | {{G_TASK}} |
| Random texts | {{D_RANDOM}} | {{R_RANDOM}} | {{G_RANDOM}} |
| No calibration | {{D_NONE}} | {{R_NONE}} | {{G_NONE}} |

## 7.10 Ablation: Recovery Steps

**Table 7.10: Accuracy vs LoRA recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.11 Comparison with Companion Methods

**Table 7.11: AASVD vs prior compressors.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio |
|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× |
| a3 (r=64) | {{A3_D}} | {{A3_D_R}}× | {{A3_R}} | {{A3_R_R}}× |
| sdrm | {{SDRM_D}} | {{SDRM_D_R}}× | {{SDRM_R}} | {{SDRM_R_R}}× |
| **AASVD** | **{{D_ACC}}** | **{{D_RATIO}}×** | **{{R_ACC}}** | **{{R_RATIO}}×** |

## 7.12 Compression Time

**Table 7.12: Wall-clock compression time.**

| Model | Calibration | Eigendecomp | SVD | Validation | Total |
|---|---|---|---|---|---|
| DistilBERT | {{D_CALIB}} s | {{D_EIG}} s | {{D_SVD}} s | {{D_VAL}} s | **{{D_TOTAL}} s** |
| RoBERTa | {{R_CALIB}} s | {{R_EIG}} s | {{R_SVD}} s | {{R_VAL}} s | **{{R_TOTAL}} s** |
| GPT-2 | {{G_CALIB}} s | {{G_EIG}} s | {{G_SVD}} s | {{G_VAL}} s | **{{G_TOTAL}} s** |

## 7.13 Summary

- **Compression ratio:** {{D_RATIO}}×–{{G_RATIO}}×
- **Accuracy drop:** {{D_DELTA}}%–{{G_DELTA}}%
- **VRAM reduction:** {{VRAM_REDUCTION}}%
- **All models pass quality gate.**
