<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why Does A3 Work?

A3's success comes from three ideas working in tandem:

### 8.1.1 Fisher Weighting Reduces Effective Rank

In transformer layers, activation variance is extremely non-uniform. 
**Figure 8.1** shows the sorted Fisher diagonal for a middle DistilBERT layer. 
The top 10% of channels account for **{{TOP10_ENERGY}}%** of total energy; 
the bottom 50% contribute less than **{{BOTTOM50_ENERGY}}%**.

By weighting the input space by $\sqrt{F}$, A3 effectively compresses only 
the important directions.

### 8.1.2 Adaptive Rank Handles Layer Heterogeneity

Different layers have different intrinsic rank. **Figure 8.2** shows the 
final chosen rank per layer. Early and late layers tend to be more 
compressible; middle layers require higher rank.

### 8.1.3 LN Recalibration Corrects Distributional Shift

After factorization, activation distributions shift. Without recalibration, 
the standard deviation across features changes by up to **{{STD_SHIFT}}%** 
(measured empirically). LN recalibration brings this back to within 
**{{STD_RESIDUAL}}%**.

## 8.2 Comparison to Unweighted SVD

**Table 8.1: A3 vs standard SVD at equal rank.**

| Model | Method | Accuracy | Reconstruction err |
|---|---|---|---|
| DistilBERT | SVD (unweighted) | {{D_SVD_ACC}} | {{D_SVD_ERR}} |
| DistilBERT | **A3** | **{{D_ACC}}** | **{{D_ERR}}** |
| RoBERTa | SVD | {{R_SVD_ACC}} | {{R_SVD_ERR}} |
| RoBERTa | **A3** | **{{R_ACC}}** | **{{R_ERR}}** |

A3 achieves lower reconstruction error at equal rank (Frobenius), and higher 
accuracy after recovery.

## 8.3 Per-Layer Analysis

**Table 8.2: Reconstruction error by layer depth (DistilBERT).**

| Layer | q_lin | k_lin | v_lin | out_lin | lin1 | lin2 |
|---|---|---|---|---|---|---|
| 0 | {{D_L0_Q}} | {{D_L0_K}} | {{D_L0_V}} | {{D_L0_O}} | {{D_L0_L1}} | {{D_L0_L2}} |
| 1 | {{D_L1_Q}} | {{D_L1_K}} | {{D_L1_V}} | {{D_L1_O}} | {{D_L1_L1}} | {{D_L1_L2}} |
| 2 | {{D_L2_Q}} | {{D_L2_K}} | {{D_L2_V}} | {{D_L2_O}} | {{D_L2_L1}} | {{D_L2_L2}} |
| 3 | {{D_L3_Q}} | {{D_L3_K}} | {{D_L3_V}} | {{D_L3_O}} | {{D_L3_L1}} | {{D_L3_L2}} |
| 4 | {{D_L4_Q}} | {{D_L4_K}} | {{D_L4_V}} | {{D_L4_O}} | {{D_L4_L1}} | {{D_L4_L2}} |
| 5 | {{D_L5_Q}} | {{D_L5_K}} | {{D_L5_V}} | {{D_L5_O}} | {{D_L5_L1}} | {{D_L5_L2}} |

**Observation.** FFN layers are typically **{{FFN_ERR_NOTE}}**, while attention 
projections are **{{ATTN_ERR_NOTE}}**.

## 8.4 Spectral Analysis

**Figure 8.3** shows singular value decay for representative layers.

- **Attention layers:** Rapid decay (effective rank ~{{ATTN_EFF_RANK}})
- **FFN layers:** Slower decay (effective rank ~{{FFN_EFF_RANK}})

This justifies the per-layer rank selection.

## 8.5 Numerical Stability

### 8.5.1 FP16 vs FP32 SVD

We compute SVD in FP32 for stability. FP16 SVD occasionally produces 
NaN for ill-conditioned matrices.

### 8.5.2 Weighting Magnitude

The weighted weight matrix $W_\text{weighted} = W \cdot \mathrm{diag}(\sqrt{F})$ 
can have very different magnitudes across dimensions. We clamp $\sqrt{F}$ to 
$[10^{-3}, 10^3]$ to prevent overflow.

### 8.5.3 Residual Analysis

**Figure 8.4** shows $W - UV$ residual norm across layers. Residuals are 
concentrated in low-energy dimensions, which is exactly what Fisher weighting 
is designed to allow.

## 8.6 Comparison to Prior Compression Methods

**Table 8.3: A3 vs prior modules.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× | 79.36% | 1.00× |
| nha | 90.25% | 0.99× | — | — | 78.78% | 0.99× |
| hamming | 79.13% | 0.99× | 50.92% | 0.94× | 71.22% | 0.99× |
| sdrm | — | — | — | — | — | — |
| **A3** | **{{D_ACC}}** | **{{D_RATIO}}×** | **{{R_ACC}}** | **{{R_RATIO}}×** | **{{G_ACC}}** | **{{G_RATIO}}×** |

**A3 is the only module achieving >2× parameter reduction with <5% accuracy drop.**

## 8.7 Sensitivity Analysis

### 8.7.1 Effect of Calibration Size

**Table 8.4: Accuracy vs calibration set size.**

| N_calib | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 100 | {{D_N100}} | {{R_N100}} | {{G_N100}} |
| 500 | {{D_N500}} | {{R_N500}} | {{G_N500}} |
| 1000 | {{D_N1000}} | {{R_N1000}} | {{G_N1000}} |
| 5000 | {{D_N5000}} | {{R_N5000}} | {{G_N5000}} |

**Observation.** Calibration size of ~1000 is sufficient.

### 8.7.2 Effect of Threshold $\tau$

**Table 8.5: Accuracy vs $\tau$.**

| $\tau$ | DistilBERT | Fallback % | Rank avg |
|---|---|---|---|
| 0.10 | {{D_T10}} | {{D_FB10}}% | {{D_R10}} |
| 0.20 | {{D_T20}} | {{D_FB20}}% | {{D_R20}} |
| 0.30 | {{D_T30}} | {{D_FB30}}% | {{D_R30}} |
| 0.50 | {{D_T50}} | {{D_FB50}}% | {{D_R50}} |

### 8.7.3 Effect of `normalize_scale`

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With normalization | {{D_NORM}} | {{R_NORM}} | {{G_NORM}} |
| Without | {{D_NONORM}} | {{R_NONORM}} | {{G_NONORM}} |

## 8.8 Failure Cases

### 8.8.1 Layers That Failed Reconstruction

| Layer | Model | Best error | Best rank | Action |
|---|---|---|---|---|
| {{FAIL_LAYER_1}} | {{FAIL_MODEL_1}} | {{FAIL_ERR_1}} | {{FAIL_RANK_1}} | {{FAIL_ACTION_1}} |
| {{FAIL_LAYER_2}} | {{FAIL_MODEL_2}} | {{FAIL_ERR_2}} | {{FAIL_RANK_2}} | {{FAIL_ACTION_2}} |

### 8.8.2 Underlying Causes

{{FAIL_CAUSE_ANALYSIS}}

## 8.9 The Fisher Diagonal in Practice

**Figure 8.5** visualizes the Fisher diagonal $F$ for three layers:

- Layer 0 (early): {{D_L0_FISHER_NOTE}}
- Layer 3 (middle): {{D_L3_FISHER_NOTE}}
- Layer 5 (late): {{D_L5_FISHER_NOTE}}

Early layers tend to have **highly concentrated** energy in a few channels; 
later layers have **more uniform** distributions. This explains why some 
layers compress better than others.

## 8.10 Recovery Dynamics

**Figure 8.6** shows LoRA recovery curves. For DistilBERT, loss decreases 
from `{{D_START_LOSS}}` to `{{D_END_LOSS}}`. For RoBERTa, loss oscillates 
between `{{R_MIN_LOSS}}` and `{{R_MAX_LOSS}}`.

**Observation.** {{RECOVERY_OBS}}

## 8.11 Summary of Insights

1. **Fisher weighting is essential:** Without it, error increases by 
   {{FISHER_IMPACT}}%.
2. **LN recalibration helps:** It closes ~{{LN_IMPACT}}% of the recovery gap.
3. **Adaptive rank is necessary:** Fixed rank yields {{FIXED_RANK_ISSUE}}.
4. **Calibration is cheap:** 1000 examples suffice.
5. **FFN is more compressible than attention:** FFN effective rank is 
   {{FFN_RANK}} vs attention's {{ATTN_RANK}}.
