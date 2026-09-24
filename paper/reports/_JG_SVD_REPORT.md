# Jacobian-Gated SVD (JG-SVD) Empirical Benchmark Report

**Date:** 2026-09-21  
**Target File:** `Codes/framework/compression_engine/adaptive_ranks.py`  
**Test Suite:** `Codes/framework/tests/test_adaptive_ranks_jg.py` (9/9 PASS)  
**Evaluation Model & Dataset:** `distilbert_sst2` on `sst2` (Rank=128, Parallel=1, Batch Size=8)  
**Execution Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM) on `hadi-windows`  

---

## 1. Executive Summary

Jacobian-Gated SVD (JG-SVD) was implemented into [`Codes/framework/compression_engine/adaptive_ranks.py`](../Codes/framework/compression_engine/adaptive_ranks.py). 

JG-SVD equips adaptive energy-thresholded SVD with activation-derivative importance weighting:
1. **FFN Layers (GELU & SiLU):** Weighted by elementwise squared derivative importance:
   - GELU: $D_{\text{FFN}} = \mathbb{E}[(\text{GELU}'(Y))^2]$, where $\text{GELU}'(y) = \Phi(y) + y\phi(y)$.
   - SiLU: $D_{\text{FFN}} = \mathbb{E}[(\text{SiLU}'(Y))^2]$, where $\text{SiLU}'(y) = \sigma(y)(1 + y(1 - \sigma(y)))$.
2. **Attention Q/K Projections:** Weighted by the empirical scalar Softmax Jacobian variance:
   - $D_{\text{Attn}} = \mathbb{E}[P(1-P)]$, where $P = \text{softmax}(QK^\top / \sqrt{d})$.
3. **Alternating Least Squares (ALS):** Two-sided objective minimizing $\| D^{1/2}(W - AB) C_X^{1/2} \|_F^2$.

All 9 comprehensive unit tests passed both locally and on the remote GPU host. A multi-seed benchmark across 3 independent seeds (`42`, `123`, `456`) was conducted comparing Adaptive Ranks with JG-SVD OFF vs. JG-SVD ON.

---

## 2. Unit Test Verification Results

The test suite [`Codes/framework/tests/test_adaptive_ranks_jg.py`](../Codes/framework/tests/test_adaptive_ranks_jg.py) validated mathematical precision and backward compatibility:

| Test ID | Test Name | Expected | Measured | Status |
|---|---|---|---|---|
| **Test 1** | GELU derivative squared at $y=0$ | $0.2500$ | $0.2500$ | **PASS** |
| **Test 2** | GELU derivative squared at $y=5$ (active saturation) | $\approx 1.0000$ | $1.0000$ | **PASS** |
| **Test 3** | GELU derivative squared at $y=-5$ (inactive saturation) | $\approx 0.0000$ | $0.0000$ | **PASS** |
| **Test 4** | SiLU derivative squared at $y=0$ | $0.2500$ | $0.2500$ | **PASS** |
| **Test 5** | Activation detection on DistilBERT (`ffn.lin1`, `intermediate.dense`) | `'gelu'` | `'gelu'` | **PASS** |
| **Test 6** | Activation detection on Qwen/LLaMA (`gate_proj`, `up_proj`) | `'silu'` | `'silu'` | **PASS** |
| **Test 7** | Attention Q/K detection (`q_lin`, `k_lin`, `q_proj`, `k_proj`) | `True` for Q/K, `False` for V/O | Matched | **PASS** |
| **Test 8** | Backward compatibility (`use_jg_svd=False`) | Unchanged pipeline | Matched | **PASS** |
| **Test 9** | Opt-in JG-SVD Factorization (`use_jg_svd=True`) | Valid $A, B$ factors | Finite error ($0.6679$) | **PASS** |

---

## 3. Empirical Benchmark Results (DistilBERT / SST-2, Rank 128)

Every metric in the tables below is sourced directly from the generated CSV files:

- `results/distilbert_sst2_sst2_sweep/summary_jg_off_ranks_seed42.csv`
- `results/distilbert_sst2_sst2_sweep/summary_jg_on_ranks_seed42.csv`
- `results/distilbert_sst2_sst2_sweep/summary_jg_off_ranks_seed123.csv`
- `results/distilbert_sst2_sst2_sweep/summary_jg_on_ranks_seed123.csv`
- `results/distilbert_sst2_sst2_sweep/summary_jg_off_ranks_seed456.csv`
- `results/distilbert_sst2_sst2_sweep/summary_jg_on_ranks_seed456.csv`

### Table 1: Multi-Seed Comparison (JG-SVD OFF vs. JG-SVD ON)

| Seed | Config | Accuracy (%) | F1 Score | Val Loss | Avg Recon Error | Compressed Layers | Compression Rate |
|---|---|---|---|---|---|---|---|
| **42** | JG-SVD OFF (Baseline) | 87.27% | 0.8746 | 0.3759 | 0.7561 | 24 / 24 | 1.1641× |
| **42** | **JG-SVD ON** | 87.16% | 0.8724 | **0.3711** | **0.7547** | 24 / 24 | 1.1641× |
| **123** | JG-SVD OFF (Baseline) | 85.21% | 0.8498 | 0.3598 | 0.7561 | 24 / 24 | 1.1641× |
| **123** | **JG-SVD ON** | **85.32%** | **0.8512** | **0.3482** | **0.7547** | 24 / 24 | 1.1641× |
| **456** | JG-SVD OFF (Baseline) | 86.01% | 0.8610 | 0.3515 | 0.7561 | 24 / 24 | 1.1641× |
| **456** | **JG-SVD ON** | **86.24%** | **0.8636** | **0.3209** | **0.7547** | 24 / 24 | 1.1641× |
| **Mean ± Std** | **JG-SVD OFF** | **86.16 ± 1.04%** | **0.8618 ± 0.0124** | **0.3624 ± 0.0124** | **0.7561 ± 0.0000** | 24 / 24 | 1.1641× |
| **Mean ± Std** | **JG-SVD ON** | **86.24 ± 0.92%** | **0.8624 ± 0.0106** | **0.3467 ± 0.0251** | **0.7547 ± 0.0000** | 24 / 24 | 1.1641× |

---

## 4. Key Findings & Scientific Analysis

1. **Reconstruction Error Reduction:**
   - Enabling Jacobian-Gated weighting consistently reduced average Frobenius reconstruction error across all 24 compressed layers from **0.7561** down to **0.7547** (a reduction of 0.0014 across the entire model).
2. **Generalization & Validation Loss:**
   - On every seed, JG-SVD yielded a lower validation cross-entropy loss:
     - Seed 42: $0.3759 \to 0.3711$ ($-0.0048$)
     - Seed 123: $0.3598 \to 0.3482$ ($-0.0116$)
     - Seed 456: $0.3515 \to 0.3209$ ($-0.0306$)
     - Mean validation loss improved from **0.3624** to **0.3467** ($-0.0157$).
3. **Accuracy Stability:**
   - Mean classification accuracy increased slightly from **86.16%** to **86.24%** (+0.08%), with a lower variance across seeds (standard deviation decreased from $1.04\%$ to $0.92\%$).
4. **Efficiency:**
   - Memory overhead is negligible: peak VRAM remained identical at **414.55 MB** across both configurations.
   - Compression rate is invariant at **1.1641×** (14.09% parameter reduction on DistilBERT).
