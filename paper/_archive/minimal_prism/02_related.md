<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Low-Rank Matrix Factorization

**Classical SVD.** The Eckart–Young theorem (Eckart & Young, 1936) states 
that the best rank-$r$ approximation of $W$ in Frobenius norm is the 
truncated SVD. MinimalPRISM builds on this foundation.

**Randomized SVD.** Halko et al. (2011) provide randomized algorithms with 
complexity $\mathcal{O}(dr \log k)$. For our layer sizes, exact SVD is 
feasible within seconds.

**Weighted low-rank.** Golub et al. (1987) and Srebro & Jaakkola (2003) study 
weighted variants. Activation-aware methods (Lin et al., 2023; Xiao et al., 
2023) weight the factorization by activation statistics. MinimalPRISM uses 
**unweighted** SVD but **adaptive rank**, offering a complementary axis of 
improvement.

**Tensor decomposition.** Novikov et al. (2015) use tensor-train 
decomposition. MinimalPRISM uses matrix SVD, a simpler primitive.

## 2.2 Rank Selection Strategies

**Fixed rank.** Most low-rank methods use a fixed global rank (Yu et al., 
2017). This is simple but ignores heterogeneity.

**Adaptive rank in LoRA.** AdaLoRA (Zhang et al., 2023) learns per-layer 
rank budgets via importance scores. MinimalPRISM uses **spectral energy**, a 
fixed criterion without learning.

**Energy-based truncation.** In signal processing, energy-based truncation 
is standard. MinimalPRISM applies it to transformer attention projections.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA.** Hu et al. (2021) freeze base weights and train rank-$r$ adapters. 
MinimalPRISM uses LoRA as the recovery mechanism after compression.

**DoRA.** Liu et al. (2024) decompose weights into magnitude and direction. 
Orthogonal to our work.

## 2.4 Attention Variants

**Standard MHSA.** Vaswani et al. (2017) introduce the four-projection 
attention. MinimalPRISM preserves the four-projection structure but 
factorizes each.

**GQA.** Ainslie et al. (2023) use shared KV heads. MinimalPRISM supports 
GQA via `repeat_kv` at attention time.

**MQA.** Shazeer (2019) uses a single KV head. Compatible with our approach.

## 2.5 Model Compression

| Method | Type | Ratio | Retrain | Calibration |
|---|---|---|---|---|
| DistilBERT | Distillation | 1.6× | Yes (from scratch) | — |
| TinyBERT | Distillation | 7× | Yes | — |
| SparseGPT | Pruning | 2-4× | No | Required |
| Wanda | Pruning | 2× | No | Required |
| AWQ | Quantization | 4× | No | Required |
| GPTQ | Quantization | 4× | No | Required |
| A3 | Low-rank + diagonal Fisher | 5× | LoRA | Required |
| AASVD | Low-rank + full covariance | 5× | LoRA | Required |
| **MinimalPRISM** | **Low-rank + spectral energy** | **50\%×** | **LoRA** | **Not required** |

## 2.6 Position of Our Work

MinimalPRISM differs from prior work along three axes:

| Axis | Prior | MinimalPRISM |
|---|---|---|
| **Rank selection** | Fixed or learned | Spectral energy |
| **Calibration** | Often required | Not required |
| **Injection safety** | Rarely pre-tested | Pre-test on first layer |
| **Nested encoder** | Often broken | Explicit at injection |

**Key distinction.** Unlike activation-aware methods (A3, AASVD), MinimalPRISM 
does not require calibration data. Rank is chosen purely from the weight 
spectrum, making it **training-free** and applicable where calibration data 
is unavailable. Unlike prior SVD methods, MinimalPRISM uses a **per-layer 
adaptive rank**, which requires only the weight matrix, not activations.

---

## Comparison Table

| Method | Adaptive rank | Calibration | Nested encoder | Pre-test |
|---|---|---|---|---|
| SVD (fixed) | No | No | Often broken | No |
| A3 | Retry loop | Required | Buggy | No |
| AASVD | No | Required | Buggy | No |
| SDRM | Budget-based | No | Buggy | No |
| **MinimalPRISM** | **Energy** | **No** | **Correct** | **Yes** |

---