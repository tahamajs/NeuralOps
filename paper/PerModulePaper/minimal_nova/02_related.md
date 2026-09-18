<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Low-Rank Matrix Factorization

**Classical SVD.** The Eckart–Young theorem (Eckart & Young, 1936) guarantees 
that the best rank-$r$ approximation of $W$ in Frobenius norm is the truncated 
SVD. MinimalNOVA uses truncated SVD as its core factorization primitive.

**Randomized SVD.** Halko et al. (2011) provide randomized algorithms with 
complexity $\mathcal{O}(dr \log k)$. For our layer sizes, exact SVD is 
feasible and preferred.

**Weighted low-rank.** Golub et al. (1987) and Srebro & Jaakkola (2003) study 
weighted variants. Related activation-aware approaches (Lin et al., 2023; 
Xiao et al., 2023) weight the factorization by activation statistics. 
MinimalNOVA uses unweighted SVD but adaptive rank.

**Tensor decomposition.** Novikov et al. (2015) use tensor-train 
decomposition for higher-order compression. MinimalNOVA uses matrix SVD, a 
simpler primitive.

## 2.2 Rank Selection Strategies

**Fixed rank.** Most SVD-based compression methods use a fixed global rank 
(Yu et al., 2017). This is simple but ignores heterogeneity.

**Adaptive rank in LoRA.** AdaLoRA (Zhang et al., 2023) learns per-layer 
rank budgets via importance scores. MinimalNOVA uses **spectral energy**, 
a fixed criterion without learning.

**Energy-based truncation.** In signal processing, energy-based truncation 
is standard. Our contribution is applying it to **transformer attention 
projections**.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA.** Hu et al. (2021) freeze base weights and train rank-$r$ adapters. 
We use LoRA as the recovery mechanism after compression.

**DoRA.** Liu et al. (2024) decompose weights into magnitude and direction. 
Orthogonal to our work.

## 2.4 Attention Variants

**Standard MHSA.** Vaswani et al. (2017) introduce the four-projection 
attention. MinimalNOVA preserves the four-projection structure but factorizes 
each.

**GQA.** Ainslie et al. (2023) use shared KV heads to reduce KV-cache size. 
MinimalNOVA supports GQA via `repeat_kv` at attention time.

**MQA.** Shazeer (2019) uses a single KV head. Compatible with our approach.

## 2.5 Model Compression

| Method | Type | Ratio | Retrain |
|---|---|---|---|
| DistilBERT | Distillation | 1.6× | Yes (from scratch) |
| TinyBERT | Distillation | 7× | Yes |
| SparseGPT | Pruning | 2-4× | No |
| Wanda | Pruning | 2× | No |
| AWQ | Quantization | 4× | No |
| GPTQ | Quantization | 4× | No |
| A3 | Low-rank + diagonal | 5× | LoRA |
| AASVD | Low-rank + full cov | 5× | LoRA |
| **MinimalNOVA** | **Low-rank + energy-adaptive** | **50\%×** | **LoRA** |

## 2.6 Position of Our Work

MinimalNOVA differs from prior work in three ways:

| Axis | Prior | MinimalNOVA |
|---|---|---|
| **Rank selection** | Fixed or learned | Energy-adaptive |
| **Weighting** | Unweighted or activation-weighted | Unweighted (spectral) |
| **Architectures** | Usually single family | Four families |
| **Nested encoder** | Often broken | Explicit bypass |
| **Conv1D (GPT-2)** | Often broken | Explicit transpose |

**Key distinction.** Unlike activation-aware methods (A3, AASVD), MinimalNOVA 
does not require calibration data. Rank is chosen purely from the weight 
spectrum, making it **training-free** and applicable to models where 
calibration data is unavailable.

---

## Comparison Table

| Method | Calibration | Multi-arch | Adaptive rank |
|---|---|---|---|
| SVD (fixed) | None | Yes | No |
| A3 | Required | Yes | Retry loop |
| AASVD | Required | Yes | No |
| SDRM | None | Yes | Budget-based |
| **MinimalNOVA** | **None** | **4 families** | **Spectral energy** |

---