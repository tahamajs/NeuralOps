<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Low-Rank Matrix Factorization

**Classical SVD.** The Eckart–Young theorem (Eckart & Young, 1936) states 
that the best rank-$r$ approximation of $W$ in Frobenius norm is the truncated 
SVD. MinimalSVD builds on this foundation.

**Randomized SVD.** Halko et al. (2011) provide randomized algorithms with 
complexity $\mathcal{O}(dr \log k)$. For our layer sizes, exact SVD is 
feasible within seconds.

**Weighted low-rank.** Golub et al. (1987) and Srebro & Jaakkola (2003) study 
weighted variants. Activation-aware methods (Lin et al., 2023; Xiao et al., 
2023) weight the factorization by activation statistics. MinimalSVD uses 
**unweighted** SVD but **adaptive rank**, offering a complementary axis of 
improvement.

**Tensor decomposition.** Novikov et al. (2015) use tensor-train 
decomposition. MinimalSVD uses matrix SVD, a simpler primitive.

## 2.2 Rank Selection Strategies

**Fixed rank.** Most low-rank methods use a fixed global rank (Yu et al., 
2017). Simple but ignores heterogeneity.

**Adaptive rank in LoRA.** AdaLoRA (Zhang et al., 2023) learns per-layer 
rank budgets. MinimalSVD uses **spectral energy**, a fixed criterion 
without learning.

**Energy-based truncation.** Standard in signal processing. MinimalSVD 
applies it to transformer attention projections.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA.** Hu et al. (2021) freeze base weights and train rank-$r$ adapters. 
MinimalSVD uses LoRA as the recovery mechanism after compression.

**DoRA.** Liu et al. (2024) decompose weights into magnitude and direction. 
Orthogonal to our work.

## 2.4 Attention Variants

**Standard MHSA.** Vaswani et al. (2017) introduce the four-projection 
attention. MinimalSVD preserves the four-projection structure but factorizes 
each.

**GQA.** Ainslie et al. (2023) share KV heads. MinimalSVD supports GQA via 
`repeat_kv`.

**MQA.** Shazeer (2019) uses a single KV head. Compatible with our approach.

## 2.5 Model Compression

| Method | Type | Ratio | Retrain | Calibration |
|---|---|---|---|---|
| DistilBERT | Distillation | 1.6× | Yes | — |
| TinyBERT | Distillation | 7× | Yes | — |
| SparseGPT | Pruning | 2-4× | No | Required |
| Wanda | Pruning | 2× | No | Required |
| AWQ | Quantization | 4× | No | Required |
| GPTQ | Quantization | 4× | No | Required |
| A3 | Low-rank + diagonal Fisher | 5× | LoRA | Required |
| AASVD | Low-rank + full covariance | 5× | LoRA | Required |
| **MinimalSVD** | **Low-rank + energy** | **50\%×** | **LoRA** | **Not required** |

## 2.6 Position of Our Work

MinimalSVD differs from prior work along three axes:

| Axis | Prior | MinimalSVD |
|---|---|---|
| **Rank selection** | Fixed or learned | Spectral energy |
| **Calibration** | Often required | Not required |
| **Code complexity** | High (framework) | Minimal (auditable) |

**Key distinction.** MinimalSVD is designed as a **reference baseline**. It 
uses the same energy-adaptive rank criterion as MinimalNOVA and MinimalPRISM, 
but without the injection-time safety protocols and whole-model traversal. 
This makes the code shorter and easier to audit, at the cost of requiring 
the caller to handle model traversal.

---

## Comparison of the Minimal Family

| Feature | MinimalSVD | MinimalNOVA | MinimalPRISM |
|---|---|---|---|
| Energy-adaptive rank | ✅ | ✅ | ✅ |
| LoRA recovery | ✅ | ✅ | ✅ |
| Multi-architecture | ✅ | ✅ | ✅ |
| Whole-model traversal | ❌ | ✅ | ✅ |
| Pre-test on first layer | ❌ | ✅ | ✅ |
| Injection-time nested detection | ❌ | ❌ | ✅ |
| Double-injection guard | ❌ | ❌ | ✅ |
| Code lines | ~280 | ~500 | ~550 |

---