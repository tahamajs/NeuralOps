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
truncated SVD. SVD-LLM V2 uses this as the core primitive.

**Symmetric vs. explicit factorization.** Standard implementations distribute 
$\Sigma^{1/2}$ between $U$ and $V^H$ (Golub & Van Loan, 2013). SVD-LLM V2 
instead keeps $\Sigma$ separate, sacrificing FP16 stability for 
interpretability.

**Randomized SVD.** Halko et al. (2011) provide randomized algorithms. For 
our layer sizes, exact SVD is feasible.

**Weighted low-rank.** Golub et al. (1987) and Srebro & Jaakkola (2003) study 
weighted variants. Activation-aware methods (Lin et al., 2023; Xiao et al., 
2023) use activation statistics. SVD-LLM V2 uses unweighted SVD with 
energy-adaptive rank.

## 2.2 Rank Selection Strategies

**Fixed rank.** Most methods use a global $r$ (Yu et al., 2017).

**AdaLoRA.** Zhang et al. (2023) learn per-layer rank budgets. SVD-LLM V2 
uses a fixed **energy threshold**, simpler and training-free.

**Energy-based truncation.** Standard in signal processing. SVD-LLM V2 
applies it to transformer attention.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA.** Hu et al. (2021) freeze base weights and train rank-$r$ adapters. 
SVD-LLM V2 uses LoRA as the recovery mechanism.

**DoRA.** Liu et al. (2024) decompose weights into magnitude and direction. 
Related in spirit to explicit $\Sigma$ storage.

## 2.4 Attention Variants

**Standard MHSA.** Vaswani et al. (2017) introduce the four-projection 
attention. SVD-LLM V2 preserves the four-projection structure.

**GQA.** Ainslie et al. (2023) share KV heads. SVD-LLM V2 supports GQA via 
mean-pooling of K and V weights.

**MQA.** Shazeer (2019) uses a single KV head. Compatible.

## 2.5 Model Compression

| Method | Type | Ratio | Retrain | Interpretable? |
|---|---|---|---|---|
| DistilBERT | Distillation | 1.6× | Yes | No |
| SparseGPT | Pruning | 2-4× | No | Partial |
| AWQ | Quantization | 4× | No | No |
| A3 | Low-rank + Fisher | 5× | LoRA | No |
| AASVD | Low-rank + covariance | 5× | LoRA | No |
| MinimalSVD | Low-rank + energy | 3-6× | LoRA | No |
| **SVD-LLM V2** | **Low-rank + explicit Σ** | **50\%×** | **LoRA** | **Yes** |

## 2.6 Position of Our Work

SVD-LLM V2 differs from prior work along three axes:

| Axis | Prior | SVD-LLM V2 |
|---|---|---|
| **Factorization** | Symmetric | **Explicit (U, Σ, V^H)** |
| **Interpretability** | Limited | **Full** |
| **Downstream flexibility** | Low | **High** |

**Key distinction.** Unlike symmetric-factorization methods, SVD-LLM V2 
preserves the singular spectrum for inspection and downstream applications 
(e.g., per-singular-value quantization, spectral analysis). This comes at 
the cost of slight numerical sensitivity in FP16.

---

## Comparison with Companion Methods

| Feature | MinimalSVD | MinimalNOVA | MinimalPRISM | **SVD-LLM V2** |
|---|---|---|---|---|
| Factorization | Symmetric | Symmetric | Symmetric | **Explicit** |
| Interpretable Σ | No | No | No | **Yes** |
| Pre-test | No | Yes | Yes | Yes |
| Whole-model | No | Yes | Yes | Yes |
| GQA pooling | General | General | General | **Per-head** |
| Bias parameter | Fused | Fused | Fused | **Separate** |

---