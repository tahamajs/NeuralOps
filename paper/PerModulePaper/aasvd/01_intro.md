<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Deploying transformer models in resource-constrained settings — mobile 
devices, edge hardware, or cost-sensitive inference services — requires 
reducing the storage, memory, and compute footprint of the model. A 
standard transformer layer contains the bulk of its parameters in four 
attention projections ($W_Q, W_K, W_V, W_O$) and two feed-forward matrices. 
For DistilBERT, these account for 6.6M of the 6.7M parameters per layer.

**Low-rank factorization** is a natural compression primitive. For a weight 
matrix $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, one replaces 
$W$ with the product $U V$ where $U \in \mathbb{R}^{d_\text{out} \times r}$ 
and $V \in \mathbb{R}^{r \times d_\text{in}}$ with $r \ll \min(d_\text{in}, d_\text{out})$. 
Parameter count drops from $d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$. 
For a $768 \times 768$ matrix at $r = 64$, this is a **6× reduction**.

Classical truncated SVD is optimal for this problem in an unweighted Frobenius 
sense, but the objective is misleading. The unweighted loss treats every 
input dimension as equally important, whereas real transformer activations 
exhibit two critical properties that unweighted SVD ignores:

1. **Heterogeneous variance.** Some input dimensions are highly active; 
   others carry almost no signal. **Figure 1.1** shows the empirical variance 
   across dimensions for a middle DistilBERT layer: the top 10% of channels 
   account for over **85%** of total energy.

2. **Inter-feature correlation.** Real activations are not independent 
   across dimensions. Residual stream dimensions are correlated through the 
   preceding LayerNorm and residual connections. Unweighted SVD ignores 
   these correlations, wasting capacity on directions that rarely occur.

**Activation-aware compression** addresses both issues by reweighting the 
factorization with a statistical summary of the input distribution. The 
natural summary is the empirical input covariance:

$$
C_X = \mathbb{E}_{x \sim \mathcal{D}}\!\left[x x^\top\right] \in \mathbb{R}^{d_\text{in} \times d_\text{in}}
\tag{1.1}
$$

Weights on high-variance or highly-correlated directions contribute more 
to the reconstruction error and should be approximated with higher fidelity. 
This is the core insight behind AASVD.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_l$ layers and a target 
compression rank $r$, we seek a compressed model $\mathcal{M}'$ such that:

$$
\mathcal{M}' \approx \mathcal{M}, \qquad
\frac{|\theta(\mathcal{M}')|}{|\theta(\mathcal{M})|} \ll 1
\tag{1.2}
$$

We impose the following constraints:

1. **Parameter reduction:** $r(d_\text{in} + d_\text{out}) < d_\text{in} d_\text{out}$ for each layer.
2. **Reconstruction bound:** $\|W - UV\|_F / \|W\|_F \leq \tau$ for threshold $\tau$.
3. **Memory-bounded calibration:** statistics accumulated in $\mathcal{O}(d^2)$ memory.
4. **No base retraining:** only LoRA adapters may be trained.
5. **Deterministic fallback:** if error exceeds $\tau$, keep the original layer.

## 1.3 Our Approach: AASVD

We propose **AASVD** (**A**ctivation-**A**ware **S**VD), which extends classical 
SVD factorization with three components:

1. **Full-covariance weighting.** We weight the SVD input space by 
   $C_X^{1/2}$, capturing both variance and correlation, not just diagonal 
   variance.

2. **Streaming EMA accumulation.** We maintain a running estimate of $C_X$ 
   using exponential moving average over calibration batches, computed 
   on CPU to prevent VRAM accumulation.

3. **Bundled calibration corpus.** We ship a default multi-domain calibration 
   corpus (English prose, code, mathematical text) so AASVD works out of the 
   box even when no downstream data is available.

## 1.4 Contributions

- **A streaming activation-aware SVD algorithm.** AASVD generalizes classical 
  SVD with full-covariance weighting and provides a production-hardened 
  implementation with EMA accumulation, safe fallback, and multi-architecture 
  support.
- **A structural replacement module.** We replace `nn.Linear` with a 
  factorized `SVDLinear` achieving genuine parameter reduction from 
  $d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$.
- **A systematic evaluation.** We benchmark on DistilBERT, RoBERTa, and GPT-2 
  on SST-2, with ablations over rank, EMA β, and covariance usage.
- **A bundled calibration corpus.** A default 5-document, multi-domain corpus 
  that achieves near-optimal calibration without user-supplied data.
- **Full reproducibility.** Code, weights, and evaluation scripts released 
  under MIT.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SVD, 
covariance estimation, and EMA. Section 4 presents AASVD. Section 5 
describes integration with three architectures. Section 6 details the 
experimental setup. Section 7 reports results. Section 8 analyzes the 
contribution of each component. Section 9 discusses practical guidance. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ | Layer weight matrix |
| $U \in \mathbb{R}^{d_\text{out} \times r}$ | Left factor |
| $V \in \mathbb{R}^{r \times d_\text{in}}$ | Right factor |
| $C_X$ | Input activation covariance |
| $C_X^{1/2}$ | Matrix square root of $C_X$ |
| $C_X^{-1/2}$ | Inverse square root |
| $\beta$ | EMA decay |
| $r$ | Compression rank |
| $\tau$ | Reconstruction error threshold |
| $\epsilon$ | Numerical stability constant |
| $N$ | Number of calibration tokens |
| $\lambda_i$ | $i$-th eigenvalue of $C_X$ |
