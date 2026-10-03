<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction — motivation, problem, contributions, organization
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages (800 - 1200 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

The quadratic parameter cost of transformer layers makes deployment of even 
moderately-sized models expensive. A standard BERT-base layer contains 
approximately 7 million parameters, of which 4.7 million (67%) live in the 
four attention projections $W_Q, W_K, W_V, W_O$ and the remaining 2.4 million 
in the FFN. For DistilBERT, this drops to 3.5M and 2.4M respectively. However, 
even DistilBERT (66.9M parameters) requires 134 MB of storage at FP16, which 
is not negligible for mobile or edge deployment.

Low-rank factorization is the classical approach to reducing this cost. Given 
a linear layer $y = x W^\top + b$, one replaces $W$ with a product $W \approx U V$ 
where $U \in \mathbb{R}^{d_\text{out} \times r}$ and $V \in \mathbb{R}^{r \times d_\text{in}}$, 
$r \ll \min(d_\text{in}, d_\text{out})$. The parameter count decreases from 
$d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$, achieving 
compression ratio $\frac{r(d_\text{in} + d_\text{out})}{d_\text{in} d_\text{out}}$. 
For a $768 \times 768$ matrix at rank $r = 64$, this is a 5.9× reduction 
(589,824 → 98,304 parameters).

Classical SVD is optimal in an unweighted Frobenius norm sense:

$$
W_r^* = \arg\min_{W_r : \text{rank}(W_r) = r} \|W - W_r\|_F
\tag{1.1}
$$

but this optimality is with respect to uniform treatment of all input 
dimensions. Transformer activations, however, have **highly non-uniform 
variance** across channels. Some input dimensions carry most of the signal; 
others are nearly constant. Reconstructing the near-constant dimensions as 
faithfully as the high-variance ones wastes capacity.

0.85

<!-- 
EXAMPLE continuation:
"Activation-aware compression addresses this by weighting the reconstruction 
objective per input dimension. The natural weight is the empirical Fisher 
diagonal, F_i = E_x[x_i²], which measures how much of the output variance is 
attributable to input dimension i. Reweighting the SVD by this quantity 
concentrates the low-rank approximation on the directions that matter."
-->

## 1.2 Problem Statement

Given a pretrained transformer model $\mathcal{M}$ with $N_l$ layers, each 
containing linear projections $\{W^{(\ell)}_k\}$, we seek a compressed model 
$\mathcal{M}'$ such that:

$$
\mathcal{M}' \approx \mathcal{M}, \qquad
\frac{|\theta(\mathcal{M}')|}{|\theta(\mathcal{M})|} \ll 1
\tag{1.2}
$$

where $|\theta(\cdot)|$ denotes parameter count.

We impose the following constraints:

1. **Strict reconstruction bound.** For each compressed layer, 
   $\|W - UV\|_F / \|W\|_F \leq \tau$ for threshold $\tau = 0.20$.
2. **No retraining of base weights.** Only LoRA adapters may be trained.
3. **Memory-bounded calibration.** Activation statistics must be collected in 
   $\mathcal{O}(d^2)$ memory, not $\mathcal{O}(Nd^2)$.
4. **Deterministic fallback.** If error exceeds $\tau$, the original layer is 
   preserved.

## 1.3 Our Approach

We propose **A3** (**A**ctivation-**A**ware **A**daptive factorization), 
which differs from standard SVD compression in four ways:

1. **Fisher-diagonal weighting.** We estimate the empirical Fisher diagonal 
   $F_i = \mathbb{E}_x[x_i^2]$ from calibration data and use it to reweight 
   the SVD input space: $W_\text{weighted} = W \cdot \mathrm{diag}(\sqrt{F + \epsilon})$.

2. **Streaming covariance accumulation.** We collect $F$ and $C_X$ using 
   online streaming, requiring only $\mathcal{O}(d^2)$ memory regardless of 
   calibration set size.

3. **Adaptive rank escalation.** If the reconstruction error at the initial 
   rank $r$ exceeds $\tau$, we incrementally increase the rank up to 
   $\min(d_\text{in}, d_\text{out})$, retrying the SVD until the constraint 
   is met or the fallback triggers.

4. **LayerNorm recalibration.** After factorization, we rescale each 
   LayerNorm's affine parameters $\gamma$ by the ratio of target to current 
   activation standard deviation, absorbing distributional shift introduced 
   by the factorization.

## 1.4 Contributions

We make the following contributions:

- **A streaming activation-aware compression algorithm.** We propose A3, 
  which combines Fisher weighting, streaming covariance accumulation, adaptive 
  rank escalation, and LayerNorm recalibration into a single pipeline.
- **A production-hardened implementation.** Our PyTorch implementation 
  handles flat, nested, and causal architectures with safe fallback, strict 
  Frobenius validation, and CoO(nstant) memory overhead.
- **A comprehensive evaluation.** We benchmark on DistilBERT, RoBERTa, and 
  GPT-2 on SST-2 with ablations over rank, weighting strategy, and LayerNorm 
  recalibration.
- **Full reproducibility.** Code, checkpoints, and evaluation scripts are 
  released under MIT.

## 1.5 Paper Organization

Section 2 reviews related work in low-rank compression, activation-aware 
pruning, and LayerNorm recalibration. Section 3 provides background on SVD, 
the Fisher information, and streaming statistics. Section 4 presents A3. 
Section 5 describes integration with three transformer architectures. Section 
6 details the experimental setup. Section 7 reports results. Section 8 
analyzes where A3 succeeds and fails. Section 9 discusses practical guidance. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ | Weight matrix of a linear layer | §1.1 |
| $U, V$ | Low-rank factors | §1.1 |
| $r$ | Target rank | §1.1 |
| $F \in \mathbb{R}^{d_\text{in}}$ | Fisher diagonal / activation energy | §1.1 |
| $C_X$ | Activation covariance matrix | §4.1 |
| $\tau$ | Reconstruction error threshold | §1.2 |
| $\epsilon$ | Numerical stability constant | §4.2 |
| $\sigma_t$ | Target standard deviation (for LN) | §4.5 |
| $\gamma$ | LayerNorm affine scale | §4.5 |
| $N$ | Number of calibration tokens | §4.1 |
