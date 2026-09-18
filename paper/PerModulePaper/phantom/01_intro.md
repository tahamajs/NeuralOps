<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer attention is expensive in two orthogonal ways. First, the 
projection matrices $W_Q, W_K, W_V, W_O$ are dense and consume the majority 
of model parameters. For DistilBERT, these four matrices contain 
**4 × 768² = 2.36M parameters per layer**, or 14.2M across six layers. Second, 
the attention matrix $Q K^\top$ scales quadratically with sequence length 
$L$, requiring $\mathcal{O}(L^2)$ memory and time.

Prior work addresses these problems separately. Low-rank factorization 
(Eckart–Young SVD) reduces projection parameters but leaves attention 
quadratic. Sparse attention (Longformer, BigBird) reduces attention cost but 
leaves projections dense. **PHANTOM addresses both problems in a single 
architecture**, using a combination of **leverage-score CUR decomposition** 
for projections and **angular LSH** for attention.

### 1.1.1 Why CUR Instead of SVD?

Classical SVD is optimal in Frobenius norm. However, the SVD factors $U, V$ 
are dense and **interpretability is lost**. CUR decomposition (Mahoney & 
Drineas, 2009) instead selects actual **columns and rows** of the original 
matrix:

$$
W \approx C \cdot U^+ \cdot R
\tag{1.1}
$$

where:
- $C \in \mathbb{R}^{d_\text{out} \times r}$ contains $r$ selected columns of $W$.
- $R \in \mathbb{R}^{r \times d_\text{in}}$ contains $r$ selected rows of $W$.
- $U^+$ is the pseudoinverse of the intersection matrix.

This has two advantages: (i) the selected rows/columns are **actual directions 
in the original space**, and (ii) the choice is governed by **statistical 
leverage scores** that measure each row's contribution to the matrix's 
spectral norm.

### 1.1.2 Why LSH for Attention?

Dense attention compares every query to every key, but **not all matches are 
relevant**. Recent work on attention entropy (Zhai et al., 2023) shows that 
attention distributions are highly peaked: fewer than 10% of positions 
typically receive non-negligible weight. LSH exploits this by grouping tokens 
with similar hidden representations into buckets and computing attention 
only **within buckets**.

Our angular LSH (Charikar, 2002) is particularly simple: for each token, 
project its normalized hidden state onto $k$ random directions, take the signs, 
and use the resulting $k$-bit code as a bucket identifier. Tokens with the 
same code attend to each other.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ layers, we seek 
a compressed model $\mathcal{M}'$ such that:

$$
\mathcal{M}' \approx \mathcal{M}, \quad
\frac{|\theta(\mathcal{M}')|}{|\theta(\mathcal{M})|} \ll 1, \quad
\text{Time}(\mathcal{M}') = \mathcal{O}(L \cdot B)
\tag{1.2}
$$

where $B$ is the average LSH bucket size.

We impose:

1. **Parameter reduction:** $r(d_\text{in} + d_\text{out}) < d_\text{in} d_\text{out}$.
2. **Reconstruction bound:** $\|W - CUR^+\|_F / \|W\|_F \leq \tau$.
3. **Causality preservation** for decoder models.
4. **Multi-architecture support** (flat, nested, Conv1D).
5. **Safe fallback** per layer when validation fails.

## 1.3 Our Approach

PHANTOM makes three contributions to the existing compression landscape:

1. **Leverage-score CUR for projections.** We use singular-vector-derived 
   leverage scores to select rows and columns, then reconstruct the weight 
   matrix as $C \cdot U^+ \cdot R$. This produces a structured factorized 
   layer with genuine parameter reduction.

2. **Multi-round angular LSH for attention.** We apply $R = 4$ independent 
   rounds of angular LSH, each producing a separate attention estimate, and 
   average the results. Multi-round averaging reduces the variance of a single 
   random projection.

3. **Rectified softpick.** We replace softmax with a rectified activation 
   that drops far-tail weights while falling back to softmax when no candidate 
   survives. This is a numerically stable variant of the softpick activation 
   proposed by Lu et al. (2024).

## 1.4 Contributions

- **A unified compression primitive.** PHANTOM compresses both projections 
  and attention in a single architecture.
- **Leverage-score CUR factorization.** We provide a principled algorithm for 
  selecting rows and columns and reconstructing the weight matrix.
- **Multi-round LSH attention.** We show that 4 rounds suffice for stable 
  attention on SST-2.
- **A production-hardened implementation.** PyTorch module with fallback, 
  Frobenius validation, GPT-2 Conv1D support, and LoRA recovery.
- **Full reproducibility.** Code, weights, and scripts released under Apache 2.0.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on CUR, LSH, 
and softpick. Section 4 presents PHANTOM. Section 5 describes architecture 
integration. Section 6 details the experimental setup. Section 7 reports 
results. Section 8 analyzes the contributions of each component. Section 9 
discusses practical guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W \in \mathbb{R}^{m \times n}$ | Weight matrix |
| $C \in \mathbb{R}^{m \times r}$ | Selected columns |
| $R \in \mathbb{R}^{r \times n}$ | Selected rows |
| $U \in \mathbb{R}^{r \times r}$ | Intersection matrix |
| $\pi_j$ | Leverage score of column $j$ |
| $\rho_i$ | Leverage score of row $i$ |
| $r$ | CUR rank |
| $R_\ell$ | Number of LSH rounds |
| $k$ | LSH bits per round |
| $B$ | Average bucket size |
| $\tau$ | Frobenius error threshold |
| $\tau_s$ | Softpick temperature |

---