<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer language models concentrate the majority of their parameters in 
the feed-forward network and multi-head self-attention. In DistilBERT-base, 
the four attention projections $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$ 
account for approximately **70%** of the parameters in each transformer 
block. Reducing this cost is a high-leverage compression strategy.

**Low-rank factorization** is the classical primitive. Given a weight matrix 
$W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, one replaces it with:

$$
W \approx U \Sigma V^H
\tag{1.1}
$$

where $U \in \mathbb{R}^{d_\text{out} \times r}$, $\Sigma \in \mathbb{R}^{r \times r}$ 
diagonal, and $V^H \in \mathbb{R}^{r \times d_\text{in}}$. The parameter count 
drops from $d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out} + 1)$.

**Two design choices.** Prior low-rank methods differ in how they handle the 
singular values:

1. **Symmetric factorization.** Store $U \Sigma^{1/2}$ and $\Sigma^{1/2} V^H$ 
   as two matrices. This balances the factor magnitudes and improves FP16 
   stability, but **loses visibility of $\Sigma$**.
2. **Explicit factorization.** Store $U$, $\Sigma$, and $V^H$ separately. 
   This **preserves the singular spectrum** for inspection, downstream 
   quantization, or spectral analysis.

SVD-LLM V2 adopts the **explicit** approach. This has three advantages:

- **Interpretability.** $\Sigma$ is directly readable and can be visualized.
- **Downstream flexibility.** Per-singular-value quantization can be applied 
  without re-factorizing.
- **Composability.** Orthogonal $U$ and $V^H$ can be re-used across 
  checkpoints or merged with other adapters.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_l$ layers and a target 
energy threshold $\eta \in (0, 1)$, we seek a compressed model 
$\mathcal{M}'$ such that:

$$
\mathcal{M}' \approx \mathcal{M}, \qquad
\frac{|\theta(\mathcal{M}')|}{|\theta(\mathcal{M})|} \ll 1
\tag{1.2}
$$

subject to:

1. **Explicit factor storage.** Each projection is stored as $U$, $\Sigma$, $V^H$.
2. **Per-layer adaptive rank.** $r_\ell$ selected by energy threshold.
3. **Parameter reduction.** $r(d_\text{in} + d_\text{out} + 1) < d_\text{in} d_\text{out}$.
4. **Multi-architecture support.** Four families (flat, nested, causal, GQA).
5. **No base retraining.** Only LoRA adapters trained.

## 1.3 Our Approach: SVD-LLM V2

**SVD-LLM V2** replaces the four attention projections with explicit 
factorized buffers:

$$
W \to (U, \Sigma, V^H), \qquad
y = ((x V^H{}^\top) \Sigma) U^\top + b + \text{LoRA}(x)
\tag{1.3}
$$

The rank is chosen per projection based on spectral energy. LoRA adapters 
$A, B$ are attached with $B = 0$, so the initial output equals the SVD 
approximation.

**Injection safety.** SVD-LLM V2 combines:
- **Pre-test on the first layer** before mutating the model
- **Injection-time nested encoder detection**
- **Double-injection guard**
- **Frobenius norm validation**

**GQA handling.** When the source model has more KV heads than the target 
(e.g., Qwen with $H_{kv} = 2 < H = 8$), K and V weights are **mean-pooled 
per head group** before factorization.

## 1.4 Contributions

- **Explicit low-rank factorization.** A variant of SVD compression that 
  keeps $U$, $\Sigma$, $V^H$ separate, preserving interpretability and 
  enabling downstream flexibility.
- **Energy-threshold rank selection.** Per-projection rank chosen by 
  cumulative spectral energy.
- **Multi-architecture hardening.** Four families with pre-test, injection-time 
  nested detection, and GQA pooling.
- **A systematic evaluation.** Benchmark on three models on SST-2 with 
  ablations over $r_\text{max}$, $\eta$, and LoRA rank.
- **Full reproducibility.** Code, checkpoints, and scripts released under 
  Apache 2.0.

## 1.5 Relationship to Prior Work

SVD-LLM V2 is a continuation of the **SVD-LLM** line:

| Version | Main features |
|---|---|
| SVD-LLM V1 | Basic SVD compression |
| **SVD-LLM V2** | Explicit U/Σ/V^H + LoRA + multi-arch hardening |

It also compares to the **Minimal family** (MinimalSVD, MinimalNOVA, 
MinimalPRISM) which use symmetric factorization:

| Family | Factorization | Interpretable? |
|---|---|---|
| Minimal family | $U\sqrt{\Sigma} \cdot \sqrt{\Sigma} V^H$ | No |
| **SVD-LLM V2** | $U \cdot \Sigma \cdot V^H$ (separate) | **Yes** |

## 1.6 Paper Organization

Section 2 reviews related work. Section 3 provides background. Section 4 
presents SVD-LLM V2. Section 5 describes architecture integration. Section 6 
details the setup. Section 7 reports results. Section 8 analyzes the 
explicit-factor approach. Section 9 discusses guidance. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ | Weight matrix |
| $U \in \mathbb{R}^{d_\text{out} \times r}$ | Left singular vectors |
| $\Sigma \in \mathbb{R}^{r \times r}$ | Diagonal singular values |
| $V^H \in \mathbb{R}^{r \times d_\text{in}}$ | Right singular vectors (conjugate transpose) |
| $r$ | Target rank |
| $r_\ell$ | Adaptive rank for layer $\ell$ |
| $\sigma_i$ | $i$-th singular value |
| $E(r)$ | Cumulative energy |
| $\eta$ | Energy threshold |
| $d, H, H_{kv}, d_h$ | Model dims |
| $\alpha, r_L$ | LoRA scaling, rank |

---