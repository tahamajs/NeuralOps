<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer attention and State Space Models (SSMs) offer complementary 
inductive biases for sequence modeling.

**Attention** (Vaswani et al., 2017) is content-based and provides exact 
retrieval: query $Q_i$ attends to all keys $K_j$ with weight proportional 
to inner product. Its cost is $\mathcal{O}(L^2)$ in sequence length.

**Selective SSMs** (Gu & Dao, 2023, "Mamba") maintain a small hidden state 
$h_t \in \mathbb{R}^{d \times N}$ that is updated recurrently:

$$
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \qquad
y_t = C_t h_t + D x_t
\tag{1.1}
$$

with input-dependent transition matrices. SSMs scale **linearly** with 
sequence length but have bounded state and cannot perform exact retrieval.

**SVD-Mamba combines both.** The core idea is:

1. Keep attention as the primary path.
2. Add an SSM branch with bounded state.
3. Add a low-rank global mixing term.
4. Fuse the branches via a learnable gate.

Additionally, all projection matrices are stored in **SVD-factorized form**:

$$
W = U \Sigma V^\top, \qquad U \in \mathbb{R}^{d_\text{out} \times r}, \quad \Sigma \in \mathbb{R}^r
\tag{1.2}
$$

reducing per-projection parameter count from $d_\text{in} d_\text{out}$ 
to $r(d_\text{in} + d_\text{out} + 1)$.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ attention layers, 
we seek a modified attention block $\mathcal{A}'$ such that:

1. $\mathcal{A}'$ preserves the pretrained attention behavior at initialization.
2. $\mathcal{A}'$ provides linear-time recurrence via an SSM branch.
3. $\mathcal{A}'$ uses SVD-factorized projections for storage efficiency.
4. $\mathcal{A}'$ works across four attention architectures.
5. $\mathcal{A}'$ requires no retraining of the base model.

We **do not** aim to reduce total parameter count. The SVD factorization 
is applied to projections; the SSM branch and gating add parameters.

## 1.3 Our Approach: SVD-Mamba

SVD-Mamba replaces MHSA with a three-path block:

1. **Attention path:** standard scaled-dot-product attention with 
   SVD-factorized $Q, K, V, O$ projections.
2. **SSM path:** Mamba-style selective state space with:
   - Input-dependent $\Delta_t, B_t, C_t$ via learned projections
   - Causal depthwise convolution
   - Chunked associative scan with bounded state
   - Gating with SiLU
3. **Global mixing:** low-rank ($d \to r_g \to \text{inner}$) mixing term.

The two main paths are fused via a **gating module**:

$$
\gamma = \sigma(W_g X), \qquad
Y = \gamma \odot Y_\text{SSM} + (1 - \gamma) \odot Y_\text{attn}
\tag{1.3}
$$

The gating bias is initialized to $-7.0$, so $\gamma \approx 0.0009$ at 
start — the SSM is essentially silent, and the model behaves like the 
pretrained attention.

## 1.4 Contributions

- **A hybrid attention+SSM block.** SVD-Mamba unifies softmax attention 
  and selective SSM in a single module.
- **SVD-factorized projections.** All projections are stored as 
  $U \Sigma V^\top$, reducing per-projection parameter count.
- **Initialization stability.** The gate bias $-7$ keeps the SSM silent 
  at start, preserving the pretrained behavior.
- **Bounded numerical scheme.** A chunked associative scan with 
  denominators clamped to $[10^{-4}, 10^{4}]$ prevents overflow.
- **Honest reporting.** SVD-Mamba increases total parameter count.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SSMs, 
selective state spaces, and SVD factorization. Section 4 presents SVD-Mamba. 
Section 5 details multi-architecture integration. Section 6 gives setup. 
Section 7 reports results. Section 8 analyzes contributions. Section 9 
provides guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden dimension |
| $d_\text{state}$ | SSM state dimension |
| $d_\text{conv}$ | Convolution kernel size |
| $\text{expand}$ | Expansion factor |
| $d_\text{inner}$ | Inner SSM dimension |
| $U, \Sigma, V$ | SVD factors |
| $r$ | SVD rank |
| $r_L$ | LoRA rank |
| $\gamma$ | Gate scalar |
| $\bar{A}, \bar{B}, C$ | Discretized SSM matrices |

---