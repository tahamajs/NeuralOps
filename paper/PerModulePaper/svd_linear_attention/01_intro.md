<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer attention is the dominant cost for long sequences. It is 
$\mathcal{O}(N^2)$ in both time and memory, and its weight matrices are large: 
for DistilBERT-base, the four attention projections account for 4.7M of the 
66.9M total parameters. Two complementary efficiency levers exist:

1. **Parameter reduction** via low-rank factorization (ASVD, AASVD).
2. **Complexity reduction** via linear attention (Performer, Linear Transformer).

These have traditionally been pursued separately. SVD-Linear Attention unifies 
both, delivering the size benefit of the former and the runtime benefit of the 
latter.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ attention layers, 
we seek a drop-in attention replacement $\mathcal{A}'$ such that:

1. Each projection matrix is stored in **factorized form** $UV^\top$ of rank 
   $r \ll \min(d_\text{in}, d_\text{out})$, reducing parameters by factor 
   $\sim (d_\text{in} + d_\text{out})/r$.
2. The attention computation runs in **linear time** $\mathcal{O}(N)$ for 
   bidirectional encoders and chunked-$\mathcal{O}(N)$ for causal decoders.
3. The module supports **streaming inference** with $\mathcal{O}(1)$ per-step 
   cost.
4. Weight transfer works across **four attention families** without 
   per-model branching.
5. **Causality is preserved** for decoders.

## 1.3 Our Approach: SVD-Linear Attention

SVD-Linear Attention replaces MHSA with a kernelized linear attention whose 
projection matrices are factorized:

$$
Q = \text{SVDLinear}_Q(X), \quad K = \text{SVDLinear}_K(X), \quad V = \text{SVDLinear}_V(X)
\tag{1.1}
$$

Attention is computed via the **kernelized associative form**:

$$
\phi(Q) \cdot (\phi(K)^\top V) \big/ \phi(Q) \cdot (\phi(K)^\top \mathbf{1})
\tag{1.2}
$$

where $\phi(x) = \text{ELU}(x) + 1$ is a positive feature map. This admits 
the **prefix-sum identity**:

$$
(\phi(K)^\top V)_t = \sum_{j \leq t} \phi(K_j) V_j^\top
\tag{1.3}
$$

which can be computed with a single `torch.cumsum` in $\mathcal{O}(N)$.

For decoders, we additionally apply an **input-dependent exponential decay**:

$$
S_t = \lambda_t S_{t-1} + \phi(K_t) V_t^\top, \qquad
\lambda_t = \exp(-\text{softplus}(W_g x_t)) \in (0, 1)
\tag{1.4}
$$

This gives the model a controllable forgetting mechanism. The decay gate 
$W_g$ is initialized with bias such that retention times are log-spaced 
between 1 and 1024 tokens, providing diverse temporal scales across heads.

## 1.4 Contributions

- **A unified compression-and-efficiency module.** SVD-Linear Attention 
  combines low-rank projection compression (real parameter reduction) with 
  $\mathcal{O}(N)$ linear attention (real FLOPs reduction).
- **Data-dependent exponential decay.** A retention factor per head 
  initialized with log-spaced decay times, controlled by a learned gate.
- **Streaming step-recurrent API.** A `step_recurrent` method enabling 
  $\mathcal{O}(1)$ per-token inference for autoregressive decoding.
- **Multi-architecture drop-in.** Same codebase handles flat, nested, 
  causal, and GQA+RoPE models.
- **Honest reporting.** SVD-Linear Attention **reduces** parameter count 
  by 50\%×–2.1\times×. This is one of the few modules in our 
  benchmark suite that achieves real compression AND real speedup.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on linear 
attention, decay gating, and SVD factorization. Section 4 presents 
SVD-Linear Attention. Section 5 details multi-architecture integration. 
Section 6 gives setup. Section 7 reports results. Section 8 analyzes 
contributions. Section 9 provides guidance. Section 10 lists limitations. 
Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $B$ | Batch size |
| $H$ | Number of heads |
| $d$ | Hidden size |
| $d_h$ | Per-head dimension |
| $r$ | SVD rank |
| $r_L$ | LoRA rank |
| $\phi(x)$ | Feature map $\text{ELU}(x) + 1$ |
| $\lambda_t$ | Decay at position $t$ |
| $S_t, Z_t$ | Associative memory state |
| $C$ | Chunk size (64) |

---