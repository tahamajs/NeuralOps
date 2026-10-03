<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Full self-attention is $\mathcal{O}(L^2)$ in sequence length $L$, which 
becomes prohibitive for $L > 512$. Sliding-window attention reduces this to 
$\mathcal{O}(L \cdot W)$, but at a fundamental cost: tokens outside the window 
cannot attend to each other. Long-range dependencies must propagate through 
the residual stream over many layers.

**Hybrid attention** addresses this by combining a cheap local branch with 
a cheap global branch. The design question is: what should the global branch 
look at?

- **Sliding window + global tokens** (Longformer): the global branch attends 
  to a fixed set of "global" tokens. Simple but limited.
- **Sliding window + compressed memory** (Infini-Attention): the global 
  branch attends to a **learned matrix**. Powerful but adds $H \cdot d_c \cdot d_h$ 
  parameters.
- **Sliding window + sequence compression** (this work, OHA): the global 
  branch attends to **$d_c$ summary slots** computed by soft-attention over 
  the sequence.

**OHA's key idea:** Rather than learning a fixed memory matrix, compute a 
**data-dependent compressed summary** of the current sequence. The compressed 
summary adapts to each input.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ attention layers, 
we seek a modified attention block $\mathcal{A}'$ such that:

1. $\mathcal{A}'$ reproduces the pretrained behavior at initialization.
2. $\mathcal{A}'$ reduces attention cost from $\mathcal{O}(L^2)$ to 
   $\mathcal{O}(L \cdot W + L \cdot d_c)$.
3. $\mathcal{A}'$ maintains access to global context via the compressed 
   summary.
4. $\mathcal{A}'$ works across four attention architectures.

We do **not** aim to reduce model size. OHA targets **inference efficiency** 
for long sequences.

## 1.3 Our Approach: OHA

OHA replaces MHSA with a two-branch block:

1. **Local branch:** sliding-window attention, $\mathcal{O}(L W)$.
2. **Global branch:** attention to $d_c$ compressed summary slots, 
   $\mathcal{O}(L d_c)$.
3. **Fusion:** per-head gate.

The compressed summary is computed by **soft-attention pooling**:

$$
C_K = \text{softmax}(W_k^\text{comp} X, \text{dim}=L) \cdot K \in \mathbb{R}^{B \times d_c \times d}
\tag{1.1}
$$

where $W_k^\text{comp} \in \mathbb{R}^{d \times d_c}$ produces pooling weights. 
Similarly for $C_V$.

The global branch attends to these compressed summaries:

$$
\text{GlobalAttn}_i = \text{softmax}\!\left(\frac{Q_i C_K^\top}{\sqrt{d_h}}\right) C_V
\tag{1.2}
$$

The two branches are fused via a per-head gate:

$$
\text{output} = \gamma \odot \text{LocalAttn} + (1 - \gamma) \odot \text{GlobalAttn}
\tag{1.3}
$$

with $\gamma = \sigma(W_g X) \in [0, 1]^{B \times H}$.

The gate bias is initialized to $2.0$, giving $\sigma(2) \approx 0.88$ — 
favoring the local branch at start.

## 1.4 Contributions

- **A sequence-compressed hybrid attention block.** OHA combines local 
  sliding-window attention with a global branch that attends to $d_c$ 
  data-dependent summary slots.
- **Soft-attention pooling.** The compression mechanism uses softmax 
  attention over the sequence, producing input-adaptive summaries.
- **Per-head gating.** Each head learns its own local/global balance, 
  initialized to favor local.
- **Multi-architecture support.** Same codebase handles flat, nested, 
  causal, and GQA+RoPE models.
- **Honest reporting.** OHA increases parameter count; the "compression" 
  refers to sequence length, not model size.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background. Section 4 
presents OHA. Section 5 details multi-architecture integration. Section 6 
gives setup. Section 7 reports results. Section 8 analyzes contributions. 
Section 9 provides guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $W$ | Sliding window size |
| $d_c$ | Compression rank (summary slots) |
| $H$ | Number of heads |
| $d_h$ | Per-head dimension |
| $C_K, C_V$ | Compressed summary keys, values |
| $\gamma$ | Per-head gate |
| $W_k^\text{comp}, W_v^\text{comp}$ | Compression projections |
| $W_g$ | Gate projection |
| $r$ | LoRA rank |

---
