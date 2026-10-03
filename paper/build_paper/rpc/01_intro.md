<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Self-attention (Vaswani et al., 2017) is the dominant compute cost in 
transformers for long sequences. For input $X \in \mathbb{R}^{B \times L \times d}$, 
the attention score matrix $S = Q K^\top \in \mathbb{R}^{B \times H \times L \times L}$ 
requires $\mathcal{O}(L^2 d_h)$ FLOPs and $\mathcal{O}(L^2)$ memory. Even 
with FlashAttention (Dao et al., 2022), which avoids materializing $S$, the 
attention FLOPs remain $\mathcal{O}(L^2 d_h)$.

**Random projections** offer a classical way to reduce the cost of 
high-dimensional inner products. The **Johnson–Lindenstrauss (JL) lemma** 
(Johnson & Lindenstrauss, 1984) states that a random linear map 
$\Omega : \mathbb{R}^d \to \mathbb{R}^r$ approximately preserves pairwise 
distances with high probability when $r = \mathcal{O}(\log N / \epsilon^2)$ 
for $N$ points. This property is widely used in nearest-neighbor search 
(Indyk & Motwani, 1998), streaming algorithms, and kernel approximations.

**RPC applies JL to attention.** Instead of computing $Q_i K_j^\top$ directly, 
RPC first projects $Q_i$ and $K_j$ into a smaller subspace:

$$
\tilde{Q}_i = Q_i \Omega, \qquad \tilde{K}_j = K_j \Omega
\tag{1.1}
$$

then computes the score in the projected space:

$$
s_{ij} = \frac{\tilde{Q}_i \tilde{K}_j^\top}{\sqrt{r}}
\tag{1.2}
$$

If $\Omega$ preserves inner products (which it does by JL), then 
$s_{ij} \approx Q_i K_j^\top / \sqrt{d_h}$.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ attention layers, 
we seek a modified attention block $\mathcal{A}'$ such that:

1. $\mathcal{A}'$ approximates the pretrained attention.
2. $\mathcal{A}'$ reduces score-computation FLOPs from $\mathcal{O}(L^2 d_h)$ 
   to $\mathcal{O}(L d_h r + L^2 r)$ with $r < d_h$.
3. $\mathcal{A}'$ works across four attention architectures.
4. $\mathcal{A}'$ preserves causality for decoders.

We **do not** aim to reduce memory or model size. RPC's efficiency benefit 
is purely in FLOPs for score computation.

## 1.3 Our Approach: RPC

RPC replaces the standard score computation with:

1. **Projection:** Multiply $Q, K$ by a fixed random matrix $\Omega \in \mathbb{R}^{d_h \times r}$.
2. **Score:** Compute $\tilde{Q} \tilde{K}^\top / \sqrt{r}$ in the projected space.
3. **Attend:** Apply softmax and multiply by $V$ (unprojected).

The projection $\Omega$ is:
- **Head-wise:** Each head has its own $\Omega_h$.
- **Fixed:** Not trained; sampled once at initialization.
- **Gaussian:** Entries $\sim \mathcal{N}(0, 1/d_h)$.

The output projection is unchanged. LoRA adapters are added to all four 
projections for recovery fine-tuning.

## 1.4 Contributions

- **A random projection attention mechanism.** RPC sketches Q/K to reduce 
  score FLOPs.
- **Head-wise projections.** Each head has its own $\Omega_h$, allowing 
  head-specific subspaces.
- **Multi-architecture support.** Same codebase handles flat, nested, 
  causal, and GQA+RoPE.
- **Honest reporting.** RPC does not reduce parameters or memory.
- **Full reproducibility.** Code and checkpoints released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on JL and 
sketching. Section 4 presents RPC. Section 5 details multi-architecture 
integration. Section 6 gives setup. Section 7 reports results. Section 8 
analyzes contributions. Section 9 provides guidance. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $H$ | Number of heads |
| $r$ | Sketch dimension |
| $\Omega$ | Random projection matrix |
| $\tilde{Q}, \tilde{K}$ | Sketched query, key |
| $r_L$ | LoRA rank |
| $W_Q, W_K, W_V, W_O$ | Projection matrices |

---