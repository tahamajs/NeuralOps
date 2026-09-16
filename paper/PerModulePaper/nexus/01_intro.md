# 1. Introduction

## 1.1 Motivation

Sparse attention has seen rapid progress in 2024–2025. Notable recent 
advances include:

- **XAttention** (Xu et al., ICML 2025): training-free block scoring via 
  antidiagonal correlations.
- **SeerAttention** (Gao et al., NeurIPS 2025): learnable per-head block 
  sparsity.
- **DuSA** (Cao et al., NeurIPS 2025): dual-stage intra-block dense + 
  inter-block sparse.
- **AnchorAttention** (Zhu et al., 2025): stripe-based fine-grained sparsity.
- **Spectraformer** (Nguyen et al., 2025): RFF-based global rectification.
- **Softpick** (Zuhri et al., 2025): rectified softmax with natural ~47% 
  sparsity.

**These techniques are complementary but rarely combined.** NEXUS is our 
attempt to unify them in a single attention module.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $H$ heads and hidden 
dimension $d$, we seek an attention mechanism $\mathcal{A}'$ such that:

$$
\text{FLOPs}(\mathcal{A}') \ll \mathcal{O}(N^2), \quad
\mathcal{A}' \approx \mathcal{A}
\tag{1.1}
$$

subject to:

1. **Modularity:** Each sparse mechanism can be enabled/disabled.
2. **Multi-architecture:** DistilBERT, RoBERTa, GPT-2.
3. **Pretrained compatibility:** Reuse pretrained Q/K/V/O weights.
4. **Causal correctness:** Preserve causality for decoders.
5. **Training stability:** No collapses.

## 1.3 Our Approach

NEXUS combines six sparse-attention mechanisms in a unified pipeline:

### 1.3.1 Block Partitioning

The sequence is partitioned into $M = \lceil N / B_s \rceil$ blocks of size 
$B_s$ (default 16).

### 1.3.2 XAttention Block Scoring

Sub-sampled queries and keys compute coarse block-to-block scores:

$$
S_{b_i, b_j} = \frac{1}{|B_i| |B_j|} \sum_{i \in B_i, j \in B_j} \frac{q_i \cdot k_j}{\sqrt{d_h}}
\tag{1.2}
$$

Top-$k$ key blocks are retained per query block.

### 1.3.3 SeerAttention Learnable Gating

A learnable bottleneck network predicts per-head block budgets:

$$
g_h = \sigma(W_q \bar{Q}_h \cdot W_k \bar{K}_h^T)
\tag{1.3}
$$

The gate overrides/adds to the XAttention mask.

### 1.3.4 DuSA Dual-Stage

Intra-block (diagonal) is always dense. Inter-block uses top-$k$.

### 1.3.5 AnchorAttention Stripe

For $N > 256$, a stripe band mask restricts fine-grained within-block 
attention to adjacent stripe groups.

### 1.3.6 Softpick

A rectified softmax:

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{x_i/T} - 1)}{\sum_j \text{ReLU}(e^{x_j/T} - 1)}
\tag{1.4}
$$

### 1.3.7 Spectraformer RFF Rectification

Periodically, a global linear attention with RFF features is applied.

## 1.4 Contributions

- **A unified framework** for combining six recent sparse attention 
  mechanisms in a single module.
- **Modular design** with per-component enable/disable flags.
- **Multi-architecture support** for flat, nested, and causal transformers.
- **Honest complexity analysis.** We report which components are active under 
  which conditions.
- **Reproducible ablation.** Each component is ablated independently.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on each of 
the six components. Section 4 presents NEXUS. Section 5 describes 
integration. Section 6 details setup. Section 7 reports results. Section 8 
analyzes per-component contributions. Section 9 discusses trade-offs. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $B_s$ | Block size |
| $M$ | Number of blocks |
| $k$ | Top-$k$ blocks |
| $F$ | Number of RFF features |
| $T$ | Softpick temperature |
| $H$ | Number of heads |
| $d_h$ | Head dimension |

---