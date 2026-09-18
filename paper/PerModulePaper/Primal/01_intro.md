<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

The softmax attention mechanism (Vaswani et al., 2017) scales quadratically 
with sequence length $L$:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

producing an $L \times L$ attention matrix that requires $\mathcal{O}(L^2)$ 
memory and time. For a sequence of 4096 tokens at 12 heads, the attention 
matrix alone occupies **200 MB at FP16**, dwarfing the model's own weight 
footprint.

**Low-rank approximation** reduces this to $\mathcal{O}(L m)$ by projecting 
onto $m \ll L$ landmarks. The Nyström method (Williams & Seeger, 2001) does 
this by:

$$
K \approx \tilde{K} = C W^{-1} C^\top
\tag{1.2}
$$

where $C \in \mathbb{R}^{L \times m}$ is the query-landmark kernel and 
$W \in \mathbb{R}^{m \times m}$ is the landmark-landmark kernel. The 
Nyströmformer (Xiong et al., 2021) applies this to attention.

**The problem with standard Nyström.** It assumes a symmetric kernel, 
requiring the same landmarks for queries and keys. In attention, however, 
queries and keys play different roles: queries ask "what am I looking for?"; 
keys answer "what am I?" Using the same landmarks ignores this asymmetry.

## 1.2 Problem Statement

Given query $Q \in \mathbb{R}^{L \times d}$, key $K \in \mathbb{R}^{L \times d}$, 
and value $V \in \mathbb{R}^{L \times d}$, we seek a low-rank approximation 
of the attention output:

$$
\text{Attn}(Q, K, V) \approx \text{Attn}_\text{low-rank}(Q, K, V)
\tag{1.3}
$$

such that:

1. **Complexity:** $\mathcal{O}(L m)$ instead of $\mathcal{O}(L^2)$, with $m \ll L$.
2. **Asymmetry:** Independent landmark selection for $Q$ and $K$.
3. **Differentiability:** End-to-end training.
4. **Compatibility:** Works with pretrained models without retraining.
5. **Stability:** Numerically robust for FP16 and long sequences.

## 1.3 Our Approach: Primal Attention

We propose **Primal Attention**, which decomposes the attention matrix 
asymmetrically:

$$
A \approx F_1 F_2^{-1} F_3
\tag{1.4}
$$

where:

- $F_1 = Q K_\text{land}^\top \in \mathbb{R}^{L \times m}$: query vs landmark keys
- $F_2 = Q_\text{land} K_\text{land}^\top \in \mathbb{R}^{m \times m}$: landmark-landmark
- $F_3 = Q_\text{land} K^\top \in \mathbb{R}^{m \times L}$: landmark queries vs keys

This is the **primal representation** of the asymmetric kernel (in contrast 
to the dual representation which uses $L \times L$ kernels). The pseudoinverse 
$F_2^{-1}$ is computed via **Schulz iteration**:

$$
X_{t+1} = X_t (2I - F_2 X_t)
\tag{1.5}
$$

which converges quadratically to $F_2^{-1}$ in a few steps and is differentiable.

**Landmark selection.** We use **segment pooling**: divide the sequence into 
$m$ equal chunks and average each chunk. This gives uniform coverage 
regardless of content.

## 1.4 Contributions

- **An asymmetric kernel SVD formulation.** Primal Attention allows independent 
  landmarks for queries and keys, generalizing standard Nyström approximation.
- **A differentiable pseudoinverse.** Schulz iteration provides end-to-end 
  trainable pseudoinverse without explicit eigendecomposition.
- **A production-hardened implementation.** Supports flat, nested, and causal 
  architectures with GQA, RoPE, and safe injection.
- **A systematic evaluation.** Benchmarked on DistilBERT, RoBERTa, and GPT-2 
  on SST-2 with ablations over landmark count and pseudoinverse iterations.
- **Full reproducibility.** Code, checkpoints, and evaluation scripts released 
  under MIT.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on softmax 
attention, Nyström approximation, and pseudoinverse. Section 4 presents 
Primal Attention. Section 5 describes integration with four architectures. 
Section 6 details the experimental setup. Section 7 reports results. Section 
8 analyzes where the method succeeds and fails. Section 9 discusses practical 
guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $m$ | Number of landmarks |
| $Q, K, V$ | Query, key, value matrices |
| $Q_\text{land}, K_\text{land}$ | Landmark summaries |
| $F_1, F_2, F_3$ | Asymmetric kernel factors |
| $X$ | Running pseudoinverse estimate |
| $r$ | LoRA rank (recovery) |
| $\sigma$ | Softmax function |

---
