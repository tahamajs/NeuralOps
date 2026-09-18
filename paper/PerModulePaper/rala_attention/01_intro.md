<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Standard softmax attention (Vaswani et al., 2017) computes:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V
\tag{1.1}
$$

with $\mathcal{O}(L^2)$ time and memory in sequence length $L$. For long 
sequences (e.g., documents, code, or genomics), this becomes prohibitive.

**Linear attention** (Katharopoulos et al., 2020) replaces the softmax with 
a kernel feature map $\phi$:

$$
\text{LinAttn}(Q, K, V) = \frac{\phi(Q) \left(\phi(K)^\top V\right)}{\phi(Q) \left(\phi(K)^\top \mathbf{1}\right)}
\tag{1.2}
$$

This reorders the computation, yielding $\mathcal{O}(L)$ time and memory. 
The numerator $\phi(K)^\top V \in \mathbb{R}^{m \times d}$ is computed once 
and reused for all queries.

**Limitation.** Linear attention is **head-agnostic**: the interaction between 
query head $h_q$ and key head $h_k$ is determined entirely by the kernel 
feature map, not by a learned interaction matrix. In softmax attention, the 
full $QK^\top$ matrix provides head-specific structure.

0.85

<!-- 
EXAMPLE continuation:
"Several attempts have been made to bridge this gap. Attention with linear 
biases (ALiBi, Press et al., 2022) adds position-dependent biases but not 
head-specific interactions. Relative positional encodings do the same. Neither 
restores the head-interaction structure of softmax attention. RALA does, by 
adding a low-rank learned term M_h = U_h V_h^T ∈ R^{d × d} per head."
-->

## 1.2 Problem Statement

We seek a linear-attention variant $\mathcal{A}'$ such that:

1. **Complexity:** $\mathcal{O}(L)$ time and memory
2. **Expressiveness:** richer than standard linear attention
3. **Stability:** strictly positive kernel map
4. **Fine-tunability:** compatible with LoRA recovery
5. **Causality:** respects strict causality for decoder models

We do **not** target parameter reduction; RALA is a computational substitution, 
not a compression method.

## 1.3 Our Approach

RALA modifies (1.2) as follows:

1. **Softplus kernel.** We use $\phi(x) = \log(1 + e^x)$, which is smooth, 
   strictly positive, and unbounded. Unlike $\mathrm{elu}(x) + 1$, softplus 
   has no kink at $x = 0$.

2. **Rank-$r$ head interaction.** We add a per-head low-rank matrix 
   $M_h = U_h V_h^\top \in \mathbb{R}^{d \times d}$ with $r \ll d$. This 
   term contributes a rank-$r$ correction to the attention output.

3. **Scalar gate.** A single learned scalar $\alpha$, initialized to $0$, 
   controls the augmentation strength. At initialization, RALA reduces 
   exactly to standard linear attention.

## 1.4 Contributions

- **A rank-augmented linear attention formulation** with softplus kernel 
  and scalar gate.
- **A correct multi-architecture implementation** supporting DistilBERT, 
  RoBERTa, GPT-2, and Qwen/LLaMA with exact weight transfer.
- **An honest evaluation** showing the accuracy impact on SST-2 across three 
  models, with explicit reporting of parameter overhead and inference cost.
- **A clear identification** of the code's causal scan as a bottleneck, 
  with recommendations for a fused kernel.

## 1.5 Paper Organization

Section 2 reviews linear attention and rank augmentation. Section 3 provides 
background on kernels, complexity, and LoRA. Section 4 presents RALA. Section 
5 describes integration. Section 6 details the setup. Section 7 reports 
results. Section 8 analyzes accuracy and speed. Section 9 discusses 
practical guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $L$ | Sequence length | §1.1 |
| $d$ | Head dimension | §1.1 |
| $H$ | Number of query heads | §1.1 |
| $H_{kv}$ | Number of KV heads | §3.2 |
| $\phi$ | Kernel feature map | §1.1 |
| $m$ | Feature map output dimension | §3.1 |
| $r$ | Rank of augmentation | §1.3 |
| $U_h, V_h$ | Augmentation factors per head | §1.3 |
| $\alpha$ | Learned augmentation gate | §1.3 |
| $M_h$ | Augmentation matrix $U_h V_h^\top$ | §1.3 |

---