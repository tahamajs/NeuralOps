<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Standard softmax attention (Vaswani et al., 2017) computes, for each of $H$ 
heads and each pair of positions $(i, j)$, a score:

$$
s_{ij}^{(h)} = \frac{(Q^{(h)} x_i)^\top (K^{(h)} x_j)}{\sqrt{d_h}}
\tag{1.1}
$$

producing an $L \times L$ score matrix per head. This is $\mathcal{O}(L^2 d_h)$ 
in time and $\mathcal{O}(L^2)$ in memory — the dominant cost for long sequences.

Yet the attention distribution is **highly concentrated**. Across numerous 
studies (Clark et al., 2019; Voita et al., 2019; Michel et al., 2019), the 
following has been repeatedly observed:

- Most attention weight is concentrated on a few keys (attention sinks, 
  heavy hitters).
- Many heads are redundant and can be pruned entirely.
- The sparsity pattern varies dramatically across heads.

**Static sparsity** — fixed windows (Longformer), fixed strides (BigBird), 
fixed random patterns — cannot adapt to this variation. A head that needs 
global context is forced into a local window; a head that only needs local 
context wastes capacity on distant keys.

**Dynamic sparsity** — top-$k$ selection per query — adapts to content, but 
existing methods suffer from two problems:

1. **Non-differentiability.** Hard top-$k$ selection has zero gradient 
   almost everywhere, making it unsuitable for end-to-end training.
2. **Uniform budget.** Most methods use the same $k$ for all heads, ignoring 
   the fact that head importance varies.

## 1.2 Problem Statement

Given attention score matrix $S \in \mathbb{R}^{L \times L}$ per head, we 
seek a sparse approximation $\tilde{S}$ such that:

$$
\tilde{S} \approx S, \qquad
\text{supp}(\tilde{S}) \ll L^2
\tag{1.2}
$$

We impose the following constraints:

1. **Differentiability:** End-to-end trainable w.r.t. the sparsity budget.
2. **Per-head adaptation:** Each head has its own sparsity budget.
3. **Quality:** $\Delta\text{Acc} \leq 5\%$ on SST-2.
4. **Memory bounded:** Peak VRAM must not exceed baseline.
5. **Compatibility:** Works with pretrained weights without retraining.

## 1.3 Our Approach: Adaptive Sparse Attention

We propose **Adaptive Sparse Attention** (ASA), which addresses the two 
problems above simultaneously:

1. **Learnable per-head budget.** Each head maintains a scalar logit 
   $\ell_h \in \mathbb{R}$. Its retention budget is:
   $$
   k_h = \lfloor (1 - \sigma(\ell_h)) \cdot L \rfloor
   \tag{1.3}
   $$
   where $\sigma$ is the sigmoid. This makes the budget a **continuous, 
   differentiable** function of a single parameter.

2. **Soft top-$k$ masking.** Hard top-$k$ selection is replaced by a soft 
   mask:
   $$
   \tilde{s}_{ij} = s_{ij} + \log \sigma\!\left(\frac{s_{ij} - \tau_h}{T}\right)
   \tag{1.4}
   $$
   where $\tau_h$ is the $k_h$-th largest score for head $h$ and $T = 0.1$ 
   is a temperature. As $T \to 0$, this converges to hard top-$k$; for finite 
   $T$, it is smooth and differentiable.

3. **Chunked computation.** To bound memory, the query dimension is 
   processed in chunks of 64. This keeps peak VRAM at $\mathcal{O}(B H \cdot 
   \text{chunk} \cdot L)$ rather than $\mathcal{O}(B H L^2)$.

## 1.4 Contributions

- **A learnable per-head sparsity mechanism.** ASA's per-head logits are the 
  first differentiable mechanism for learning head-specific sparsity budgets 
  without auxiliary losses.
- **A soft top-$k$ threshold.** We replace hard top-$k$ with a sigmoid-based 
  soft mask that concentrates mass on the retained entries while remaining 
  differentiable.
- **A production-hardened implementation.** The implementation supports flat, 
  nested, and causal architectures with RoPE, GQA, T5-style relative position 
  bias, and safe injection.
- **A systematic evaluation.** Benchmarked on DistilBERT, RoBERTa, and GPT-2 
  on SST-2 with ablations over sparsity ratio, temperature, and head budget 
  distribution.
- **Head budget analysis.** We show that learned budgets are highly non-uniform, 
  with a consistent bimodal distribution across layers.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on attention and 
sparsity. Section 4 presents ASA. Section 5 describes integration with three 
architectures. Section 6 details the experimental setup. Section 7 reports 
results. Section 8 analyzes head heterogeneity and sparsity patterns. Section 
9 discusses practical guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $H$ | Number of attention heads |
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $s_{ij}^{(h)}$ | Attention score between positions $i, j$ for head $h$ |
| $\ell_h$ | Learnable logit for head $h$ |
| $k_h$ | Retention budget for head $h$ |
| $\tau_h$ | Threshold (k-th largest score) for head $h$ |
| $T$ | Soft-mask temperature |
| $\sigma$ | Sigmoid function |
| $\rho_h$ | Sparsity ratio for head $h$ |

---