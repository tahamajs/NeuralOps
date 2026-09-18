<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

The quadratic cost of attention is dominated by the product $QK^\top$ and 
the subsequent multiplication $AV$. Both scale with the number of key/value 
tokens. If we can reduce the effective number of KV tokens from $L$ to 
$L' < L$, both operations scale down: $QK^\top$ becomes $O(L \cdot L')$ 
instead of $O(L^2)$, and $AV$ becomes $O(L \cdot L' \cdot d_h)$.

**Token pruning** methods exploit this by dropping a subset of KV tokens 
at inference time. The question is **which tokens to drop**.

0.85

<!-- 
EXAMPLE continuation:
"The classical answer is to use attention scores as importance: H2O (Zhang 
et al., 2023) keeps tokens whose accumulated attention mass exceeds a 
threshold, and StreamingLLM (Xiao et al., 2024) always pins the first few 
tokens as attention sinks. FASA proposes a different signal: the variance 
of the key representation. The intuition is that tokens with high-variance 
keys have distinctive representations and are likely to be attended to."
-->

## 1.2 Problem Statement

Given keys $K \in \mathbb{R}^{B \times H \times L \times d_h}$ and values 
$V \in \mathbb{R}^{B \times H \times L \times d_h}$, select a subset 
$\mathcal{S} \subset \{1, \ldots, L\}$ with $|\mathcal{S}| = r L$ such that:

$$
\text{Attn}(Q, K_{\mathcal{S}}, V_{\mathcal{S}}) \approx \text{Attn}(Q, K, V)
\tag{1.1}
$$

while computing the left-hand side at a fraction $r$ of the cost.

## 1.3 Our Approach

FASA ranks tokens by the mean-over-heads variance of their key vectors:

$$
\text{imp}_i = \frac{1}{H} \sum_{h=1}^{H} \text{Var}_d(K_{i,h,d})
\tag{1.2}
$$

and retains the top-$rL$ tokens. Index 0 (the CLS token) is always 
preserved, motivated by the attention-sink literature.

**Special cases:**
- **Causal decoders:** FASA does not prune. The causal code path uses full 
  attention to preserve strict lower-triangular masking.
- **Cross-attention:** FASA does not prune. Pruning would break the 
  alignment between query and key positions.

## 1.4 Contributions

- **FASA**, a variance-based token pruning method with a fixed keep ratio.
- **Bidirectional-only application.** We document that causal decoders 
  cannot use FASA without breaking causality.
- **Empirical evaluation** on DistilBERT, RoBERTa, and GPT-2 on SST-2.
- **Full reproducibility**, including code and checkpoints.

## 1.5 Paper Organization

Section 2 reviews token pruning, KV cache compression, and attention sinks. 
Section 3 provides background on attention and variance. Section 4 presents 
FASA. Section 5 covers integration. Section 6 details the setup. Section 7 
reports results. Section 8 analyzes the pruning behavior. Section 9 
discusses. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $L$ | Sequence length | §1.1 |
| $d_h$ | Head dimension | §1.1 |
| $H$ | Number of heads | §1.2 |
| $r$ | Keep ratio | §1.3 |
| $\text{imp}_i$ | Importance score of token $i$ | §1.3 |
| $\mathcal{S}$ | Selected subset | §1.2 |
</parameter>

---