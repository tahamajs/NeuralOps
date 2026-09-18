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

with $\mathcal{O}(L^2)$ time and memory. For $L = 4096$ and batch size 8 with 
12 heads, the intermediate attention matrix alone requires 
$8 \cdot 12 \cdot 4096^2 \cdot 2 \approx 3.2$ GB in FP16 — a hard limit on 
sequence length.

**Sparse attention** methods (Longformer, BigBird, Reformer) reduce this by 
restricting each query to a subset of keys. The subset can be defined by:

- **Position** (sliding window, dilated patterns)
- **Content** (learned hashing, clustering)
- **Random** (as in BigBird's random component)

**Content-based sparsity** is attractive because it adapts to the input: 
similar tokens attend to each other, dissimilar ones do not.

0.85

<!-- 
EXAMPLE continuation:
"Existing content-based methods use locality-sensitive hashing (Reformer, 
Kitaev et al., 2020) or learned permutations (Sparse Transformer, Child et al., 
2019). Both have limitations: LSH requires hand-tuned hashing, and learned 
permutations are not input-dependent. We propose Routing Attention, which 
uses online k-means on the joint query-key representation to assign tokens 
to clusters, then restricts each query to keys within its cluster."
-->

## 1.2 Problem Statement

We seek a sparse attention mechanism $\mathcal{A}'$ such that:

1. **Sparsity:** each query attends to $O(L/K)$ keys on average, where $K$ 
   is the number of clusters
2. **Content-awareness:** cluster assignments depend on the input
3. **Compatibility:** works with pretrained weights and LoRA recovery
4. **Causality:** respects strict causal masking for decoder models

We **do not** claim parameter reduction — this is an architectural 
substitution, not a compression method.

## 1.3 Our Approach

Routing Attention consists of three steps:

1. **Cluster assignment.** Compute a joint representation $Q + K$ 
   (broadcasted as needed), normalize with $\ell_2$, and run $k$-means for 
   `n_iters` iterations. This yields cluster IDs per token.

2. **Cluster-restricted attention.** For each cluster, compute attention 
   among its tokens only. Token 0 is included as a global context anchor.

3. **Self-token preservation.** Each token always attends to itself, 
   regardless of its cluster.

## 1.4 Contributions

- **A content-based sparse attention formulation** using online k-means on 
  the joint query-key representation, with a token-0 context anchor.
- **A multi-architecture implementation** supporting DistilBERT, RoBERTa, 
  GPT-2, and Qwen/LLaMA with exact weight transfer.
- **An honest evaluation** showing that accuracy is preserved within 
  **-26.95%–-28.44%** but that the current implementation does 
  **not** achieve real speedups due to a naive loop over clusters.
- **A clear roadmap** for making routing practical: sparse matmul kernels, 
  dense clustering with differentiable assignment, and a single-pass 
  implementation.

## 1.5 Paper Organization

Section 2 reviews sparse attention methods. Section 3 provides background 
on attention, k-means, and LoRA. Section 4 presents Routing Attention. 
Section 5 describes architecture integration. Section 6 details the setup. 
Section 7 reports results. Section 8 analyzes why the naive implementation 
does not achieve speedups. Section 9 discusses practical guidance. Section 
10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $B$ | Batch size | §1.1 |
| $L$ | Sequence length | §1.1 |
| $H$ | Number of heads | §1.1 |
| $d$ | Head dimension | §1.1 |
| $K$ | Number of clusters | §1.3 |
| $K_{\text{num}}$ | Effective number of clusters ($\min(K, L)$) | §4.2 |
| $n_{\text{iters}}$ | k-means iterations | §4.2 |
| $r$ | LoRA rank | §4.6 |
| $\mathcal{C}_k$ | Set of tokens in cluster $k$ | §4.3 |
</parameter>

---