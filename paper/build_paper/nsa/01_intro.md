<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Softmax attention computes an $L \times L$ score matrix, but the resulting 
distribution is **highly sparse**: the top few keys usually receive most 
of the attention mass. Empirical studies (Clark et al., 2019; Voita et al., 
2019; Zhang et al., 2023) consistently find:

- **Local dominance:** Most mass lies within a few tokens of the query.
- **A few global heads:** Some heads attend to distant content, typically 
  across a small number of positions.
- **Content dependence:** Which distant positions matter varies with the 
  input.

This suggests two orthogonal strategies:

1. **Fixed-window attention** (Longformer, Mistral): cheap, local.
2. **Content-based sparse attention** (Routing, Reformer): expensive, 
   content-aware.

**Neither alone is sufficient.** Fixed windows miss global dependencies; 
content-based methods require clustering or hashing overhead.

## 1.2 Problem Statement

Given a pretrained transformer with dense attention, we seek a modified 
module $\mathcal{A}'$ such that:

1. **Local capture:** Nearby tokens are always attended.
2. **Content-aware global:** A small number of distant tokens are selected 
   based on actual scores.
3. **Differentiable:** Selection is differentiable (or at least 
   trainable end-to-end).
4. **Vectorized:** No Python loop over queries.
5. **Quality:** $\Delta\text{Acc} \leq 0.02\%$ on SST-2.
6. **Pretrained compatibility:** Only LoRA + $O(1)$ parameters are trainable.

## 1.3 Our Approach: NSA

**Native Sparse Attention** (NSA) combines a fixed sliding window and a 
dynamic top-$k$ selection. For each query $i$:

$$
\mathcal{S}_i = \mathcal{W}_i \cup \text{TopK}_{j \notin \mathcal{W}_i}(s_{ij})
\tag{1.1}
$$

where $\mathcal{W}_i = \{j : |i - j| \leq w\}$ is the local window and 
$\text{TopK}$ selects the $k$ highest-scoring keys outside the window.

**Key design choices:**

1. **No Python loop.** The top-$k$ selection uses `torch.topk` on a 
   masked score matrix. Fully vectorized.
2. **Union mask.** After top-$k$, we build a boolean mask and apply it 
   before softmax. No special kernels needed.
3. **Causal-aware top-$k$.** For decoders, the top-$k$ is computed 
   **after** causal masking, so it never picks future positions.
4. **Padding-safe.** Padding keys are excluded from top-$k$ selection.

## 1.4 Contributions

- **A hybrid sparse attention module.** NSA combines sliding windows with 
  dynamic top-$k$ content-based routing.
- **Full vectorization.** No clustering, no hashing, no Python loop.
- **A production-hardened implementation.** Supports four architectures 
  with RoPE, GQA, LoRA recovery, and safe injection.
- **A systematic evaluation.** Benchmarked on DistilBERT, RoBERTa, and GPT-2 
  on SST-2 with ablations over window size, top-$k$, and causal handling.
- **Selection analysis.** We show that top-$k$ selected tokens vary with 
  layer depth, revealing content-level head specialization.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on sparse 
attention. Section 4 presents NSA. Section 5 describes integration with four 
architectures. Section 6 details the experimental setup. Section 7 reports 
results. Section 8 analyzes the top-$k$ selection behavior. Section 9 
discusses practical guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $H$ | Number of attention heads |
| $H_{kv}$ | Number of KV heads |
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $w$ | Window size |
| $k$ | Top-$k$ count |
| $\mathcal{W}_i$ | Local window at query $i$ |
| $\mathcal{S}_i$ | Selected keys at query $i$ |
| $s_{ij}$ | Score of query $i$ against key $j$ |
| $\rho$ | Attention density |

---