<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer attention has a well-known locality–globality trade-off. 
Local attention (sliding window, block-diagonal) is fast but misses 
long-range dependencies. Global attention (dense, low-rank, kernel-based) 
captures long-range structure but is either expensive or coarse.

**Hybrid attention** resolves this by combining multiple branches:

$$
\text{Output} = \sum_{i} g_i \cdot \text{Branch}_i(X)
\tag{1.1}
$$

The branches cover different ranges and costs, and the gates determine 
how much each contributes.

## 1.2 Our Approach

**NASC** uses three branches with complementary inductive biases:

1. **Local window:** sliding window of size $W = 128$, with 
   strict causal masking.
2. **Sparse block:** partitions the sequence into blocks of size 
   $B_s = 64$ and selects the top-$k$ blocks by mean 
   query-key alignment.
3. **Global spectral RFF:** random Fourier features (RFF) with cos/sin 
   concatenation, computed via state accumulation for $\mathcal{O}(L)$ 
   cost.

The three branches are fused by a **gate** computed from the batch-mean 
hidden state:

$$
g = \text{softmax}(W_g \cdot \bar{x}), \quad \bar{x} = \frac{1}{L} \sum_t x_t
\tag{1.2}
$$

## 1.3 Honest Disclosures

Three implementation details are documented for transparency:

**The gate is batch-level.** All tokens in a batch share the same $g$. 
This is a simplification compared to per-token gating.

**`softpick` uses a batch-level fallback.** The ReLU-based attention 
sparsification falls back to standard softmax when **all** rows in a 
batch have zero denominator. Per-row fallback would be more robust.

**`inject_nasc` raises on failure.** Unlike other modules in our framework 
(e.g., `inject_cosformer`, `inject_fasa`), which return `None` on failure, 
`inject_nasc` re-raises exceptions. This is a design choice but reduces 
robustness in production.

## 1.4 Contributions

- **A three-branch cascade** combining local, sparse-block, and spectral-RFF 
  attention.
- **A batch-level softmax gate** over the three branches.
- **A hardened `softpick`** with batch-level softmax fallback.
- **An empirical evaluation** on DistilBERT, RoBERTa, and GPT-2 on SST-2.
- **Full reproducibility** including the code paths and diagnostic logs.

## 1.5 Paper Organization

Section 2 reviews hybrid attention, sparse block attention, and RFF. 
Section 3 provides background. Section 4 presents NASC. Section 5 covers 
integration. Section 6 details the setup. Section 7 reports results. 
Section 8 analyzes the gates and branch contributions. Section 9 discusses. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $L$ | Sequence length | §1.1 |
| $d$ | Head dimension | §1.1 |
| $H$ | Number of heads | §1.1 |
| $W$ | Window size | §1.2 |
| $B_s$ | Block size (sparse branch) | §1.2 |
| $k$ | Top-k blocks | §1.2 |
| $F$ | RFF features | §1.2 |
| $g$ | Gate vector | §1.2 |
| $\text{softpick}$ | ReLU-based attention | §3.4 |
</parameter>

---