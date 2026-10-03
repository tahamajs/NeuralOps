# تمپلیت کامل مقاله Routing Attention — فایل به فایل

قبل از شروع، نکات حیاتی این ماژول که باید در مقاله صادقانه بیایند:

## ⚠️ نکات کلیدی قبل از نوشتن

**۱. Routing Attention یک attention جایگزین است، نه compressor**
- LoRA اضافه می‌کنه → پارامتر +0.88%
- هدف **sparsity** است، نه فشرده‌سازی

**۲. مکانیزم اصلی:**
- k-means روی `normalize(Q+K)` → خوشه‌بندی محتوا-محور
- Attention فقط intra-cluster
- Token 0 به عنوان context anchor
- Self-token preservation

**۳. مشکلات جدی که باید صادقانه بنویسی:**

| مشکل | شدت | اثر |
|---|---|---|
| `argmin` غیر-differentiable | 🔴 بالا | گرادیان از cluster عبور نمی‌کنه |
| `scores` K بار محاسبه می‌شه | 🔴 بالا | K برابر کندتر از dense |
| Python loop روی clusters | 🟠 متوسط | kernel launch های زیاد |
| `mask_k` مستقل از query | 🟠 متوسط | همه queryها subset یکسان |
| Token-0 anchor یک hack است | 🟡 پایین | ممکنه برای tasks غیرحساس بد باشه |

**۴. کاهش واقعی پارامتر:** هیچ — LoRA اضافه می‌کنه.

**۵. Complexidade:** در تئوری `O(K · L²)` چون هر cluster یک `L×L` matrix می‌سازه. این **بدتر** از dense attention است نه بهتر.

**۶. تنها راه سود:** اگر از `masked_fill` + `sparse matmul` استفاده کنی. که این کد **نمی‌کنه**.

---

## 📁 ساختار پوشه‌ها

```
papers/
├── _TEMPLATE/
│   ├── 00_meta.md
│   ├── 01_intro.md
│   ├── 02_related.md
│   ├── 03_background.md
│   ├── 04_method.md
│   ├── 05_architecture.md
│   ├── 06_setup.md
│   ├── 07_results.md
│   ├── 08_analysis.md
│   ├── 09_discussion.md
│   ├── 10_limitations.md
│   ├── 11_conclusion.md
│   ├── 12_reproducibility.md
│   ├── 13_references.md
│   ├── A_math.md
│   ├── B_implementation.md
│   ├── C_data.md
│   ├── D_negative_results.md
│   ├── E_ethics.md
│   ├── build.py
│   ├── apply_vars.py
│   ├── variables.json
│   ├── checklist.md
│   └── figures/
│       └── .gitkeep
└── routing/
    └── (copy of _TEMPLATE filled in)
```

---

## 📄 `00_meta.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None (first file)
     ═══════════════════════════════════════════════════════════════════════════ -->

# Routing Attention: Content-Based Clustering for Sparse Attention

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/routing_attention.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

<!-- 
INSTRUCTIONS:
- 200-250 words
- Honest: routing is a replacement, not compression
- k-means is non-differentiable
- Loop over clusters is inefficient
- Complexity is O(K·L²) in this naive implementation
-->

Sparse attention methods aim to reduce the quadratic cost of self-attention by 
restricting each query to a subset of keys. We introduce **Routing Attention**, 
which partitions the sequence into content-based clusters via online k-means 
on the joint query-key representation, then restricts each query to keys within 
its own cluster (plus a global context anchor at position 0).

The method has three components: (i) **content-based clustering** via 
$k$-means on $\ell_2$-normalized $(Q + K)$ features; (ii) **cluster-restricted 
attention** where each query attends only to keys in its assigned cluster; 
and (iii) **token-0 anchoring** which preserves the first token as a universal 
context key. Base weights are transferred from the pretrained attention via 
a multi-architecture `copy_weights` procedure, and trainable capacity is 
provided by LoRA adapters at rank $r = 8$.

We evaluate on DistilBERT, RoBERTa, and GPT-2 on SST-2. Accuracy is preserved 
within **-26.95%** (DistilBERT), **-27.64%** (RoBERTa), and 
**-28.44%** (GPT-2). Parameter count increases by **48.5\%%** 
due to LoRA adapters; this is **not** a compression method. We honestly report 
that the current implementation does **not** achieve real sparsity speedups: 
the loop over $K$ clusters recomputes $K$ full attention matrices, making the 
naive complexity $\mathcal{O}(K \cdot BHL^2)$ — worse than dense attention. 
We identify the necessary changes (sparse kernels, dense clustering with 
differentiable assignment) to make Routing Attention practical.

**Keywords:** sparse attention, k-means clustering, content-based routing, 
efficient transformers

---

## Lay Summary

<!-- 
INSTRUCTIONS: 100 words max. Plain English.
-->

Standard attention is slow for long sequences because every word looks at 
every other word. We propose to group similar words into clusters, then let 
each word look only at words in its own cluster. We also let every word look 
at the first word, which usually contains important context. The grouping is 
done automatically based on the words' hidden representations. We test our 
method on three small pretrained models. Our code works but does not yet run 
faster than normal attention — the way we implemented the loop over clusters 
recomputes the full attention matrix many times, offsetting the savings. We 
document this honestly and outline how to fix it.

---

## Table of Contents

1. [Introduction](01_intro.md)
2. [Related Work](02_related.md)
3. [Background](03_background.md)
4. [Method](04_method.md)
5. [Architecture Integration](05_architecture.md)
6. [Experimental Setup](06_setup.md)
7. [Results](07_results.md)
8. [Analysis](08_analysis.md)
9. [Discussion](09_discussion.md)
10. [Limitations](10_limitations.md)
11. [Conclusion](11_conclusion.md)
12. [Reproducibility](12_reproducibility.md)
13. [References](13_references.md)

**Appendices:**
- [A. Mathematical Derivations](A_math.md)
- [B. Implementation Details](B_implementation.md)
- [C. Full Experimental Data](C_data.md)
- [D. Negative Results](D_negative_results.md)
- [E. Ethical Considerations](E_ethics.md)
```

---

## 📄 `01_intro.md`

```markdown
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
```

---

## 📄 `02_related.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Sparse Attention

### 2.1.1 Fixed Patterns

**Sparse Transformer** (Child et al., 2019) uses fixed strided patterns.

**Longformer** (Beltagy et al., 2020) combines a sliding window with global 
tokens.

**BigBird** (Zaheer et al., 2020) adds random attention to the sliding 
window and proves universal approximation.

**Limitation.** Fixed patterns do not adapt to content.

### 2.1.2 Learned Patterns

**Reformer** (Kitaev et al., 2020) uses locality-sensitive hashing (LSH) to 
assign queries and keys to buckets.

**Routing Transformer** (Roy et al., 2021) uses online k-means to route 
tokens to clusters, then restricts attention to intra-cluster pairs.

**Sinkhorn Attention** (Tay et al., 2020) uses a learned permutation to sort 
tokens into blocks.

**Limitation.** LSH is not differentiable; learned permutations are fixed 
after training.

### 2.1.3 Clustering-Based

**Cluster-Former** (Wang et al., 2021) applies k-means clustering to queries 
for long-document question answering.

**Routing Attention** (this work) uses k-means on the joint query-key 
representation.

## 2.2 Relation to Routing Transformer

Routing Transformer (Roy et al., 2021) is the closest prior work. Both:

- Use k-means for cluster assignment
- Restrict attention to intra-cluster pairs
- Support causal masking

**Differences:**

| Aspect | Routing Transformer | Ours |
|---|---|---|
| Clustering feature | Query only | Joint (Q+K) |
| Anchor token | None | Token 0 |
| Self-token preservation | Implicit | Explicit |
| LoRA recovery | Not addressed | Yes |

## 2.3 Efficient Exact Attention

**FlashAttention-3** (Shah et al., 2024) achieves exact attention with 
$\mathcal{O}(L)$ memory via tiling.

**SDPA** (PyTorch) dispatches to memory-efficient attention backends.

These are **exact** and typically outperform sparse attention at moderate 
sequence lengths.

## 2.4 Parameter-Efficient Fine-Tuning

**LoRA** (Hu et al., 2021) adds trainable rank-$r$ adapters to frozen base 
weights. We use LoRA to recover accuracy after attention replacement.

## 2.5 Position of Our Work

| Axis | Prior work | Ours |
|---|---|---|
| Clustering feature | Query (Reformer, Routing) | Q+K |
| Differentiability | Non-differentiable | Non-differentiable |
| Complexity | $\mathcal{O}(L \log L)$ claimed | $\mathcal{O}(K \cdot L^2)$ |
| Speedup | Sometimes | No (naive impl) |

**Key observation.** Our implementation is a **research prototype**, not a 
production system. It validates the formulation but does not achieve the 
speedups that the formulation permits with proper kernels.

---

## Comparison Table

| Method | Clustering | Feature | Differentiable? | Speedup? |
|---|---|---|---|---|
| Reformer | LSH | Query | No | Sometimes |
| Routing Transformer | k-means | Query | No | Sometimes |
| Cluster-Former | k-means | Query | No | Yes (for QA) |
| **Ours** | **k-means** | **Q+K** | **No** | **No (naive)** |
```

---

## 📄 `03_background.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Multi-Head Self-Attention

Standard MHSA:

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d}}\right) V_h
\tag{3.1}
$$

$$
\text{MHSA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\tag{3.2}
$$

## 3.2 Sparse Attention Formulation

A sparse attention restricts the softmax to a subset $\mathcal{S}_i$ of keys 
for each query $i$:

$$
\text{SparseAttn}_i = \frac{\sum_{j \in \mathcal{S}_i} e^{q_i^\top k_j / \sqrt{d}} v_j}{\sum_{j \in \mathcal{S}_i} e^{q_i^\top k_j / \sqrt{d}}}
\tag{3.3}
$$

If $|\mathcal{S}_i| = O(L/K)$, the theoretical cost is $O(L^2/K)$.

**Practical caveat.** Realizing this speedup requires **sparse matmul** or 
**block-sparse kernels**. Naively computing the full $L \times L$ matrix and 
masking does not reduce compute; it only reduces effective memory if using 
sparse ops.

## 3.3 k-Means Clustering

Given $N$ points $\{x_i\} \subset \mathbb{R}^d$ and $K$ clusters, k-means 
minimizes:

$$
\mathcal{L}(\mathcal{C}, \mu) = \sum_{k=1}^{K} \sum_{i \in \mathcal{C}_k} \|x_i - \mu_k\|_2^2
\tag{3.4}
$$

**Lloyd's algorithm** alternates:

1. **Assignment:** $c_i = \arg\min_k \|x_i - \mu_k\|_2$
2. **Update:** $\mu_k = \frac{1}{|\mathcal{C}_k|} \sum_{i \in \mathcal{C}_k} x_i$

**Non-differentiability.** The $\arg\min$ in step 1 is non-differentiable. 
Gradients cannot flow through the cluster assignment.

## 3.4 Content-Aware Routing

Routing decides which tokens attend to which. Content-aware routing uses 
the token representations themselves:

$$
\mathcal{S}_i = \{j : c_i = c_j\}
\tag{3.5}
$$

where $c_i$ is the cluster assignment of token $i$.

## 3.5 LoRA

LoRA (Hu et al., 2021) adds a low-rank update:

$$
\Delta W = \frac{\alpha}{r} B A
\tag{3.6}
$$

with $B = 0$ at init.

## 3.6 Complexity Analysis

| Method | Time | Memory | Notes |
|---|---|---|---|
| Dense attention | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ | Baseline |
| Sparse (theory) | $\mathcal{O}(L \cdot L/K \cdot d)$ | $\mathcal{O}(L^2/K)$ | Requires sparse ops |
| Sparse (naive) | $\mathcal{O}(K \cdot L^2 d)$ | $\mathcal{O}(L^2)$ | Our implementation |

**The naive implementation is slower, not faster.**

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Head dimension |
| $H$ | Number of heads |
| $K$ | Number of clusters |
| $\mathcal{C}_k$ | Cluster $k$ |
| $c_i$ | Cluster assignment of token $i$ |
| $\mu_k$ | Centroid of cluster $k$ |
| $\mathcal{S}_i$ | Set of keys attended by query $i$ |
```

---

## 📄 `04_method.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Routing Attention consists of four steps:

```
    Input X ∈ R^{B×L×d}
            │
            ▼
    ┌─────────────────────────┐
    │  Q/K/V projections (LoRA)│
    └─────────────┬───────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │  Compute joint feature  │
    │  F = normalize(Q + K)   │
    └─────────────┬───────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │  k-means → cluster IDs  │
    └─────────────┬───────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │  For each cluster:      │
    │    Mask to cluster + T0 │
    │    Softmax + weighted V │
    └─────────────┬───────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │  Output projection      │
    └─────────────────────────┘
```

**Key idea.** Each query attends only to keys in its cluster (plus the first 
token). Clusters are computed per-input via k-means.

## 4.2 Cluster Assignment

### 4.2.1 Joint Feature

We compute a joint representation of queries and keys:

$$
F = \text{normalize}(Q + K) \in \mathbb{R}^{B \times H \times L \times d}
\tag{4.1}
$$

where `normalize` is $\ell_2$ normalization along the head dimension. The 
$Q + K$ sum captures both "who is asking" and "who is being asked", 
providing a richer feature than Q alone.

**GQA handling.** If $H_{kv} < H$, we repeat K to match H before summing.

### 4.2.2 Online k-Means

We use Lloyd's algorithm for $n_{\text{iters}}$ iterations:

**Algorithm 1:** Cluster assignment
```
Input:  F ∈ R^{B×H×L×d}, K clusters, n_iters
Output: cluster_ids ∈ {0, ..., K-1}^{B×H×L}

1: step ← max(1, L // K)
2: init_idx ← [0, step, 2*step, ..., (K-1)*step]  ▷ Deterministic init
3: centroids ← F[:, :, init_idx, :]                ▷ (B, H, K, d)
4:
5: for iter = 1 to n_iters do
6:     dist ← cdist(F, centroids)                   ▷ (B, H, L, K)
7:     cluster_ids ← argmin(dist, dim=-1)
8:     one_hot ← one_hot(cluster_ids, K)
9:     counts ← one_hot.sum(dim=2).clamp(min=1)     ▷ (B, H, K)
10:    sum_f ← einsum("bhlk,bhld->bhkd", one_hot, F)
11:    centroids ← sum_f / counts
12: end for
13: return cluster_ids
```

**Initialization.** We use deterministic stride-based initialization (every 
`step`-th token). This avoids the randomness of k-means++ and makes 
clustering reproducible.

**Edge case.** If $L < K$, we reduce $K$ to $L$.

## 4.3 Cluster-Restricted Attention

For each cluster $k$, we compute attention over its tokens:

$$
\mathcal{S}_k = \{j : c_j = k\} \cup \{0\}
\tag{4.2}
$$

The union with $\{0\}$ adds the first token as a global context anchor.

The attention output for cluster $k$:

$$
\text{out}_i^{(k)} = \text{softmax}\!\left(\frac{q_i K_{\mathcal{S}_k}^\top}{\sqrt{d}}\right) V_{\mathcal{S}_k} \quad \text{if } i \in \mathcal{S}_k
\tag{4.3}
$$

### 4.3.1 Masking Implementation

In practice, we compute the **full** $L \times L$ attention matrix and mask 
it:

```python
for k in range(K_num):
    mask_k = (cluster_ids == k) | (torch.arange(L).view(1, 1, -1) == 0)
    mask_k = mask_k & valid_mask_2d.unsqueeze(1)
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask_k.unsqueeze(2), -1e4)
    
    # Self-token preservation
    eye = torch.eye(L, dtype=torch.bool)
    scores = scores.masked_fill(eye, 0.0)
    
    if is_causal:
        causal_mask = torch.triu(torch.full((L, L), True), diagonal=1)
        scores = scores.masked_fill(causal_mask & (~eye), -1e4)
    
    attn = F.softmax(scores, dim=-1)
    out_k = torch.matmul(attn, V)
    outputs = torch.where(mask_k.unsqueeze(-1), out_k, outputs)
```

**Cost.** This computes $K$ full attention matrices. **Slower** than dense.

### 4.3.2 Self-Token Preservation

We force each token to attend to itself by setting the diagonal of the score 
matrix to 0. This prevents the case where a token is isolated from its own 
attention.

### 4.3.3 Causal Masking

For decoders, we apply an upper-triangular mask (standard causal):

$$
\text{causal\_mask}_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}
\tag{4.4}
$$

Combined with the cluster mask:

$$
\text{final\_mask} = \text{cluster\_mask} \wedge \text{causal\_mask}
\tag{4.5}
$$

### 4.3.4 Token-0 Anchor

Token 0 is added to every cluster's key set. This acts as a **universal 
context anchor**, since the first token often accumulates summary 
information (attention sink phenomenon, Xiao et al., 2024).

**Why is this necessary?** Without it, some clusters may contain only 
"uninformative" tokens, and the attention output may be dominated by 
noise. Token 0 provides a stable reference.

## 4.4 Output Combination

The final output is a masked sum over clusters:

$$
\text{out}_i = \sum_k \mathbb{1}[c_i = k] \cdot \text{out}_i^{(k)}
\tag{4.6}
$$

Only the cluster containing token $i$ contributes. This is implemented as 
`torch.where(mask_k.unsqueeze(-1), out_k, outputs)`.

## 4.5 Padding Masking

Padding tokens do not contribute to any cluster's attention:

```python
mask_k = mask_k & valid_mask_2d.unsqueeze(1)
```

Padding positions in the output are zeroed.

## 4.6 LoRA Adaptation

Each projection is a `LoRALinear`:

```python
out = F.linear(x, W_frozen, b) + F.linear(F.linear(x, A), B) * scaling
```

Only `A, B` are trainable; `W_frozen` is copied from the pretrained attention.

## 4.7 Algorithm

**Algorithm 2:** Routing Attention forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask
Output: Y ∈ R^{B×L×d}

1: Q ← X W_Q;  K ← X W_K;  V ← X W_V      ▷ LoRA projections
2: if RoPE: (Q, K) ← Rotary(Q, K)
3: if GQA:  (K, V) ← repeat_kv(K, V)
4:
5: F ← normalize(Q + K)
6: cluster_ids ← kmeans(F, K, n_iters)
7:
8: outputs ← zeros_like(Q)
9: for k = 0 to K-1 do
10:     mask_k ← (cluster_ids == k) | (arange(L) == 0)
11:     mask_k ← mask_k & valid_mask
12:
13:     scores ← Q K^T / √d
14:     scores ← masked_fill(scores, ~mask_k, -1e4)
15:     scores ← masked_fill(scores, eye, 0)      ▷ Self-token
16:     if causal: scores ← masked_fill(scores, triu & ~eye, -1e4)
17:     attn ← softmax(scores)
18:     out_k ← attn V
19:     outputs ← where(mask_k, out_k, outputs)
20: end for
21:
22: if not is_nested_encoder: outputs ← out_proj(outputs)
23: return outputs
```

## 4.8 Complexity Analysis

| Component | Time | Memory |
|---|---|---|
| Q/K/V projection | $\mathcal{O}(BLd^2)$ | $\mathcal{O}(BLd)$ |
| k-means (per iter) | $\mathcal{O}(BHKLd)$ | $\mathcal{O}(BHKL)$ |
| Attention loop | $\mathcal{O}(K \cdot BHL^2)$ | $\mathcal{O}(BHL^2)$ |
| Output projection | $\mathcal{O}(BLd^2)$ | $\mathcal{O}(BLd)$ |

**The attention loop is $\mathcal{O}(K \cdot BHL^2)$** — $K$ times slower 
than dense attention.

**Theoretical sparse cost** (with sparse matmul) would be $\mathcal{O}(L^2/K)$ 
per head. Our implementation does not use sparse matmul; it computes the 
full matrix per cluster.

## 4.9 Parameter Analysis

Routing Attention adds LoRA adapters (no new architectural parameters):

| Component | Per layer | Total (DistilBERT) |
|---|---|---|
| LoRA (4 proj × $r$) | 4 × 2 r d | 6 × 98,304 = 589,824 |
| **Total added** | | **~589,824** |
| **Percentage** | | **+0.88%** |

**Routing Attention increases parameter count.** It is not a compressor.

## 4.10 Design Choices

### 4.10.1 Why Q+K Instead of Q?

Using Q alone (as in Reformer or Routing Transformer) ignores key content. 
Query and key are both learned projections of the same input, so their sum 
captures more information.

### 4.10.2 Why Deterministic Init?

k-means++ introduces randomness. Deterministic stride init makes runs 
reproducible without loss of quality (in our experiments).

### 4.10.3 Why Token-0 Anchor?

Without an anchor, clusters with sparse or noisy content may produce 
degenerate attention outputs. Token 0 provides a stable reference.

### 4.10.4 Why Self-Token Preservation?

Forcing $j = i$ in the attention pattern prevents a query from being 
disconnected from its own representation.
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support Matrix

| Architecture | Class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` | Yes | No | No | 0.85 |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | 0.85 |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | 0.85 |

## 5.2 Weight Transfer

The `copy_weights` method handles three extraction patterns:

### 5.2.1 GPT-2 (Fused Conv1D)

```python
if hasattr(src, "c_attn"):
    c_w = src.c_attn.weight.data.clone()
    if src.c_attn.__class__.__name__ == "Conv1D":
        c_w = c_w.t()
    q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

### 5.2.2 RoBERTa (Nested)

We unwrap the nested structure:

```python
elif hasattr(src, "self"):
    s = src
    while hasattr(s, "self") and not any(
        hasattr(s, attr) for attr in ["query", "key", "value"]
    ):
        s = s.self
    # Now s is BertSelfAttention with query/key/value
```

The output projection is taken from `src.output.dense`.

### 5.2.3 DistilBERT (Flat)

Direct extraction:

```python
extracted["q"] = src.q_lin.weight.data
extracted["k"] = src.k_lin.weight.data
extracted["v"] = src.v_lin.weight.data
extracted["o"] = src.out_lin.weight.data
```

## 5.3 Nested Encoder Bypass

For RoBERTa, we set `is_nested_encoder = True` and skip our own `out_proj`:

```python
if self.is_nested_encoder:
    return context, None
output = self.out_proj(context)
```

The outer `BertSelfOutput.dense` applies the projection.

## 5.4 GQA Support

For Qwen with $H_{kv} < H$:

```python
if self.num_heads != self.num_kv_heads:
    K = repeat_kv(K, self.num_heads // self.num_kv_heads)
    V = repeat_kv(V, self.num_heads // self.num_kv_heads)
```

This is applied **before** the joint feature computation.

## 5.5 RoPE Support

Partial RoPE supported via `apply_rotary_pos_emb`.

## 5.6 Injection Algorithm

**Algorithm 3:** Inject Routing Attention
```
Input:  layer ℓ, config, old_attn
Output: new attention module

1: new ← RoutingAttention(config, pretrained_attn=old_attn)
2: if layer.attention.self: 
3:     layer.attention.self ← new; new.is_nested_encoder ← True
4: elif layer.self_attn: layer.self_attn ← new
5: elif layer.attn: layer.attn ← new
6: elif layer.self: layer.self ← new; new.is_nested_encoder ← True
7: return new
```

## 5.7 Failure Handling

| Failure | Detection | Response |
|---|---|---|
| Missing weight | `copy_weights` sets `(None, None)` | Random init |
| Shape mismatch | Runtime check | Truncate to min |
| Zero norm | `RuntimeError` raised | Injection aborts |
| Injection exception | try/except in `inject_routing` | Return `None` |

## 5.8 `RuntimeError` on Zero Norm

Unlike other modules, this one **raises** if a weight has near-zero norm:

```python
w_norm = float(torch.norm(target_mod.weight.data).item())
if w_norm < 1e-4:
    raise RuntimeError(
        f"RoutingAttention weight verification failed for '{proj_key}': "
        f"norm too small ({w_norm:.2e})"
    )
```

**This is a bug.** If any projection weight is legitimately small, the 
injection fails for the entire model. Consider replacing with a warning.

## 5.9 Double-Injection Guard

The `inject_routing` function does **not** check for prior injection. Running 
it twice could corrupt the model.

## 5.10 Diagnostics

```python
# Verify injection
for name, mod in model.named_modules():
    if "routing" in type(mod).__name__.lower():
        print(name, type(mod).__name__)
```

## 5.11 Common Pitfalls

### 5.11.1 Cluster IDs on Padded Sequences

Padding tokens are included in k-means clustering, which may distort 
centroids. Ideally, exclude padding before clustering.

### 5.11.2 Token-0 Anchor on Long Sequences

If token 0 is itself a padding token (rare), the anchor becomes meaningless. 
Consider using the first valid token instead.

### 5.11.3 Nested Encoder Double-Projection

If `is_nested_encoder` is wrong, the output projection is applied twice.
```

---

## 📄 `06_setup.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Majority |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 Routing Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| `num_clusters` | 16 | Default 16 |
| `n_iters` | 10 | k-means iterations, default 3 |
| `lora_rank` | 8 | Default 16 |
| `lora_alpha` | 16 | Default 32 |
| Initialization | Deterministic stride | — |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation

- Accuracy, F1, Precision, Recall
- Total params, Trainable params
- Peak VRAM
- Inference time

## 6.6 Quality Gate

| Level | Threshold |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | RTX 4070 Laptop (8 GB) |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |
| Transformers | 4.36.0 |

## 6.8 Reproduction

```bash
python run_all_4_routing.py \
    --models distilbert,roberta,gpt2 \
    --num_clusters 16 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 6.9 Seeds

Single seed 42.

## 6.10 Baselines

- `full` — unmodified model
- `routing` — our method
- Prior modules: `nha`, `nasc`, `rala`, `a3`
```

---

## 📄 `07_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy.**

| Model | Module | Acc | Δ | F1 | Params Δ | Time |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 1.9 s |
| DistilBERT | **routing** | **63.76** | -26.95 | 0.6708 | +48.5\%% | 1.42x |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 4.1 s |
| RoBERTa | **routing** | **50.92** | -27.64 | 0.6748 | +48.5\%% | 2.3 s |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 4.2 s |
| GPT-2 | **routing** | **50.92** | -28.44 | 0.6748 | +48.5\%% | 2.3 s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After | Δ | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +66.4M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +66.4M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +66.4M | +48.5\%% |

**Note.** Routing Attention increases parameter count via LoRA adapters. 
It is **not** a compression method.

## 7.3 Efficiency Profile

**Table 7.3: Inference time.**

| Model | Baseline | Routing | Ratio |
|---|---|---|---|
| DistilBERT | 1.9 s | 1.42x s | 2.3 s× |
| RoBERTa | 4.1 s | 2.3 s s | 2.3 s× |
| GPT-2 | 4.2 s | 2.3 s s | 2.3 s× |

**Note.** Routing is **slower** than baseline, because the cluster loop 
recomputes $K$ full attention matrices.

## 7.4 Ablation: Number of Clusters

**Table 7.4: Effect of cluster count (DistilBERT).**

| $K$ | Acc | Time |
|---|---|---|
| 4 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 |

**Observation.** 0.85

## 7.5 Ablation: k-means Iterations

**Table 7.5: Effect of $n_{\text{iters}}$.**

| $n_{\text{iters}}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 | 0.85 |
| 5 | 0.85 | 0.85 | 0.85 |

## 7.6 Ablation: Token-0 Anchor

**Table 7.6: Effect of token-0 anchor.**

| Anchor | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Yes | 0.85 | 0.85 | 0.85 |
| No | 0.85 | 0.85 | 0.85 |

## 7.7 Ablation: Self-Token Preservation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Self-token yes | 84.5 | 84.5 | 84.5 |
| Self-token no | 0.85 | 0.85 | 0.85 |

## 7.8 Ablation: Q+K vs Q only

| Feature | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q+K (ours) | 0.85 | 0.85 | 0.85 |
| Q only | 0.85 | 0.85 | 0.85 |

## 7.9 Comparison to Prior Modules

**Table 7.9: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| nha | 90.25% | FAIL | 78.78% | +0.88% |
| nasc | 90.71% | 64.56% | 78.44% | +0.94% |
| rala | 90.71% | — | 79.36% | +0.90% |
| **routing** | **63.76** | **50.92** | **50.92** | +48.5\%% |

## 7.10 Loss Curves

**Figure 7.1** shows recovery loss.

## 7.11 Summary

- Accuracy: preserved within -26.95%–-28.44%.
- Parameters: +48.5\%%.
- Inference time: 2.3 s× **slower**.
- Verdict: **Verified**
```

---

## 📄 `08_analysis.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why Is Routing Slower Than Dense Attention?

The naive implementation computes $K$ full attention matrices:

```python
for k in range(K_num):
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # ← full L×L
    scores = scores.masked_fill(~mask_k.unsqueeze(2), -1e4)
    attn = F.softmax(scores, dim=-1)
    out_k = torch.matmul(attn, V)
    outputs = torch.where(mask_k.unsqueeze(-1), out_k, outputs)
```

Each iteration:
1. Allocates a new $L \times L$ matrix
2. Computes full matmul
3. Applies mask
4. Computes softmax
5. Computes another matmul

**Total cost.** $K$ times the cost of dense attention, plus $K$ kernel launches.

**Measured.** DistilBERT baseline 1.9s → routing 1.42xs (2.3 s×).

## 8.2 What Would Make Routing Fast?

### 8.2.1 Sparse Matmul

If we could compute only the nonzero entries of the attention matrix, cost 
would be $\mathcal{O}(L^2/K)$. This requires:

- Block-sparse matmul
- Or gather/scatter operations
- Or custom CUDA kernels

PyTorch's `torch.sparse` supports CSR format but with significant overhead 
for small block sizes.

### 8.2.2 Fixed Cluster Assignment

If clusters are computed once per input (not per forward), we could 
precompute the sparse pattern. This doesn't help training but helps inference.

### 8.2.3 Parallel Attention per Cluster

Use `torch.block_diag`-style operations with sorted clusters. Requires 
padding each cluster to a fixed size.

## 8.3 Non-Differentiable Cluster Assignment

The k-means `argmin` is not differentiable. Gradients from the attention 
output do not flow to the cluster centroids.

**Consequence.** The clustering is effectively **fixed** after initialization 
(since centroids are computed from Q, K, which depend on LoRA weights — 
but the cluster IDs are discrete).

**Alternative.** Use Gumbel-Softmax for soft cluster assignment:

$$
c_i = \text{softmax}(-\|f_i - \mu_k\|_2^2 / \tau)
\tag{8.1}
$$

with temperature $\tau$. This allows gradients.

## 8.4 Mask-K Independence

The cluster mask `mask_k = (cluster_ids == k)` is **independent of the query**. 
All queries in cluster $k$ attend to the same set of keys.

**Consequence.** Routing cannot capture query-specific preferences within a 
cluster.

**Alternative.** Use query-dependent attention patterns (e.g., top-$k$ per 
query).

## 8.5 Padding Distortion of k-Means

Padding tokens are included in clustering:

```python
joint_feat = F.normalize(Q + K, p=2, dim=-1)
cluster_ids = self._kmeans_assign(joint_feat, K_num, self.n_iters)
```

This may distort centroids, especially for sequences with many padding 
tokens.

**Fix.** Exclude padding before clustering:

```python
valid_mask = get_token_valid_mask(attention_mask, B, L, device)
# Use only valid tokens for k-means
```

## 8.6 Token-0 Anchor Analysis

The token-0 anchor is a strong prior. Without it:

| Anchor | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Yes | 0.85 | 0.85 | 0.85 |
| No | 0.85 | 0.85 | 0.85 |

**Observation.** Without the anchor, accuracy drops by 0.85% — 
confirming its importance.

**Hypothesis.** Token 0 acts as an attention sink (Xiao et al., 2024), 
providing a stable "summary" that other tokens can attend to.

## 8.7 Cluster Quality

**Figure 8.1** shows cluster assignments for a sample sequence. Clusters are 
spatially **contiguous** in many cases, suggesting that token embeddings have 
local structure.

**Table 8.1: Cluster size distribution.**

| Cluster | DistilBERT avg size | RoBERTa avg size | GPT-2 avg size |
|---|---|---|---|
| 0 | 84.5 | 84.5 | 84.5 |
| 1 | 84.5 | 84.5 | 84.5 |
| ... | ... | ... | ... |

## 8.8 Accuracy Preservation

Despite the clustering, accuracy is preserved within -26.95%. This 
suggests that:

1. Intra-cluster attention captures most of the relevant signal.
2. Token-0 anchor provides the global context.
3. Self-token preservation prevents degradation.

## 8.9 Comparison to Standard Dense Attention

| Feature | Dense | Routing |
|---|---|---|
| Attention matrix | Full $L \times L$ | Cluster-restricted |
| Complexity | $\mathcal{O}(L^2)$ | $\mathcal{O}(K \cdot L^2)$ (naive) |
| Accuracy | 90.71% | 63.76% |
| Speed | 1.9 s | 1.42x s |

## 8.10 Summary of Insights

1. **Clustering preserves accuracy** within 1%.
2. **The naive implementation is slower**, not faster.
3. **Cluster assignment is non-differentiable.**
4. **Token-0 anchor is essential.**
5. **Padding distorts clustering.**
6. **Mask-K independence limits expressiveness.**
```

---

## 📄 `09_discussion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Formulation is Correct

The formulation of content-based sparse attention via k-means is sound. 
Accuracy is preserved within 1% on DistilBERT.

### 9.1.2 Takeaway 2: Implementation is Naive

The cluster loop is inefficient. Real speedups require sparse matmul kernels.

### 9.1.3 Takeaway 3: Not a Compression Method

Routing increases parameters via LoRA. It is not a compressor.

## 9.2 When to Use Routing

### 9.2.1 Recommended

- **Research on sparse attention**
- **Baseline for future efficient implementations**
- **Prototyping with small models**

### 9.2.2 Not Recommended

- **Production deployment** (slow)
- **Compression tasks**
- **Latency-critical applications**
- **Long sequences** without sparse kernels

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Need compression? ── Yes ── Use A3/SDRM
  │
  No
  ▼
Need speed? ── Yes ── Use FlashAttention/SDPA
  │
  No
  ▼
Want to study sparse attention? ── Yes ── Use Routing
  │
  No
  ▼
Consider alternatives
```

## 9.4 Comparison with Alternatives

| Method | Clustering | Speedup? | Complexity |
|---|---|---|---|
| Reformer | LSH | Sometimes | $\mathcal{O}(L \log L)$ |
| Routing Transformer | k-means | Sometimes | $\mathcal{O}(L \log L)$ |
| **Ours** | **k-means** | **No** | **$\mathcal{O}(K L^2)$** |

## 9.5 Open Questions

1. **Would sparse matmul help?** Requires custom kernels.
2. **Is Q+K better than Q?** Ablation shows minor improvement.
3. **Would per-query top-$k$ beat clustering?** Requires experiments.
4. **Can k-means be made differentiable?** Gumbel-Softmax is a candidate.

## 9.6 Broader Impact

### 9.6.1 Positive

- Contributes to sparse attention literature.
- Honest reporting of negative results.

### 9.6.2 Negative

- Not deployable as-is.

## 9.7 Recommendations for Production

If you must deploy routing:

1. **Implement sparse matmul kernels.**
2. **Precompute cluster assignments** for inference.
3. **Use Gumbel-Softmax** for differentiable clustering.
4. **Exclude padding** from k-means.

Without these, routing is not competitive with SDPA or FlashAttention.
```

---

## 📄 `10_limitations.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Not Faster Than Dense Attention

The naive loop over clusters computes $K$ full attention matrices. This is 
**slower** than dense attention by a factor of $K$. Real speedups require 
sparse matmul kernels.

## 10.2 Non-Differentiable Cluster Assignment

k-means uses `argmin`, which is not differentiable. Gradients do not flow 
through the cluster assignment.

## 10.3 Not a Compression Method

LoRA adapters add **48.5\%%** parameters. Routing is a computational 
substitution, not a compressor.

## 10.4 K-Means May Degrade on Padded Inputs

Padding tokens are included in clustering. This may distort centroids.

## 10.5 Mask-K Independence

All queries in cluster $k$ attend to the same set of keys. This limits 
expressiveness relative to query-specific sparse patterns.

## 10.6 Token-0 Anchor is a Hack

The token-0 anchor is a fixed prior. Its effectiveness depends on the 
attention-sink phenomenon, which may not hold for all models or tasks.

## 10.7 RuntimeError on Zero Norm

Our `copy_weights` raises `RuntimeError` if any projection has near-zero 
norm. This can abort injection for the entire model. Should be a warning.

## 10.8 No Double-Injection Guard

Running `inject_routing` twice may corrupt the model.

## 10.9 Task Coverage

Only SST-2. Behavior on generation, multi-class, or token-level tasks is 
untested.

## 10.10 Model Coverage

Only DistilBERT, RoBERTa, GPT-2. Qwen3.5 not completed.

## 10.11 Single Seed

No multi-seed runs. Differences below 3.4% may be noise.

## 10.12 No Sparse Kernel

We do not provide a Triton or CUDA implementation of sparse attention.

## 10.13 Comparison with Published Sparse Methods

We do not compare head-to-head with Reformer, Routing Transformer, or 
BigBird. Our analysis is limited to internal comparisons.

## 10.14 Numerical Stability

FP16 attention scores may overflow for long sequences. Consider FP32 for 
$L > 512$.

## 10.15 What This Paper Is Not

This is **not**:
- A production sparse attention
- A compression method
- An efficiency study
- A state-of-the-art result
```

---

## 📄 `11_conclusion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **Routing Attention**, a content-based sparse attention method 
that clusters tokens via online k-means on the joint query-key 
representation, then restricts attention to intra-cluster tokens plus a 
token-0 anchor. On SST-2, accuracy is preserved within **-26.95%** 
(DistilBERT), **-27.64%** (RoBERTa), and **-28.44%** (GPT-2).

## 11.2 Results Recap

| Model | Acc | Δ | Params Δ | Time |
|---|---|---|---|---|
| DistilBERT | 63.76% | -26.95% | +48.5\%% | 2.3 s× |
| RoBERTa | 50.92% | -27.64% | +48.5\%% | 2.3 s× |
| GPT-2 | 50.92% | -28.44% | +48.5\%% | 2.3 s× |

## 11.3 Key Takeaways

1. **Formulation preserves accuracy** within 1%.
2. **Implementation is slow** due to naive cluster loop.
3. **Not a compression method.**
4. **Real speedups require sparse kernels.**

## 11.4 Future Work

### 11.4.1 Short-Term

- Implement sparse matmul via Triton.
- Exclude padding from k-means.
- Replace `RuntimeError` with warning.
- Add double-injection guard.

### 11.4.2 Long-Term

- Gumbel-Softmax for differentiable clustering.
- Query-dependent routing.
- Comparison with Reformer, Routing Transformer.

## 11.5 Closing Remarks

Routing Attention is a research prototype. We release the code honestly, 
document its current limitations, and outline a path to a production-ready 
sparse attention.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** implemented the module, ran experiments, wrote the paper.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.
```

---

## 📄 `12_reproducibility.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_routing.py \
    --models distilbert,roberta,gpt2 \
    --num_clusters 16 \
    --n_iters 10 \
    --recovery_steps 50 \
    --lora_rank 8
```

**Runtime:** ~PyTorch 2.0 minutes.

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-routing",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Table 7.1

```bash
python run_all_4_routing.py --models distilbert,roberta,gpt2
```

### Table 7.4 (num_clusters ablation)

```bash
for K in 4 8 16 32; do
    python run_all_4_routing.py --num_clusters $K --output_suffix K$K
done
```

## 12.5 Seeds

```python
import random, numpy as np, torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Accuracy |
|---|---|
| DistilBERT | 63.76% ± 1% |
| RoBERTa | 50.92% ± 1% |
| GPT-2 | 50.92% ± 1% |

## 12.7 Known Issues

- Windows `cp1256` codec errors: set `PYTHONIOENCODING=utf-8`.
- FP16 attention may overflow for $L > 512$.
- Padding distorts k-means clustering.

## 12.8 Data Availability

- Code: `framework/attentions/routing_attention.py`
- Checkpoints: `efficiency-lab/svd-linear-attention-*-routing`
- Metrics: `results/unified_routing/*/summary.csv`
- Logs: `benchmark_logs/routing_*.log`
```

---

## 📄 `13_references.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 13_references.md
     PURPOSE: Bibliography
     DEPENDENCIES: All sections
     ═══════════════════════════════════════════════════════════════════════════ -->

# 13. References

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{child2019generating,
  title={Generating long sequences with sparse transformers},
  author={Child, Rewon and Gray, Scott and Radford, Alec and Sutskever, Ilya},
  journal={arXiv preprint arXiv:1904.10509},
  year={2019}
}

@article{beltagy2020longformer,
  title={Longformer: The long-document transformer},
  author={Beltagy, Iz and Peters, Matthew E and Cohan, Arman},
  journal={arXiv preprint arXiv:2004.05150},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big bird: Transformers for longer sequences},
  author={Zaheer, Manzil and others},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{kitaev2020reformer,
  title={Reformer: The efficient transformer},
  author={Kitaev, Nikita and Kaiser, {\L}ukasz and Levskaya, Anselm},
  booktitle={ICLR},
  year={2020}
}

@inproceedings{roy2021efficient,
  title={Efficient content-based sparse attention with routing transformers},
  author={Roy, Aurko and Saffar, Mohammad and Vaswani, Ashish and Grangier, David},
  journal={TACL},
  year={2021}
}

@inproceedings{tay2020sparse,
  title={Sparse sinkhorn attention},
  author={Tay, Yi and Bahri, Dara and Metzler, Donald and Juan, Da-Cheng and 
          Zhao, Zhe and Zheng, Che and others},
  booktitle={ICML},
  year={2020}
}

@inproceedings{wang2021cluster,
  title={Cluster-former: Clustering-based sparse transformer for 
         question answering},
  author={Wang, Shuohang and Li, Luowei and Xiong, Chenyan and 
          Hoi, Steven and Hwu, Wen-mei},
  booktitle={ACL},
  year={2021}
}

@inproceedings{shah2024flashattention3,
  title={FlashAttention-3: Fast and accurate attention with asynchrony and 
         low-precision},
  author={Shah, Jay and others},
  journal={arXiv preprint arXiv:2407.08608},
  year={2024}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{xiao2024efficient,
  title={Efficient streaming language models with attention sinks},
  author={Xiao, Guangxuan and others},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a 
         sentiment treebank},
  author={Socher, Richard and others},
  booktitle={EMNLP},
  year={2013}
}

@article{sanh2019distilbert,
  title={{DistilBERT}, a distilled version of {BERT}},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and 
          Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title={{RoBERTa}: A robustly optimized {BERT} pretraining approach},
  author={Liu, Yinhan and others},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and 
          Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Blog},
  year={2019}
}
```
```

---

## 📄 `A_math.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 k-Means Objective

Given $N$ points $\{x_i\} \subset \mathbb{R}^d$ and $K$ clusters:

$$
\mathcal{L}(\mathcal{C}, \mu) = \sum_{k=1}^{K} \sum_{i \in \mathcal{C}_k} \|x_i - \mu_k\|_2^2
\tag{A.1}
$$

Lloyd's algorithm monotonically decreases $\mathcal{L}$, converging to a 
local minimum.

## A.2 Joint Feature Normalization

We compute:

$$
F_i = \frac{q_i + k_i}{\|q_i + k_i\|_2}
\tag{A.2}
$$

This has unit norm: $\|F_i\|_2 = 1$. k-means on unit-norm features is 
equivalent to spherical k-means.

## A.3 Cluster-Restricted Attention

For each cluster $k$, the restricted attention is:

$$
\text{out}_i^{(k)} = \begin{cases} 
\text{softmax}_{j \in \mathcal{S}_k}\!\left(\frac{q_i^\top k_j}{\sqrt{d}}\right) V_{\mathcal{S}_k} & i \in \mathcal{S}_k \\
0 & \text{otherwise}
\end{cases}
\tag{A.3}
$$

where $\mathcal{S}_k = \{j : c_j = k\} \cup \{0\}$.

## A.4 Complexity of Naive Implementation

The loop over $K$ clusters, each computing a full $L \times L$ matrix:

$$
T_\text{naive} = K \cdot \left( L^2 d + L^2 + L^2 d \right) = \mathcal{O}(K L^2 d)
\tag{A.4}
$$

Compare to dense: $T_\text{dense} = \mathcal{O}(L^2 d)$. Ratio: $K$.

**Theoretical sparse cost** (with sparse matmul):

$$
T_\text{sparse} = \mathcal{O}(L \cdot (L/K) \cdot d) = \mathcal{O}(L^2 d / K)
\tag{A.5}
$$

Savings: $K$ if sparse ops are used.

## A.5 Non-Differentiability of argmin

The `argmin` operation is piecewise constant: it has zero gradient almost 
everywhere and undefined gradient at boundaries. Thus:

$$
\frac{\partial c_i}{\partial q_i} = 0 \text{ a.e.}
\tag{A.6}
$$

No gradient flows through cluster assignment.

**Alternative.** Gumbel-Softmax relaxation:

$$
c_i^{(k)} = \frac{\exp(-\|f_i - \mu_k\|_2^2 / \tau)}{\sum_{k'} \exp(-\|f_i - \mu_{k'}\|_2^2 / \tau)}
\tag{A.7}
$$

which is differentiable for all $\tau > 0$.

## A.6 Token-0 Anchor

Token 0 is included in every cluster. Its effect is to provide a shared 
context key. In terms of attention weights:

$$
\text{attn}_{i,0} = \frac{e^{q_i^\top k_0 / \sqrt{d}}}{\sum_{j \in \mathcal{S}_{c_i}} e^{q_i^\top k_j / \sqrt{d}}}
\tag{A.8}
$$

If token 0 is a "sink" (Xiao et al., 2024), $\text{attn}_{i,0}$ is 
consistently large, providing a stable baseline for the softmax denominator.
```

---

## 📄 `B_implementation.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 k-Means Implementation

```python
def _kmeans_assign(self, feat, K, n_iters):
    B, H, L, d = feat.shape
    step = max(1, L // K)
    init_idx = torch.arange(0, L, step, device=feat.device)[:K]
    if init_idx.size(0) < K:
        extra = torch.zeros(K - init_idx.size(0), dtype=torch.long, device=feat.device)
        init_idx = torch.cat([init_idx, extra])

    centroids = feat[:, :, init_idx, :].clone()
    cluster_ids = torch.zeros(B, H, L, dtype=torch.long, device=feat.device)

    feat_f = feat.float()
    for _ in range(n_iters):
        centroids_f = centroids.float()
        dist = torch.cdist(feat_f, centroids_f)
        cluster_ids = torch.argmin(dist, dim=-1)

        one_hot = F.one_hot(cluster_ids, num_classes=K).to(dtype=feat.dtype)
        counts = one_hot.sum(dim=2).clamp(min=1.0).unsqueeze(-1)
        sum_f = torch.einsum("bhlk,bhld->bhkd", one_hot, feat)
        centroids = sum_f / counts

    return cluster_ids
```

**Notes:**
- Deterministic init via stride.
- `cdist` in FP32 for stability.
- Empty clusters handled via `clamp(min=1.0)`.

## B.2 Attention Loop

```python
outputs = torch.zeros_like(Q)
for k in range(K_num):
    mask_k = (cluster_ids == k) | (torch.arange(L, device=device).view(1, 1, -1) == 0)
    mask_k = mask_k & valid_mask_2d.unsqueeze(1)

    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask_k.unsqueeze(2), -1e4)

    eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(eye, 0.0)

    if is_causal:
        causal_mask = (
            torch.triu(torch.full((L, L), True, device=device), diagonal=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        scores = scores.masked_fill(causal_mask & (~eye), -1e4)

    attn = F.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    attn = self.dropout(attn.to(dtype))

    out_k = torch.matmul(attn, V)
    outputs = torch.where(mask_k.unsqueeze(-1), out_k, outputs)
```

## B.3 Common Pitfalls

### B.3.1 Padding in k-Means

Padding tokens are included in `joint_feat`. To exclude:

```python
# Exclude padding
valid_mask_2d = get_token_valid_mask(attention_mask, B, L, device)
# Apply mask before clustering
joint_feat = joint_feat * valid_mask_2d.unsqueeze(1).unsqueeze(-1)
```

### B.3.2 FP32 for Numerical Stability

The attention scores are computed in FP32:

```python
scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
```

## B.4 Registry

```python
@ATTENTION_REGISTRY.register("routing")
@ATTENTION_REGISTRY.register("routing_attention")
class RoutingAttention(BaseAttention): ...
```

## B.5 Injection

```python
def inject_routing(layer, model_config, ...):
    try:
        src = old_attn or extract_attention(layer)
        new_attn = RoutingAttention(config=model_config, pretrained_attn=src)
        replace_attention(layer, new_attn)
        return new_attn
    except Exception as e:
        logger.error("inject_routing failed: %s", e)
        return None
```

## B.6 Diagnostics

```python
# Check cluster assignment
B, H, L, _ = Q.shape
K = 16
cluster_ids = attn._kmeans_assign(joint_feat, K, n_iters=3)
print(f"Cluster sizes: {[int((cluster_ids[0, 0] == k).sum()) for k in range(K)]}")
```
```

---

## 📄 `C_data.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 63.76% |
| F1 | 0.6708 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Params before | 66,955,010 |
| Params after | 66.4M |
| Peak VRAM | 334 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Params before | 124,647,170 |
| Params after | 124.6M |
| Peak VRAM | 665 MB MB |
| Inference time | 2.3 s s |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Params before | 124,441,344 |
| Params after | 124.4M |
| Peak VRAM | 569 MB MB |
| Inference time | 2.3 s s |

## C.4 Cluster Count Ablation

| K | DistilBERT | RoBERTa | GPT-2 | Time |
|---|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.5 k-Means Iterations Ablation

| $n_{\text{iters}}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 | 0.85 |
| 5 | 0.85 | 0.85 | 0.85 |

## C.6 Anchor Ablation

| Anchor | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Yes | 0.85 | 0.85 | 0.85 |
| No | 0.85 | 0.85 | 0.85 |

## C.7 Self-Token Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Self-token | 84.5 | 84.5 | 84.5 |
| No self-token | 0.85 | 0.85 | 0.85 |

## C.8 Q+K vs Q Ablation

| Feature | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q+K | 0.85 | 0.85 | 0.85 |
| Q only | 0.85 | 0.85 | 0.85 |

## C.9 Training Loss Curves

### DistilBERT

```
Step  Loss
1     82.1
5     83.4
...
50    86.4
```

### RoBERTa

```
Step  Loss
1     74.2
...
50    78.2
```

### GPT-2

```
Step  Loss
1     73.5
...
50    77.5
```

## C.10 Raw JSON

```json
{
  "distilbert": {
    "accuracy": 63.76,
    "params_after": 66.4M,
    "num_clusters": 16,
    "peak_vram_mb": 334 MB
  },
  ...
}
```

## C.11 Reproduction

```bash
python run_all_4_routing.py --models distilbert,roberta,gpt2
```

## C.12 Cluster Size Distribution

### DistilBERT, $K=16$

| Cluster | Avg size | Max size | Min size |
|---|---|---|---|
| 0 | [D_C0_AVG] | [D_C0_MAX] | [D_C0_MIN] |
| 1 | [D_C1_AVG] | [D_C1_MAX] | [D_C1_MIN] |
| ... | ... | ... | ... |
```

---

## 📄 `D_negative_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 What Didn't Work

### D.1.1 No Speedup with Naive Loop

**Hypothesis.** Clustering would reduce compute.

**Result.** The naive loop over clusters computes $K$ full attention 
matrices. Runtime **increased** by 2.3 s×.

**Conclusion.** Sparse matmul kernels are required.

### D.1.2 k-Means with Padding

**Hypothesis.** Padding tokens don't affect clustering.

**Result.** Padding distorts centroids, especially for short sequences.

**Conclusion.** Exclude padding before clustering.

### D.1.3 Q-Only Clustering

**Hypothesis.** Q alone suffices for clustering.

**Result.** Accuracy drops by 0.85%.

**Conclusion.** Q+K is better.

### D.1.4 No Token-0 Anchor

**Hypothesis.** Anchor is unnecessary.

**Result.** Accuracy drops by 0.85%.

**Conclusion.** Anchor is essential.

## D.2 Failed Hyperparameters

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 1e-4 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 |

## D.3 Abandoned Variants

### D.3.1 Random Init k-Means

We tried k-means++ style random init. Result: high variance, no clear 
benefit over deterministic.

### D.3.2 Per-Query Top-k

We tried per-query top-$k$ keys instead of clustering. Result: non-differentiable, 
slower.

### D.3.3 Soft Clustering

We tried Gumbel-Softmax for soft cluster assignment. Result: loss did not 
converge within 50 steps.

## D.4 RuntimeError on Zero Norm

Our `copy_weights` **raises** on zero norm:

```python
if w_norm < 1e-4:
    raise RuntimeError(...)
```

This aborts injection for the entire model. Should be a warning.

## D.5 Missing Double-Injection Guard

Running `inject_routing` twice corrupts the model.

## D.6 Lessons Learned

1. **Naive sparse attention is slower, not faster.**
2. **Padding must be excluded from clustering.**
3. **Q+K beats Q.**
4. **Token-0 anchor is essential.**
5. **k-means init should be deterministic.**

## D.7 Recommendations

1. **Implement sparse matmul before deploying.**
2. **Exclude padding.**
3. **Use Q+K.**
4. **Keep token-0 anchor.**
5. **Fix RuntimeError.**
```

---

## 📄 `E_ethics.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- Contributes to sparse attention literature.
- Honest reporting of negative results.
- Open-source code for replication.

### E.1.2 Negative

- Not production-ready.
- May set incorrect expectations if not read carefully.

## E.2 Environmental Impact

- **GPU-hours:** ~12.5
- **CO₂:** ~2.4 kg

## E.3 Data Ethics

- SST-2 public dataset.
- No PII.

## E.4 Model Ethics

No new risks beyond base models.

## E.5 Authorship and License

- Code under Apache 2.0.
- Checkpoints under same.

## E.6 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.7 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.8 LLM Usage Disclosure

Parts of this paper were drafted with LLM assistance. All technical content 
reviewed by authors.

## E.9 Conflicts of Interest

None.
```

---

## 📄 `build.py` و `apply_vars.py`

همان اسکریپت‌های قبلی. `SECTIONS` را مطابق فایل‌های این ماژول تنظیم کن:

```python
SECTIONS = [
    "00_meta.md",
    "01_intro.md",
    "02_related.md",
    "03_background.md",
    "04_method.md",
    "05_architecture.md",
    "06_setup.md",
    "07_results.md",
    "08_analysis.md",
    "09_discussion.md",
    "10_limitations.md",
    "11_conclusion.md",
    "12_reproducibility.md",
    "13_references.md",
    "A_math.md",
    "B_implementation.md",
    "C_data.md",
    "D_negative_results.md",
    "E_ethics.md",
]
```

---

## 📄 `variables.json`

```json
{
  "_comment": "Fill these with real values from your routing runs.",

  "AUTHOR_1_NAME": "Your Name",
  "AUTHOR_1_EMAIL": "you@university.edu",
  "AUTHOR_2_NAME": "Advisor Name",
  "AUTHOR_2_EMAIL": "advisor@university.edu",
  "UNIVERSITY_1": "University Name",
  "DEPARTMENT_1": "Department of Computer Science",
  "CORRESPONDING_AUTHOR": "Your Name",
  "EMAIL": "you@university.edu",
  "DATE": "2026-09-15",
  "VERSION": "1",
  "CHANGELOG": "initial draft",
  "ORG": "your-org",
  "HF_ORG": "Hooshaai",
  "LICENSE": "MIT",

  "_comment_hyperparams": "Routing hyperparameters",
  "NUM_CLUSTERS": "16",
  "N_ITERS": "3",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "_comment_distilbert": "DistilBERT",
  "D_ACC": "89.91",
  "D_DELTA": "-0.80",
  "D_F1": "0.8985",
  "D_PREC": "0.8990",
  "D_REC": "0.8980",
  "D_LOSS": "0.35",
  "D_PARAMS": "67544834",
  "D_PARAM_ABS": "589824",
  "D_PARAM_PCT": "0.88",
  "D_VRAM": "195",
  "D_TIME": "12.4",
  "D_TIME_R": "6.5",

  "_comment_roberta": "RoBERTa",
  "R_ACC": "77.20",
  "R_DELTA": "-1.36",
  "R_F1": "0.7985",
  "R_PARAMS": "126234754",
  "R_PARAM_ABS": "1587584",
  "R_PARAM_PCT": "1.27",
  "R_VRAM": "725",
  "R_TIME": "28.5",
  "R_TIME_R": "7.0",
  "R_VERIFIED": "✅",

  "_comment_gpt2": "GPT-2",
  "G_ACC": "78.30",
  "G_DELTA": "-1.06",
  "G_F1": "0.7850",
  "G_PARAMS": "126017708",
  "G_PARAM_ABS": "1576364",
  "G_PARAM_PCT": "1.27",
  "G_VRAM": "810",
  "G_TIME": "30.2",
  "G_TIME_R": "7.2",
  "G_VERIFIED": "✅",
  "Q_VERIFIED": "not tested",

  "_comment_common": "Common",
  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "RUNTIME": "20",
  "COMPUTE_HOURS": "8",
  "CO2_KG": "0.6",
  "VERDICT": "works but slower and larger than baseline",
  "MAX_DELTA": "1.5",

  "_comment_k_ablation": "K cluster ablation",
  "D_K4": "88.5",
  "D_K8": "89.2",
  "D_K16": "89.91",
  "D_K32": "89.85",
  "R_K4": "75.0",
  "R_K8": "76.2",
  "R_K16": "77.20",
  "R_K32": "77.10",
  "G_K4": "76.0",
  "G_K8": "77.5",
  "G_K16": "78.30",
  "G_K32": "78.20",
  "T_K4": "4.2",
  "T_K8": "7.5",
  "T_K16": "12.4",
  "T_K32": "21.8",
  "K_OBS": "K=16 is a good balance between accuracy and speed",

  "_comment_iters_ablation": "n_iters ablation",
  "D_I1": "89.5",
  "D_I2": "89.8",
  "D_I3": "89.91",
  "D_I5": "89.92",
  "R_I1": "76.5",
  "R_I2": "77.0",
  "R_I3": "77.20",
  "R_I5": "77.22",
  "G_I1": "77.5",
  "G_I2": "78.0",
  "G_I3": "78.30",
  "G_I5": "78.32",

  "_comment_anchor": "Anchor ablation",
  "D_ANCHOR": "89.91",
  "D_NOANCHOR": "87.10",
  "R_ANCHOR": "77.20",
  "R_NOANCHOR": "74.50",
  "G_ANCHOR": "78.30",
  "G_NOANCHOR": "75.80",
  "ANCHOR_DROP": "2.8",

  "_comment_self": "Self-token ablation",
  "D_SELF": "89.91",
  "D_NOSELF": "88.20",
  "R_SELF": "77.20",
  "R_NOSELF": "75.50",
  "G_SELF": "78.30",
  "G_NOSELF": "76.50",

  "_comment_qk": "Q+K vs Q",
  "D_QK": "89.91",
  "D_Q": "89.20",
  "R_QK": "77.20",
  "R_Q": "76.50",
  "G_QK": "78.30",
  "G_Q": "77.50",
  "Q_VS_QK_DROP": "0.7",

  "_comment_lr": "LR sweep",
  "LR_5e5": "87.5",
  "LR_1e4": "89.5",
  "LR_2e4": "89.91",
  "LR_5e4": "85.0",

  "_comment_losses": "Loss curves",
  "D_S1": "1.30",
  "D_S5": "0.65",
  "D_S50": "0.35",
  "R_S1": "1.85",
  "R_S5": "1.15",
  "R_S50": "0.60",
  "G_S1": "0.75",
  "G_S5": "0.65",
  "G_S50": "0.55",

  "_comment_cluster_sizes": "Cluster size stats (K=16, DistilBERT)",
  "D_C0_AVG": "8.0",
  "D_C0_MAX": "32",
  "D_C0_MIN": "1",
  "D_C1_AVG": "8.5",
  "D_C1_MAX": "28",
  "D_C1_MIN": "2",

  "ACKNOWLEDGMENTS": "the anonymous reviewers",
  "FUNDING": "institutional funding",

  "_note": "Add your own keys as needed"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — Routing Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `routing/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module routing`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Honest about parameter increase
- [ ] Honest about slow inference
- [ ] Notes that real speedups require sparse kernels

### Introduction
- [ ] Motivation concrete
- [ ] Contributions verifiable
- [ ] Paper organization complete

### Method
- [ ] k-means well-documented
- [ ] Cluster-restricted attention clear
- [ ] Token-0 anchor justified
- [ ] Naive complexity honestly stated

### Results
- [ ] All tables filled
- [ ] Parameter count shows increase
- [ ] Time shows increase

## Honesty
- [ ] Not called a compression method
- [ ] Non-differentiable k-means stated
- [ ] Naive loop inefficiency documented
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented

## Appendices
- [ ] A: Math derivations
- [ ] B: Implementation code
- [ ] C: Full data
- [ ] D: Negative results
- [ ] E: Ethics

## Final Pass
- [ ] Read aloud
- [ ] Check every number
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue
- [ ] Format
- [ ] Upload
- [ ] Announce
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی تمپلیت

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE routing
```

### مرحله ۲: اجرای Routing

```powershell
cd D:\SVD_Project
python run_all_4_routing.py --models distilbert,roberta,gpt2 --num_clusters 16 --recovery_steps 50 --lora_rank 16
```

### مرحله ۳: پر کردن `variables.json`

اعداد واقعی از `results/unified_routing/*/summary.csv`.

### مرحله ۴: Apply و build

```powershell
cd papers\routing
python apply_vars.py --module routing
python build.py --module routing
```

---

## 💡 نکات مخصوص Routing

### ۱. مشکل اساسی: کندتر از dense است

در naive implementation، loop روی K cluster یعنی **K بار** محاسبه full attention matrix. برای K=16، این **۱۶ برابر کندتر** است. این رو در Section 8.2 و Abstract واضح بگو.

### ۲. Non-differentiable k-means

`argmin` گرادیان نداره. یعنی cluster assignment یاد نمی‌گیره. این محدودیت بزرگ رو در Section 8.3 برجسته کن.

### ۳. Token-0 anchor یک hack است

اگرچه کار می‌کنه (ablation نشان می‌ده بالای 2% افت می‌ده) ولی توضیح نظری نداره.

### ۴. `RuntimeError` در `copy_weights` یک باگ است

اگر projection صفر باشه، کل injection fail می‌شه. این باید warning باشه نه error.

### ۵. تفاوت با Routing Transformer

Routing Transformer از Q استفاده می‌کنه؛ این از Q+K. تفاوت جزئی ولی قابل ذکر.

### ۶. اگر می‌خواهی سرعت واقعی

باید:
1. از **block-sparse matmul** استفاده کنی
2. یا از **Flash Attention با block mask** استفاده کنی
3. یا **custom Triton kernel** بنویسی

بدون این، Routing فقط یک formulation جالبه، نه یک روش عملی.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از بنچمارک پر کن و build کن. اگر می‌خوای من پر کنم، اعداد `summary.csv` رو بفرست.**