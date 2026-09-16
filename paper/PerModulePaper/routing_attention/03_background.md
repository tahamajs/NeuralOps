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

---