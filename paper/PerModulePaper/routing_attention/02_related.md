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

---