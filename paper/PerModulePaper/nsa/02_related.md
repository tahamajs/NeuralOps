<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Fixed Sparse Attention

**Longformer** (Beltagy et al., 2020): Sliding window + global tokens. 
Deterministic pattern.

**BigBird** (Zaheer et al., 2020): Sliding window + random + global. 
Theoretical universality.

**Mistral** (Jiang et al., 2023): Fixed 4096-token sliding window.

**Sparse Transformer** (Child et al., 2019): Strided + local patterns.

**NSA's local branch** uses a sliding window (same as Longformer) but 
**augments** with content-based global.

## 2.2 Content-Based Sparse Attention

**Routing Transformer** (Roy et al., 2021): K-means clustering of queries 
and keys.

**Reformer** (Kitaev et al., 2020): Locality-sensitive hashing (LSH).

**Sinkhorn Attention** (Tay et al., 2020): Sorted bucket attention.

**NSA differs** by using **direct top-$k$ selection** — no clustering, no 
hashing.

## 2.3 Dynamic Sparsity

**H2O** (Zhang et al., 2023): Heavy-hitter oracle for KV cache eviction. 
Selects top-k keys for each query.

**StreamingLLM** (Xiao et al., 2024): Attention sinks + sliding window.

**NSA** generalizes both: it retains local window + top-$k$ global in **every 
forward pass**, not just at cache time.

## 2.4 Learned Sparsity

**Adaptive Span** (Sukhbaatar et al., 2019): Learned per-head window.

**Adaptive Sparse Attention** (our earlier work): Learned per-head top-$k$.

**NSA** uses **fixed** $w, k$ but **dynamic content-based selection**.

## 2.5 Position of Our Work

| Axis | Longformer | Routing | H2O | NSA |
|---|---|---|---|---|
| Local | Fixed window | — | — | **Fixed window** |
| Global | Global tokens | K-means | Heavy hitters | **Top-k** |
| Content-aware | No | Yes | Yes | **Yes** |
| Vectorized | Yes | No | Yes | **Yes** |
| Pretrained | Yes | Rarely | Yes | **Yes** |

## 2.6 Comparison Table

| Method | Local | Global | Selection | Loop-free |
|---|---|---|---|---|
| Longformer | Window | Tokens | Fixed | Yes |
| BigBird | Window + random | Tokens | Fixed | Yes |
| Routing | — | K-means | Cluster | No |
| Reformer | — | LSH | Hash | No |
| H2O | — | Heavy hitters | Score | Yes |
| **NSA** | **Window** | **Top-k** | **Score** | **Yes** |

---