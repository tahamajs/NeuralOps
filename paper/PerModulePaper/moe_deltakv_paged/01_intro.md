# 1. Introduction

## 1.1 Motivation

Standard transformer attention is quadratic in sequence length, with three 
dominant costs:

1. **Compute:** $QK^\top$ requires $\mathcal{O}(L^2 d_h)$ FLOPs.
2. **Memory:** The attention matrix is $\mathcal{O}(L^2)$.
3. **KV cache:** During decoding, caching all $K, V$ pairs costs 
   $\mathcal{O}(L d)$ per layer per head.

Three families of methods target these costs independently:

- **Sparse/block attention** (MoBA, BigBird, Longformer) reduces compute 
  by attending only to relevant blocks.
- **KV cache compression** (H2O, Delta-KV) reduces memory by quantizing or 
  evicting KV pairs.
- **Paged attention** (vLLM) improves memory efficiency and throughput 
  during serving.

**MoE-DeltaKV-Paged** combines all three, providing a single attention module 
that:
- Routes queries to the top-$k$ key pages via a learned router.
- Compresses KV deltas using low-rank SVD + 4-bit quantization.
- Operates on fixed-size pages for cache-friendly computation.

## 1.2 Problem Statement

Given input $X \in \mathbb{R}^{B \times N \times d}$, we seek an attention 
mechanism $\mathcal{A}'$ that:

$$
\mathcal{A}' \approx \mathcal{A}, \qquad
\text{FLOPs}(\mathcal{A}') < \mathcal{O}(L^2)
\tag{1.1}
$$

subject to:

1. **Page-level sparsity:** Each query page attends to at most $k_\text{exp}$ 
   key pages.
2. **Differentiable compression:** SVD and 4-bit quantization must be 
   trainable end-to-end.
3. **CLS preservation:** The first page (containing `[CLS]`) must always be 
   selected.
4. **Multi-architecture:** DistilBERT (bidirectional), RoBERTa (nested), 
   GPT-2 (causal).
5. **Load balancing:** The router must not collapse to a few pages.

## 1.3 Our Approach

### 1.3.1 Mixture-of-Block Routing

The sequence is partitioned into $\text{num\_pages}$ pages of size 
$\text{page\_size}$. Each query page $i$ computes a router logit against 
each key page $j$:

$$
\ell_{ij} = \frac{(W_\text{rg} \bar{q}_i) \cdot \bar{k}_j}{\sqrt{d_h}}
\tag{1.2}
$$

where $\bar{q}_i, \bar{k}_j$ are page-level mean projections. The top-$k$ 
pages are selected and attended to.

### 1.3.2 Delta-KV Compression

Given keys $K \in \mathbb{R}^{B \times H \times N \times d_h}$, we compute 
temporal deltas:

$$
\Delta K_t = K_t - K_{t-1}, \quad t = 1, \dots, N-1
\tag{1.3}
$$

The deltas are compressed via SVD to rank $r$, quantized to 4-bit, and 
reconstructed:

$$
\hat{K}_t = K_0 + \sum_{\tau=1}^t \widehat{\Delta K}_\tau
\tag{1.4}
$$

### 1.3.3 Differentiable Quantization

We use the Straight-Through Estimator (STE) for 4-bit uniform quantization:

$$
\text{STE}(x) = x + (Q_{4\text{bit}}(x) - x).\text{detach}()
\tag{1.5}
$$

Forward: uses $Q_{4\text{bit}}(x)$. Backward: gradient flows through $x$.

### 1.3.4 CLS Preservation

The router logit for page 0 is boosted by a large constant:

$$
\ell_{i, 0} \leftarrow \ell_{i, 0} + 10^4
\tag{1.6}
$$

This ensures page 0 is always selected by top-$k$.

### 1.3.5 Load Balancing

An auxiliary loss encourages uniform page selection:

$$
\mathcal{L}_\text{aux} = \text{num\_pages} \cdot \sum_{j} \rho_j \cdot \pi_j
\tag{1.7}
$$

where $\rho_j$ is the density of page $j$ (fraction of queries selecting it) 
and $\pi_j$ is the mean router probability for page $j$.

## 1.4 Contributions

- **A hybrid attention module** combining MoBA-style block routing, Delta-KV 
  temporal compression, and paged attention.
- **Differentiable 4-bit quantization** via STE, applied to temporal deltas.
- **CLS preservation** via page-0 logit boosting.
- **Load balancing** via a MoE-style auxiliary loss.
- **Multi-architecture support** for bidirectional and causal models.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on MoBA, 
Delta-KV, and STE. Section 4 presents our method. Section 5 describes 
integration. Section 6 details setup. Section 7 reports results. Section 8 
analyzes trade-offs. Section 9 discusses practical usage. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $N$ | Sequence length |
| $H$ | Query heads |
| $d_h$ | Head dimension |
| $P$ | Page size |
| $N_p$ | Number of pages |
| $k_\text{exp}$ | Number of selected pages (experts) |
| $r$ | SVD rank for delta compression |
| $W_\text{rg}$ | Router weight |
| $\bar{q}, \bar{k}$ | Page-mean query/key |

---