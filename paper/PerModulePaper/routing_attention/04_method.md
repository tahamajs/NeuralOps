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

---