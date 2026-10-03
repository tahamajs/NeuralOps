# 4. Method

## 4.1 Overview

ClusterFormer replaces the standard attention operation with four 
components:

1. **Fast differentiable K-means** over queries (§4.2).
2. **Centroid-to-key cross-attention** (§4.3).
3. **Soft assignment with learned temperature** (§4.4).
4. **Local sliding window residual** (§4.5).

**Figure 4.1** shows the pipeline.

```
    ┌─────────────────────────────────────────────┐
    │  Input X → Q, K, V projections (frozen+LoRA) │
    └─────────────────────┬───────────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        ┌───────────┐           ┌───────────┐
        │ K-means   │           │ Local     │
        │ on Q      │           │ Window    │
        │ → C       │           │ Attention │
        └─────┬─────┘           └─────┬─────┘
              │                       │
              ▼                       │
        ┌───────────┐                 │
        │ Centroid- │                 │
        │ to-Key    │                 │
        │ → V_cent  │                 │
        └─────┬─────┘                 │
              │                       │
              ▼                       │
        ┌───────────┐                 │
        │ Soft      │                 │
        │ Assign    │                 │
        │ → Global  │                 │
        └─────┬─────┘                 │
              │                       │
              └─────────┬─────────────┘
                        ▼
                ┌──────────────┐
                │ Blend with α │
                └──────┬───────┘
                       │
                       ▼
                  ┌────────┐
                  │ out    │
                  └────────┘
```

## 4.2 Phase 1: Fast Differentiable K-Means

### 4.2.1 Initialization

Given queries $Q \in \mathbb{R}^{B \times H \times L \times d_h}$, we need 
$K$ initial centroids. Padding tokens must be excluded.

**Algorithm.** Given a valid-token mask $m \in \{0, 1\}^{B \times L}$:

1. Compute the rank of each valid token: $r_i = \sum_{j \leq i} m_j - 1$.
2. Compute target ranks: $t_k = \lfloor k \cdot N_\text{valid} / K \rfloor$ 
   for $k = 0, \dots, K-1$.
3. For each centroid $k$, pick the valid token whose rank is closest to $t_k$.

This spreads initial centroids uniformly across the valid sequence.

### 4.2.2 Lloyd Iterations

We run **3 iterations** of standard Lloyd's algorithm:

**Assignment:**
$$
a_{b,h,i} = \arg\min_{k \in [K]} \|Q_{b,h,i} - C_{b,h,k}\|_2^2
\tag{4.1}
$$

Implemented via `torch.cdist` on FP32 for numerical stability.

**Update:**
$$
C_{b,h,k} = \frac{\sum_{i: a_{b,h,i} = k} Q_{b,h,i}}{\sum_{i: a_{b,h,i} = k} 1}
\tag{4.2}
$$

Counts are clamped to ≥1 to avoid division by zero. Empty clusters keep 
their previous centroid.

### 4.2.3 Why 3 Iterations?

Empirically, 3 iterations are enough for attention clustering:

- More iterations increase cost $\mathcal{O}(K L d)$ per iteration.
- The soft-assignment step (Phase 4) corrects for imperfect centroids.
- We ablate this in Section 7.6.

## 4.3 Phase 2: Centroid-to-Key Cross-Attention

For each centroid $C_k$, compute attention to all keys:

$$
\text{scores}_{b,h,k,j} = \frac{C_{b,h,k} \cdot K_{b,h,j}}{\sqrt{d_h}}
\tag{4.3}
$$

$$
\alpha_{b,h,k,j} = \text{softmax}_j(\text{scores}_{b,h,k,j})
\tag{4.4}
$$

$$
V_{\text{cent}, b,h,k} = \sum_j \alpha_{b,h,k,j} V_{b,h,j}
\tag{4.5}
$$

**Complexity.** $\mathcal{O}(K \cdot L \cdot d_h)$ — linear in $L$!

**Padding.** Invalid keys are masked with $-\infty$ before softmax.

**Causality.** For decoders, we apply a causal mask by construction: 
scores are restricted to $j \leq$ some threshold. See Section 5.4.

## 4.4 Phase 3: Soft Assignment with Temperature

Each query is assigned to centroids via soft attention:

$$
\text{assign}_{b,h,i,k} = \text{softmax}_k\!\left(\frac{Q_{b,h,i} \cdot C_{b,h,k}}{\sqrt{d_h} \cdot \tau}\right)
\tag{4.6}
$$

with learnable temperature $\tau$. The global context is:

$$
\text{Global}_{b,h,i} = \sum_k \text{assign}_{b,h,i,k} \cdot V_{\text{cent}, b,h,k}
\tag{4.7}
$$

**Why temperature?** Small $\tau$ sharpens the assignment toward the 
nearest centroid (harder clustering); large $\tau$ softens it. We initialize 
$\tau = 0.1$ and let recovery training learn the optimal value.

**Clamping.** $\tau$ is clamped to $[0.1, 10.0]$ for stability.

**Empty clusters.** If a cluster has zero assigned tokens, its assignment 
logit is masked to $-\infty$, ensuring no query attends to it.

## 4.5 Phase 4: Local Sliding Window Residual

Clustering captures global structure but can miss fine-grained local 
dependencies. We add a sliding-window attention branch:

$$
\text{scores}^\text{local}_{b,h,i,j} = \frac{Q_{b,h,i} \cdot K_{b,h,j}}{\sqrt{d_h}} \cdot \mathbf{1}[|i - j| \leq W]
\tag{4.8}
$$

$$
\text{Local}_{b,h,i} = \sum_{j: |i-j| \leq W} \text{softmax}_j(\text{scores}^\text{local}) \cdot V_{b,h,j}
\tag{4.9}
$$

**Complexity.** $\mathcal{O}(L \cdot W \cdot d_h)$.

### 4.5.1 Blending

The final output is:

$$
\text{Output} = (1 - \sigma(\alpha)) \cdot \text{Global} + \sigma(\alpha) \cdot \text{Local}
\tag{4.10}
$$

where $\alpha \in \mathbb{R}$ is a **learnable scalar** initialized to 0 
(so $\sigma(0) = 0.5$, giving 50/50 blend at start).

**Why sigmoid?** Enforces $\alpha_\text{blend} \in (0, 1)$, avoiding 
negative or >1 weights.

## 4.6 The Full Algorithm

**Algorithm 1:** ClusterFormer forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask, position_embeddings
Output: Y, attn_weights

 1: Q ← LoRALinear_Q(X)                       ▷ Frozen base + LoRA
 2: K ← LoRALinear_K(X)
 3: V ← LoRALinear_V(X)
 4:
 5: (Q, K) ← apply_rope(Q, K, cos, sin)
 6: if GQA: K, V ← repeat_kv(K, V)
 7:
 8: if decoder:                                 ▷ Exact causal path
 9:     S ← QK^T / √d_h
10:     S ← causal_mask(S)
11:     A ← softmax(S)
12:     Y_ctx ← A @ V
13: else:                                        ▷ Cluster path
14:     C, ids, counts ← kmeans(Q, K, iters=3)   ▷ Cluster queries
15:     S_cent ← C K^T / √d_h
16:     A_cent ← softmax(S_cent)
17:     V_cent ← A_cent @ V                       ▷ Centroid contexts
18:     S_assign ← Q C^T / (√d_h · τ)
19:     A_assign ← softmax(S_assign)
20:     Y_global ← A_assign @ V_cent
21:
22:     if W > 0:
23:         S_local ← QK^T / √d_h
24:         S_local ← window_mask(S_local, W)
25:         A_local ← softmax(S_local)
26:         Y_local ← A_local @ V
27:         α ← σ(learned_scalar)
28:         Y_ctx ← (1-α) Y_global + α Y_local
29:     else:
30:         Y_ctx ← Y_global
31:
32: Y ← Y_ctx.reshape(B, L, H·d_h)
33: if nested: Y_out ← Y
34: else: Y_out ← LoRALinear_O(Y)
35:
36: return Y_out
```

## 4.7 Complexity Analysis

### 4.7.1 Time Complexity

| Component | Complexity |
|---|---|
| Q/K/V projections | $\mathcal{O}(L d^2)$ |
| K-means (3 iters) | $\mathcal{O}(K L d_h)$ |
| Centroid-to-key | $\mathcal{O}(K L d_h)$ |
| Soft assignment | $\mathcal{O}(K L d_h)$ |
| Local window | $\mathcal{O}(L W d_h)$ |
| Output projection | $\mathcal{O}(L d^2)$ |
| **Total** | $\mathcal{O}(L d^2 + L K d_h + L W d_h)$ |

For $K, W \ll L$, this is effectively **linear in $L$**.

### 4.7.2 Comparison with Baseline

| Config | Attention cost |
|---|---|
| Baseline | $\mathcal{O}(L^2 d_h)$ |
| ClusterFormer ($K=16, W=32$) | $\mathcal{O}(L \cdot 48 \cdot d_h)$ |
| **Reduction at $L=128$** | **$\approx 2.7\times$** |
| **Reduction at $L=512$** | **$\approx 10.7\times$** |

### 4.7.3 Parameter Count

| Component | Parameters |
|---|---|
| LoRA ($r_L=16$) | $4 \times r_L \times d$ per layer |
| Learnable temperature | 1 |
| Learnable blend | 1 |
| **Per layer** | $4 r_L d + 2$ |
| **Total ($N_l$ layers)** | $4 r_L d N_l + 2 N_l$ |

For DistilBERT ($d=768$, $r_L=16$, $N_l=6$): **+294,916 params**.

**ClusterFormer is not a compression method.** Parameter count increases 
by **0.88%**.

## 4.8 Theoretical Notes

### 4.8.1 K-Means Convergence

Lloyd's algorithm monotonically decreases the objective (3.3) and converges 
in finite iterations. We use 3 iterations as a practical approximation.

### 4.8.2 Complexity Proof

**Claim.** ClusterFormer's attention cost is $\mathcal{O}(K L d_h)$.

**Proof.** Direct from the algorithm:

- K-means: each iteration is $\mathcal{O}(L K d_h)$ via cdist. Three 
  iterations = $\mathcal{O}(3 L K d_h)$.
- Centroid-to-key: matrix product of $(K \times d_h)$ and $(d_h \times L)$ 
  is $\mathcal{O}(K L d_h)$.
- Soft assignment: product of $(L \times d_h)$ and $(d_h \times K)$ is 
  $\mathcal{O}(L K d_h)$.

Sum is $\mathcal{O}(L K d_h)$. $\square$

### 4.8.3 Approximation vs Exactness

Unlike Performer/Linformer, ClusterFormer is **exact** for the attention 
distribution over centroids. It approximates the attention over tokens by 
grouping tokens. The approximation error is bounded by the K-means 
distortion:

$$
\epsilon \leq \sum_i \|Q_i - C_{a_i}\|_2^2
\tag{4.11}
$$

See Appendix A.5 for details.

## 4.9 Hyperparameter Guidance

| Hyperparameter | Default | Effect |
|---|---|---|
| $K$ (num_clusters) | 16 | Higher = better fidelity, more cost |
| Iterations | 3 | Higher = better centroids |
| $W$ (window_size) | 32 | Higher = more local context |
| $\tau$ (temp) | learnable | Smaller = harder clustering |
| $\alpha$ (blend) | learnable | Higher = more local |

**Choosing K:**

- $K = 8$: aggressive clustering, faster
- $K = 16$: balanced (default)
- $K = 32$: more fidelity, slower
- $K = L$: no compression (full attention)

---