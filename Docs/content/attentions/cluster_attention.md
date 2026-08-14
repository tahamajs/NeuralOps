# ClusterFormer: Dynamic Token-Clustering Attention via Batch Iterative K-Means

## 1. Executive Abstract & Architectural Motivation
Dense Multi-Head Self-Attention (MHSA) computes $L \times L$ pairwise interaction matrices across all tokens in a sequence, leading to quadratic time complexity $\mathcal{O}(L^2 d)$ and severe memory constraints for long context models. Standard sparse attention paradigms rely either on fixed positional windows or static random graphs, which fail to adapt to content-dependent semantic clustering in hidden representations.

**ClusterFormer (`ClusterAttention`)** reformulates sequence-wide self-attention as a dynamic, content-driven semantic partitioning problem. Prior to computing attention matrices, token embeddings are dynamically grouped into $K$ spatial clusters using a batch-wise iterative $K$-means clustering algorithm (Lloyd's algorithm) operating in feature space. Pairwise attention is subsequently restricted strictly to tokens within the same cluster assignment $C_k$. 

By constraining attention calculations to intra-cluster token sets of average size $N_k \approx L / K$, ClusterFormer reduces attention score evaluations from $L^2$ to $\sum_{k=1}^K |C_k|^2 \approx L^2 / K$. This achieves an asymptotic $K \times$ reduction in attention matrix computations while preserving long-range semantic interactions among globally distant tokens that share geometric proximity in embedding space.

---

## 2. Mathematical Formulations & Derivations

### 2.1 Batch-Wise Iterative $K$-Means Clustering
Given input hidden states $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$, $K = \min(\text{num\_clusters}, L)$ initial centroids $\boldsymbol{\mu}^{(0)} \in \mathbb{R}^{B \times K \times D}$ are initialized via uniform random permutation without replacement from token vectors $\mathbf{X}$.

For $t = 1, \dots, I$ iterations (where $I = \text{n\_iters}$):

1. **Euclidean Distance Computation**:
   $$\mathbf{D}_{b, i, k}^{(t)} = \|\mathbf{X}_{b, i, :} - \boldsymbol{\mu}_{b, k, :}^{(t-1)}\|_2 = \sqrt{\sum_{d=1}^D \left( \mathbf{X}_{b, i, d} - \boldsymbol{\mu}_{b, k, d}^{(t-1)} \right)^2}$$

2. **Nearest Centroid Assignment**:
   $$c_{b, i}^{(t)} = \operatorname{arg\,min}_{k \in \{1, \dots, K\}} \mathbf{D}_{b, i, k}^{(t)} \quad \in \{0, \dots, K-1\}^{B \times L}$$

3. **Centroid Coordinate Recalculation**:
   $$\boldsymbol{\mu}_{b, k, :}^{(t)} = \frac{\sum_{i=1}^L \mathbb{I}\left( c_{b, i}^{(t)} = k \right) \mathbf{X}_{b, i, :}}{\max\left(1, \sum_{i=1}^L \mathbb{I}\left( c_{b, i}^{(t)} = k \right)\right)}$$

### 2.2 Intra-Cluster Masked Attention Computation
Input activations are mapped to Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) spaces via linear projections $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times D}$:
$$\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times H \times L \times d_h}$$
where $H$ is the number of heads and $d_h = D / H$.

For each cluster partition $k \in \{0, \dots, K-1\}$, a binary cluster mask $\mathbf{M}^{(k)} \in \{0, 1\}^{B \times 1 \times 1 \times L}$ isolates tokens assigned to cluster $k$:
$$\mathbf{M}_{b, 1, 1, j}^{(k)} = \mathbb{I}\left( c_{b, j} = k \right)$$

An additive mask $\mathbf{G}^{(k)} \in \{0, -\infty\}^{B \times 1 \times 1 \times L}$ masks out keys outside cluster $k$:
$$\mathbf{G}_{b, 1, 1, j}^{(k)} = \begin{cases} 0 & \text{if } c_{b, j} = k \\ -10^9 & \text{if } c_{b, j} \neq k \end{cases}$$

The intra-cluster attention score matrix $\mathbf{S}^{(k)} \in \mathbb{R}^{B \times H \times L \times L}$ for cluster $k$ is evaluated as:
$$\mathbf{S}_{b, h, i, j}^{(k)} = \frac{\mathbf{Q}_{b, h, i, :} \mathbf{K}_{b, h, j, :}^\top}{\sqrt{d_h}} + \mathbf{G}_{b, 1, 1, j}^{(k)} + \mathbf{P}_{b, 1, 1, j}$$
where $\mathbf{P}$ incorporates any optional padding mask.

Softmax normalization computes cluster-gated attention weights $\mathbf{A}^{(k)}$:
$$\mathbf{A}_{b, h, i, j}^{(k)} = \operatorname{Softmax}_j\left( \mathbf{S}_{b, h, i, j}^{(k)} \right) = \frac{\exp\left( \mathbf{S}_{b, h, i, j}^{(k)} \right)}{\sum_{m=1}^L \exp\left( \mathbf{S}_{b, h, i, m}^{(k)} \right)}$$
$$\mathbf{O}_{b, h, i, :}^{(k)} = \sum_{j=1}^L \mathbf{A}_{b, h, i, j}^{(k)} \mathbf{V}_{b, h, j, :}$$

### 2.3 Output Aggregation & Reassembly
The intermediate cluster context vectors $\mathbf{O}^{(k)}$ are zero-masked for queries outside cluster $k$ and summed across all $K$ clusters:
$$\mathbf{O}_{\text{final}, b, h, i, :} = \sum_{k=0}^{K-1} \mathbb{I}\left( c_{b, i} = k \right) \cdot \mathbf{O}_{b, h, i, :}^{(k)}$$

The multi-head output $\mathbf{O}_{\text{final}} \in \mathbb{R}^{B \times H \times L \times d_h}$ is reshaped to $\mathbb{R}^{B \times L \times D}$ and passed through the final output projection:
$$\mathbf{Y} = \operatorname{Reshape}\left( \mathbf{O}_{\text{final}} \right) \mathbf{W}_O \quad \in \mathbb{R}^{B \times L \times D}$$

---

## 3. Step-by-Step Tensor Execution Trace

| Step | Operation | Input Shape | Output Shape | Tensor Transformation Description |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Projections | `[B, L, D]` | `[B, H, L, d_h]` | Linear map via $W_Q, W_K, W_V$ & split heads |
| 2 | Initial Centroids | `[B, L, D]` | `[B, K, D]` | Random permutation selection of $K$ token vectors |
| 3 | Distance Matrix | `[B, L, D]`, `[B, K, D]` | `[B, L, K]` | Batch Pairwise Euclidean Distance `cdist(X, centroids)` |
| 4 | Nearest Assignment | `[B, L, K]` | `[B, L]` | `argmin(dist, dim=-1)` index extraction |
| 5 | Centroid Recalculation| `[B, L, D]`, `[B, L]` | `[B, K, D]` | Masked mean sum over assigned tokens ($I$ iterations) |
| 6 | Cluster Loop ($k=1..K$)| `[B, L]` | `[B, 1, 1, L]` | Construct binary additive mask $-10^9 \cdot (1 - \mathbb{I}(c=k))$ |
| 7 | Masked Attention | `Q`, `K`, `V`, `Mask` | `[B, H, L, d_h]` | Scaled dot-product + additive cluster mask + Softmax |
| 8 | Context Mask & Add | `[B, H, L, d_h]`, `c` | `[B, H, L, d_h]` | Accumulate `O_k * IS_IN_CLUSTER_K` into global `out` |
| 9 | Output Projection | `[B, L, D]` | `[B, L, D]` | Reshape multi-head context & linear project $W_O$ |

```
                       [B, L, D] Input Tensor
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      Q, K, V Projections          K-Means Clustering
      [B, H, L, d_h]               (I Iterations)
             │                           │
             │                           ▼
             │                   Cluster IDs c [B, L]
             │                           │
             └─────────────┬─────────────┘
                           ▼
               Iterate over Clusters k (0..K-1)
                           │
                           ▼
          Generate Additive Mask G^(k) [B, 1, 1, L]
             G = 0 if c_j == k else -1e9
                           │
                           ▼
          Masked Softmax Attention Scores S^(k)
             Softmax(Q K^T / sqrt(d_h) + G^(k))
                           │
                           ▼
          Cluster Context O^(k) = A^(k) @ V
                           │
                           ▼
          Zero-Mask Non-Cluster Queries
          O_final += O^(k) * I(c_i == k)
                           │
                           ▼
          Reshape & Linear Projection W_O
                           │
                           ▼
                      [B, L, D] Output
```

---

## 4. Analytical Complexity & Hardware Profiling

### 4.1 FLOPs Computation Breakdown
Let $B$ be batch size, $L$ sequence length, $D$ hidden dimension, $H$ attention heads, $d_h = D/H$, $K$ total clusters, and $I$ clustering iterations.

1. **Linear Projections ($\mathbf{Q}, \mathbf{K}, \mathbf{V}$)**:
   $$\operatorname{FLOPs}_{\text{proj}} = 3 \times 2 B L D^2 = 6 B L D^2$$
2. **Iterative $K$-Means Clustering**:
   In each iteration $t \in \{1, \dots, I\}$:
   - Euclidean Distance (`cdist`): $3 B L K D$ FLOPs (squared differences + sum + sqrt).
   - Centroid mean updates: $2 B L D$ FLOPs.
   Across $I$ iterations:
   $$\operatorname{FLOPs}_{\text{kmeans}} = I \cdot (3 B L K D + 2 B L D) = 3 I B L K D + 2 I B L D$$
3. **Sequential Intra-Cluster Attention Loops**:
   The Python loop executes $K$ passes. In pass $k$, full $L \times L$ matrix multiplies $\mathbf{Q} \mathbf{K}^\top$ are computed:
   - Attention score matmul: $2 B H L^2 d_h = 2 B L^2 D$
   - Softmax & context matmul: $2 B H L^2 d_h = 2 B L^2 D$
   Summing across all $K$ loop iterations in the un-optimized implementation:
   $$\operatorname{FLOPs}_{\text{attn\_naive}} = K \times (4 B L^2 D) = 4 K B L^2 D$$
   *(Note: In a theoretical grouped implementation without dense $L \times L$ masks, FLOPs scale as $4 B L (L / K) D = \frac{4 B L^2 D}{K}$)*.
4. **Output Projection ($\mathbf{W}_O$)**:
   $$\operatorname{FLOPs}_{\text{out}} = 2 B L D^2$$

$$\text{Total Naive FLOPs} = 8 B L D^2 + 3 I B L K D + 4 K B L^2 D$$

### 4.2 MACs & Memory Footprint Analysis
- **Multiply-Accumulate Operations (MACs)**:
  $$\text{Total MACs} \approx 4 B L D^2 + 1.5 I B L K D + 2 K B L^2 D$$
- **Peak Activation Memory**:
  - Distance Tensor: $B \cdot L \cdot K \times 4 \text{ bytes}$
  - Cluster Masks: $K \times B \cdot 1 \cdot 1 \cdot L \times 4 \text{ bytes}$
  - Full Attention Matrix per cluster iteration: $B \cdot H \cdot L \cdot L \times 2 \text{ bytes}$.
  - Peak memory scale: $\mathcal{O}(B H L^2 + B L K)$ due to un-fused $L \times L$ attention matrices inside the loop.

### 4.3 Asymptotic Scalability Comparison
- **Standard Dense Attention**: $\mathcal{O}(L^2 D)$ compute, $\mathcal{O}(L^2 H)$ memory.
- **Un-Fused PyTorch ClusterFormer**: $\mathcal{O}(K L^2 D + I L K D)$ compute — **slower** than standard attention due to executing $K$ sequential $L \times L$ dense matrix multiplications in PyTorch!
- **Optimized Grouped ClusterFormer**: $\mathcal{O}\left(I L K D + \frac{L^2 D}{K}\right)$ compute, achieving an optimal $K \times$ speedup.

---

## 5. Failure Modes, Rank Collapse & Edge Cases

### 5.1 The Sequential $K \times L^2$ Execution Trap
In the naive PyTorch implementation:
```python
for k in range(num_k):
    scores_k = torch.matmul(Q_k, K_k.transpose(-2, -1)) * scale # [B, H, L, L]
    scores_k = scores_k + mask_add
    attn_k = F.softmax(scores_k, dim=-1)
    out_k = torch.matmul(attn_k, V_k) # [B, H, L, D_h]
```
Instead of computing attention exclusively on the subset of tokens inside cluster $k$, the code evaluates a **full $L \times L$ matrix product** $K$ separate times, relying on $-10^9$ additive masks to zero out non-cluster tokens.
- **Computational Disparity**: Rather than accelerating attention by $K \times$, this implementation runs $K$ times slower than standard attention, scaling as $\mathcal{O}(K L^2 D)$.

### 5.2 Empty Cluster Collapse & Non-Differentiable Assignments
1. **Empty Cluster Deadlocks**: If initial centroid sampling picks duplicate or outlier tokens, some clusters receive zero token assignments ($\sum \mathbb{I}(c_i = k) = 0$). `count_k.clamp(min=1.0)` prevents division-by-zero, but leaves dead centroids that waste computation.
2. **Hard Assignment Gradient Disconnect**: The operation $c_i = \operatorname{arg\,min}_k \mathbf{D}_{i, k}$ is non-differentiable ($\frac{\partial c_i}{\partial \mathbf{X}} = 0$). Gradients during backpropagation cannot flow through cluster assignments to optimize the embedding space specifically for $K$-means clustering.

### 5.3 Cluster Imbalance & Outlier Domination
In natural language, semantic token distributions are highly skewed (e.g., common stop words form massive clusters, while domain-specific entity tokens form tiny clusters).
- **Worst-Case Scaling**: If a single cluster absorbs $90\%$ of sequence tokens ($|C_1| \approx 0.9 L$), effective complexity reverts to $\mathcal{O}(L^2 D)$, eliminating sparse speedup advantages.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

To realize true sub-quadratic speedups, tokens must be physically re-ordered (permuted) into contiguous cluster blocks in SRAM before executing dense attention.

### 6.1 Triton Hardware Acceleration Pipeline

```
+-----------------------------------------------------------------------------------+
|                        TRITON FUSED CLUSTERATTN KERNEL                            |
+-----------------------------------------------------------------------------------+
|  1. K-MEANS CENTROID REDUCTION KERNEL                                             |
|     - Computes fused cdist(X, centroids) in SRAM registers.                       |
|     - Performs warp-level reduction to find min_dist cluster index c_i.           |
|                                                                                   |
|  2. TOKEN SORT & PERMUTATION KERNEL                                               |
|     - Sorts token indices by cluster ID: perm_idx = argsort(c).                   |
|     - Generates cluster boundary offsets: cluster_offsets = [0, N1, N1+N2, ...].   |
|     - Writes permuted tensors Q_sorted, K_sorted, V_sorted into contiguous memory.|
|                                                                                   |
|  3. BLOCK-DIAGONAL FLASHATTENTION KERNEL                                          |
|     - Grid launched over cluster tiles.                                           |
|     - For cluster k: reads contiguous Q_k, K_k, V_k of size [N_k, d_h].           |
|     - Computes dense FlashAttention ONLY on [N_k, N_k] matrix.                    |
|     - Writes results back to un-permuted output buffer Y[perm_idx].               |
+-----------------------------------------------------------------------------------+
```

### 6.2 Implementation Code Blueprint (Triton Permuted Grouped Kernel)

```python
import triton
import triton.language as tl

@triton.jit
def _cluster_block_flash_attn_kernel(
    Q_sorted_ptr, K_sorted_ptr, V_sorted_ptr, Out_sorted_ptr,
    Cluster_Offsets_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_K: tl.constexpr
):
    cluster_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    batch_id = tl.program_id(axis=2)

    # Read start and end sequence bounds for target cluster
    start_idx = tl.load(Cluster_Offsets_ptr + cluster_id)
    end_idx = tl.load(Cluster_Offsets_ptr + cluster_id + 1)
    cluster_len = end_idx - start_idx

    # If cluster is empty, exit early
    if cluster_len <= 0:
        return

    offs_d = tl.arange(0, D_K)

    # Loop over query tiles within this cluster's range
    for m_offset in range(0, cluster_len, BLOCK_M):
        offs_m = start_idx + m_offset + tl.arange(0, BLOCK_M)
        mask_m = offs_m < end_idx

        # Load Q tile [BLOCK_M, D_K]
        q_ptrs = Q_sorted_ptr + batch_id * stride_qb + head_id * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D_K], dtype=tl.float32)

        # Loop over key tiles ONLY within the same cluster
        for n_offset in range(0, cluster_len, BLOCK_N):
            offs_n = start_idx + n_offset + tl.arange(0, BLOCK_N)
            mask_n = offs_n < end_idx

            k_ptrs = K_sorted_ptr + batch_id * stride_kb + head_id * stride_kh + (offs_n[None, :] * stride_kn + offs_d[:, None])
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

            # Local intra-cluster dot product [BLOCK_M, BLOCK_N]
            s = tl.dot(q, k) * scale
            s = tl.where(mask_m[:, None] & mask_n[None, :], s, float("-inf"))

            # Online Softmax update
            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            p = tl.exp(s - m_new[:, None])
            l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p, axis=1)

            acc_scale = tl.exp(m_i - m_new)
            acc = acc * acc_scale[:, None]

            v_ptrs = V_sorted_ptr + batch_id * stride_vb + head_id * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            acc = tl.dot(p.to(v.dtype), v, acc)

            l_i = l_new
            m_i = m_new

        # Normalize and store back
        acc = acc / l_i[:, None]
        out_ptrs = Out_sorted_ptr + batch_id * stride_ob + head_id * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
        tl.store(out_ptrs, acc.to(Out_sorted_ptr.dtype.element_ty), mask=mask_m[:, None])
```

### 6.3 Hardware Acceleration Gains
1. **True $K \times$ Compute Reduction**: By grouping tokens contiguously into `cluster_len` sub-matrices, total matmul calculations drop from $L^2$ to $\sum N_k^2 \approx L^2 / K$.
2. **Zero Waste on Masking**: Eliminates $K$ separate $L \times L$ full-matrix allocations and $-10^9$ mask additions, maximizing memory bandwidth and Tensor Core throughput.

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Differentiable Soft $K$-Means Gating**: Replace hard `argmin` distance assignments with temperature-annealed Softmax clustering probabilities to allow end-to-end backpropagation through cluster centroid projections.
2. **Hierarchical Balanced Balanced-Tree K-Means**: Enforce equal-sized cluster constraints ($|C_k| = L / K$) using spatial median splits to eliminate cluster size imbalance.
3. **Cross-Cluster Top-$m$ Routing**: Allow queries in cluster $k$ to attend to the top-$m$ nearest cluster centroids, preventing information loss at cluster boundaries.
4. **FP8 Cluster Centroid Distance Kernels**: Execute distance calculations $\mathbf{D} = \|\mathbf{X} - \boldsymbol{\mu}\|_2$ in FP8 precision (E4M3), delivering $4\times$ throughput speedup on NVIDIA H100 Tensor Cores.
