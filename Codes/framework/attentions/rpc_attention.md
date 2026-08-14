# RPC Attention: Random Projection Compressed Subspace Sketching

## 1. Abstract & Theoretical Motivation
High-dimensional head space evaluations $d$ in multi-head attention require expensive dot-product calculations across query and key matrices. Random Projection Compressed (RPC) Attention leverages the Johnson-Lindenstrauss (JL) Lemma, which states that any set of vectors in a high-dimensional space can be projected into a significantly lower-dimensional subspace while approximately preserving pairwise distances and inner products.

RPC Attention constructs fixed head-wise Gaussian random projection matrices $\mathbf{P} \in \mathbb{R}^{H \times d \times d_p}$, sketching query and key heads down to compressed rank $d_p \ll d$. Computing query-key similarity scores in the sketched manifold reduces FLOP requirements for score matrix generation while maintaining concentration of measure bounds on scalar products.

---

## 2. Mathematical Derivations & Formulations

Given input sequence representations $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$, multi-head query, key, and value tensors are generated via linear projections for head $h \in \{1, \dots, H\}$:

\[
\mathbf{Q}_h = \mathbf{X} \mathbf{W}_Q^{(h)} \in \mathbb{R}^{B \times N \times d}, \quad \mathbf{K}_h = \mathbf{X} \mathbf{W}_K^{(h)} \in \mathbb{R}^{B \times N \times d}, \quad \mathbf{V}_h = \mathbf{X} \mathbf{W}_V^{(h)} \in \mathbb{R}^{B \times N \times d}
\]

A head-wise Gaussian random projection matrix buffer $\mathbf{P} \in \mathbb{R}^{H \times d \times d_p}$ is initialized with isotropic Gaussian entries $\mathbf{P}_{h, i, j} \sim \mathcal{N}\left(0, \frac{1}{d}\right)$ where $d_p$ is the target projection rank.

Queries and keys are sketched into the $d_p$-dimensional random subspace:

\[
\widetilde{\mathbf{Q}}_h = \mathbf{Q}_h \mathbf{P}_h \in \mathbb{R}^{B \times N \times d_p}, \quad \widetilde{\mathbf{K}}_h = \mathbf{K}_h \mathbf{P}_h \in \mathbb{R}^{B \times N \times d_p}
\]

Specifically, using Einstein summation over head and feature dimensions:

\[
\widetilde{\mathbf{Q}}_{b, h, i, e} = \sum_{m=1}^d \mathbf{Q}_{b, h, i, m} \mathbf{P}_{h, m, e}, \quad \forall e \in \{1, \dots, d_p\}
\]

Scaled similarity logits $\mathbf{S}_h \in \mathbb{R}^{B \times N \times N}$ are computed between sketched query and key representations using scaling factor $d_p^{-1/2}$:

\[
\mathbf{S}_h = \frac{\widetilde{\mathbf{Q}}_h \widetilde{\mathbf{K}}_h^\top}{\sqrt{d_p}} + \mathbf{M}_{\text{mask}}
\]

Softmax probability mapping and value aggregation yield the contextual head output:

\[
\mathbf{A}_h = \operatorname{softmax}\left( \mathbf{S}_h, \text{dim}=-1 \right) \in \mathbb{R}^{B \times N \times N}, \quad \mathbf{O}_h = \mathbf{A}_h \mathbf{V}_h \in \mathbb{R}^{B \times N \times d}
\]

Multi-head outputs are concatenated and linearly projected back to the hidden space:

\[
\mathbf{Y} = \operatorname{Concat}(\mathbf{O}_1, \dots, \mathbf{O}_H) \mathbf{W}_O \in \mathbb{R}^{B \times N \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input Hidden States X [B, N, D]
  │
  ├──> Linear Projections (W_Q, W_K, W_V) -> Q, K, V [B, H, N, d]
  │
  ├──> Random Subspace Projection (Buffer P [H, d, d_p]):
  │    ├──> Q_feat = einsum('bhld, hde -> bhle', Q, P) ───> [B, H, N, d_p]
  │    └──> K_feat = einsum('bhld, hde -> bhle', K, P) ───> [B, H, N, d_p]
  │
  ├──> Sketched Score MatMul:
  │    S = (Q_feat @ K_feat^T) / sqrt(d_p) ─────────────> [B, H, N, N]
  │    S = S + Mask ────────────────────────────────────> [B, H, N, N]
  │
  ├──> Softmax & Dropout:
  │    A = Softmax(S, dim=-1) ──────────────────────────> [B, H, N, N]
  │    A_drop = Dropout(A) ─────────────────────────────> [B, H, N, N]
  │
  └──> Value Aggregation & Output Projection:
       Context = A_drop @ V ────────────────────────────> [B, H, N, d]
       Y = Reshape(Context) @ W_O ──────────────────────> [B, N, D]
```

### Detailed Shape Progression Table

| Step | Operation | Input Tensor Shapes | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| 1 | Input Projections ($W_Q, W_K, W_V$) | `X: [B, N, D]` | `Q, K, V: [B, N, D]` |
| 2 | Multi-Head Reshape | `Q, K, V: [B, N, D]` | `Q, K, V: [B, H, N, d]` |
| 3 | Query Subspace Sketching | `Q: [B, H, N, d]`, `P: [H, d, d_p]` | `Q_feat: [B, H, N, d_p]` |
| 4 | Key Subspace Sketching | `K: [B, H, N, d]`, `P: [H, d, d_p]` | `K_feat: [B, H, N, d_p]` |
| 5 | Sketched Score Product | `Q_feat: [B, H, N, d_p]`, `K_feat^T` | `S: [B, H, N, N]` |
| 6 | Softmax Normalization | `S: [B, H, N, N]` | `A: [B, H, N, N]` |
| 7 | Value Context Product | `A: [B, H, N, N]`, `V: [B, H, N, d]` | `Context: [B, H, N, d]` |
| 8 | Output Projection | `Context: [B, H, N, d]` | `Y: [B, N, D]` |

---

## 4. Analytical Complexity Profiling

### FLOP & MAC Counting
1. **Linear Projections ($W_Q, W_K, W_V, W_O$)**: $8 B N D^2$ FLOPs.
2. **Subspace Sketching ($\mathbf{Q} \mathbf{P}, \mathbf{K} \mathbf{P}$)**:
   - Query Projection: $2 B H N d d_p = 2 B N D d_p$ FLOPs.
   - Key Projection: $2 B H N d d_p = 2 B N D d_p$ FLOPs.
3. **Sketched Attention Score Product ($\widetilde{\mathbf{Q}} \widetilde{\mathbf{K}}^\top$)**: $2 B H N^2 d_p$ FLOPs.
4. **Value Matrix Product ($\mathbf{A} \mathbf{V}$)**: $2 B H N^2 d = 2 B N^2 D$ FLOPs.

**Total Computational Cost**:
\[
\text{Total FLOPs} = 8 B N D^2 + 4 B N D d_p + 2 B H N^2 d_p + 2 B N^2 D
\]

When sketch rank $d_p < d$ (e.g., $d_p = d/2$), score multiplication FLOPs decrease proportionally.

### Memory Footprint & Tensor Store
- **Parameters**: $4 D^2$ floats. Buffer $\mathbf{P} \in \mathbb{R}^{H \times d \times d_p}$ contains $H d d_p = D d_p$ floats.
- **Activation Memory**:
  - Sketched Queries/Keys ($\widetilde{\mathbf{Q}}, \widetilde{\mathbf{K}}$): $2 B H N d_p$ floats.
  - Score & Softmax Tensors ($\mathbf{S}, \mathbf{A}$): $B H N^2$ floats.
  - Peak Activation Memory: $\mathcal{O}(B N D + B H N d_p + B H N^2)$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

1. **Johnson-Lindenstrauss Distortion Variance**: For small values of projection rank $d_p$, random sketching distorts high-dimensional inner product angles. This introduces variance into attention scores, leading to rank collapse and reduced model capacity.
2. **Static Unlearned Projection Manifold**: Because buffer $\mathbf{P}$ is frozen Gaussian noise, the random subspace cannot dynamically adapt to key-query feature correlation patterns during optimization.
3. **Persistent $\mathcal{O}(N^2)$ Memory Scaling**: While feature dimension compute drops from $d$ to $d_p$, the attention score matrix $\mathbf{S}$ remains $N \times N$, retaining quadratic space complexity $\mathcal{O}(N^2)$.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Memory Access & Vectorization Barriers
In standard PyTorch, executing `torch.einsum("bhld,hde->bhle", q, proj)` materializes intermediate sketched tensors $\widetilde{\mathbf{Q}}, \widetilde{\mathbf{K}}$ into DRAM before loading them back for score computation.

### High-Performance Triton Kernel Fusion Strategy
1. **Fused Sketching & FlashAttention Kernel**: Implement a Triton kernel that loads tiles of $\mathbf{Q}, \mathbf{K}$, and random buffer $\mathbf{P}$ into SRAM registers, executes random projection on-chip, and immediately computes dot-product scores.

```python
# Triton Pseudocode for Fused RPC Flash Kernel
@triton.jit
def _rpc_fused_sketch_kernel(
    Q_ptr, K_ptr, V_ptr, P_ptr, Out_ptr,
    N, D, d: tl.constexpr, dp: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Load Q block [BLOCK_N, d] and P buffer [d, dp] into SRAM
    q = tl.load(Q_ptr + offsets_q)
    p = tl.load(P_ptr + offsets_p)
    
    # On-chip random projection sketch inside registers
    q_feat = tl.dot(q, p)  # [BLOCK_N, dp]
    
    # Stream K blocks, project K_feat = K @ P in registers, and compute scores
    k = tl.load(K_ptr + offsets_k)
    k_feat = tl.dot(k, p)  # [BLOCK_N, dp]
    scores = tl.dot(q_feat, tl.trans(k_feat)) * scale
    
    # Softmax & Accumulate with V directly in SRAM
```

2. **Constant Shared Memory Placement for Projection Buffer**: Pre-load projection matrix $\mathbf{P}$ into GPU Shared Memory (`smem`), ensuring high arithmetic intensity across all sequence tile iterations.