# AdaSplash Attention: Tsallis 1.5-Entmax Regularized Sparse Activation

## 1. Abstract & Theoretical Motivation
Standard softmax self-attention utilizes exponential normalization $\operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$, assigning strictly positive non-zero probabilities to every key token across the sequence. Consequently, standard attention matrices are dense, spending compute on irrelevant tokens and suffering from noise accumulation across long context lengths.

AdaSplash Attention resolves this by replacing standard softmax activation with the Tsallis $\alpha$-entmax family, specifically 1.5-entmax (Peters et al., 2019; Correia et al., 2019). 1.5-entmax optimizes an entropy-regularized objective derived from Tsallis 1.5-entropy. Unlike softmax, 1.5-entmax projects attention logits onto a simplex boundary, introducing exact zeros into the activation distribution. This produces adaptive, sparse, and interpretable attention weights with controllable sparsity thresholds.

---

## 2. Mathematical Derivations & Formulations

For a given score vector $\mathbf{z} \in \mathbb{R}^N$, the general $\alpha$-entmax transformation solves the constrained optimization problem:

\[
\operatorname{entmax}_\alpha(\mathbf{z}) = \arg\max_{\mathbf{p} \in \Delta^N} \left( \mathbf{p}^\top \mathbf{z} + H_\alpha(\mathbf{p}) \right)
\]

where $\Delta^N = \{ \mathbf{p} \in \mathbb{R}_+^N \mid \sum_{i=1}^N p_i = 1 \}$ is the probability simplex, and $H_\alpha(\mathbf{p})$ is the Tsallis $\alpha$-entropy defined for $\alpha = 1.5$ as:

\[
H_{1.5}(\mathbf{p}) = \frac{1}{1.5 (1.5 - 1)} \left( 1 - \sum_{i=1}^N p_i^{1.5} \right) = 4 \left( 1 - \sum_{i=1}^N p_i^{1.5} \right)
\]

Solving the KKT dual formulation yields the closed-form thresholded mapping:

\[
\operatorname{entmax}_{1.5}(\mathbf{z}) = \left[ \frac{1}{2} (\mathbf{z} - \tau^* \mathbf{1}_N) \right]_+^2
\]

where $[\cdot]_+ = \max(\cdot, 0)$, and $\tau^* \in \mathbb{R}$ is the unique Lagrange multiplier enforcing normalization $\sum_{i=1}^N p_i = 1$.

To evaluate $\operatorname{entmax}_{1.5}$ efficiently on modern hardware without explicit sorting, AdaSplash implements an iterative fixed-point normalization scheme. Starting from initial softmax probabilities $\mathbf{p}^{(0)} = \operatorname{softmax}(\mathbf{z})$, the algorithm updates:

\[
\mathbf{q}^{(t)} = \sqrt{\mathbf{p}^{(t)}}, \quad \tau^{(t)} = \sum_{i=1}^N q_i^{(t)} p_i^{(t)}, \quad \mathbf{p}^{(t+1)} = \frac{\mathbf{q}^{(t)}}{\tau^{(t)} + \epsilon}
\]

Iterating for $T_{\text{iter}}$ steps converges rapidly to the target sparse distribution $\mathbf{p}^* = \operatorname{entmax}_{1.5}(\mathbf{z})$.

Given input $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$, query, key, and value matrices are computed:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{B \times H \times N \times d}, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{B \times H \times N \times d}, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{B \times H \times N \times d}
\]

Scaled similarity logits $\mathbf{S} \in \mathbb{R}^{B \times H \times N \times N}$ are computed and masked:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}} + (1 - \mathbf{M}_{\text{float}}) \cdot (-10^9)
\]

Applying 1.5-entmax yields sparse attention weights $\mathbf{A} \in \mathbb{R}^{B \times H \times N \times N}$, followed by context readout:

\[
\mathbf{A} = \operatorname{entmax}_{1.5}(\mathbf{S}), \quad \mathbf{O} = (\mathbf{A} \mathbf{V}) \mathbf{W}_O \in \mathbb{R}^{B \times N \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input Hidden States X [B, N, D]
  │
  ├──> Projections (W_Q, W_K, W_V) -> Q, K, V [B, H, N, d]
  │
  ├──> Similarity Matrix: S = Q @ K^T / sqrt(d) ──────> [B, H, N, N]
  │
  ├──> Apply Mask: S = S + (1 - Mask) * (-1e9) ─────────> [B, H, N, N]
  │
  ├──> Iterative 1.5-Entmax Fixed-Point Normalization (T_iter steps):
  │    ├──> Initial Probabilities: p^(0) = Softmax(S, dim=-1) -> [B, H, N, N]
  │    ├──> q^(t) = sqrt(p^(t)) ────────────────────────> [B, H, N, N]
  │    ├──> tau^(t) = sum(q^(t) * p^(t), dim=-1) ────────> [B, H, N, 1]
  │    └──> p^(t+1) = q^(t) / (tau^(t) + eps) ──────────> [B, H, N, N]
  │
  ├──> Dropout: A = Dropout(p^(final)) ─────────────────> [B, H, N, N]
  │
  └──> Context Readout & Out Projection:
       Context = A @ V ─────────────────────────────────> [B, H, N, d]
       Y = Context @ W_O ───────────────────────────────> [B, N, D]
```

### Detailed Shape Progression Table

| Step | Operation | Input Tensor Shapes | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| 1 | Projections ($W_Q, W_K, W_V$) | `X: [B, N, D]` | `Q, K, V: [B, N, D]` |
| 2 | Reshape to Multi-Head | `Q, K, V: [B, N, D]` | `Q, K, V: [B, H, N, d]` |
| 3 | Query-Key MatMul | `Q: [B, H, N, d]`, `K^T: [B, H, d, N]` | `S: [B, H, N, N]` |
| 4 | Additive Masking | `S: [B, H, N, N]`, `Mask: [B, 1, 1, N]` | `S_masked: [B, H, N, N]` |
| 5 | Initial Softmax $p^{(0)}$ | `S_masked: [B, H, N, N]` | `p_0: [B, H, N, N]` |
| 6 | Fixed-Point Iteration $t$ | `p_t: [B, H, N, N]` | `p_{t+1}: [B, H, N, N]` |
| 7 | Context Matrix Product | `A: [B, H, N, N]`, `V: [B, H, N, d]` | `Context: [B, H, N, d]` |
| 8 | Output Linear Projection | `Context: [B, H, N, d]` | `Y: [B, N, D]` |

---

## 4. Analytical Complexity Profiling

### FLOP & MAC Counting
1. **Linear Projections ($W_Q, W_K, W_V, W_O$)**: $8 B N D^2$ FLOPs.
2. **Score Matrix Product ($\mathbf{Q} \mathbf{K}^\top$)**: $2 B H N^2 d = 2 B N^2 D$ FLOPs.
3. **Fixed-Point Entmax Iterations ($T_{\text{iter}}$ steps)**:
   - Initial Softmax: $3 B H N^2$ FLOPs.
   - Square root ($\sqrt{p}$): $B H N^2$ FLOPs per step.
   - Reduction sum ($\tau = \sum q_i p_i$): $2 B H N^2$ FLOPs per step.
   - Element-wise division ($q / \tau$): $B H N^2$ FLOPs per step.
   - Total Entmax Compute: $B H N^2 (3 + 4 T_{\text{iter}})$.
4. **Context Matrix Product ($\mathbf{A} \mathbf{V}$)**: $2 B H N^2 d = 2 B N^2 D$ FLOPs.

**Total Computational Cost**:
\[
\text{Total FLOPs} = 8 B N D^2 + 4 B N^2 D + B H N^2 (3 + 4 T_{\text{iter}})
\]

For $T_{\text{iter}}=10$, entmax iteration adds $\sim 43 B H N^2$ element-wise operations on top of matrix products.

### Memory Footprint & Tensor Store
- **Parameters**: $4 D^2$ floats.
- **Activation Memory**:
  - Scores & Probabilities ($\mathbf{S}, \mathbf{p}^{(t)}, \mathbf{q}^{(t)}$): $3 \times B H N^2$ floats.
  - Tau Normalization Vector ($\boldsymbol{\tau}^{(t)}$): $B H N$ floats.
  - Peak Activation Memory: $\mathcal{O}(B N D + B H N^2)$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

1. **Fixed-Point Iteration Overhead**: Running $T_{\text{iter}}=10$ sequential loops across full $N \times N$ activation tensors introduces un-fused DRAM access overhead, slowing down training relative to fused CUDA FlashAttention kernels.
2. **Simplex Saturation & Gradient Vanishing**: When scores $\mathbf{z}$ contain extreme variance, entmax projects weights directly onto the boundary of the simplex (exact zero weights). For zeroed positions, gradients $\frac{\partial \mathbf{p}_i}{\partial \mathbf{z}_j} = 0$, halting weight updates for those tokens.
3. **Quadratic Scaling Limit**: Because AdaSplash computes explicit $N \times N$ score matrices prior to sparse entmax mapping, space and time complexities remain quadratic $\mathcal{O}(N^2)$.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Memory Access & Vectorization Barriers
Naive PyTorch implementations load and store full $B \times H \times N \times N$ attention tensors $T_{\text{iter}}$ times back and forth from GPU HBM during entmax updates, bottlenecking memory bandwidth.

### High-Performance Triton Kernel Fusion Strategy
1. **Fused SRAM Bisection / Fixed-Point Entmax**: Implement a fused Triton block-kernel that loads tiles of $\mathbf{Q}$ and $\mathbf{K}$ into SRAM registers, computes dot-product scores, and executes fixed-point entmax iterations entirely within shared memory without writing intermediate $N \times N$ tensors to DRAM.

```python
# Triton Pseudocode for Fused Entmax Flash Kernel
@triton.jit
def _adasplash_fused_entmax_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    N, D, BLOCK_N: tl.constexpr, T_ITER: tl.constexpr
):
    # Load Query block [BLOCK_N, D]
    q = tl.load(Q_ptr + offsets_q)
    # Stream Key blocks [BLOCK_N, D] & Compute scores in SRAM
    scores = tl.dot(q, tl.trans(k)) * scale
    
    # Execute Entmax Fixed-Point loop inside SRAM registers
    p = tl.softmax(scores, dim=-1)
    for i in range(T_ITER):
        q_sqrt = tl.sqrt(p)
        tau = tl.sum(q_sqrt * p, axis=-1, keep_dims=True)
        p = q_sqrt / (tau + 1e-12)
        
    # Multiply by V directly in SRAM registers
    acc += tl.dot(p, v)
```

2. **Block-Sparse SRAM Skip**: Because entmax produces exact zeros, tiles where $\max(\mathbf{z}) < \tau^*$ can be dynamically pruned in SRAM, skipping the downstream matrix product with $\mathbf{V}$ to save tensor core cycles.