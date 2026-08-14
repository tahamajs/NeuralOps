# Nyströmformer Attention: Landmark-Based Kernel Approximation

## 1. Abstract & Theoretical Motivation
Full softmax self-attention computes an explicit $N \times N$ Gram affinity matrix $\mathbf{A} = \operatorname{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}}\right)$, creating an $\mathcal{O}(N^2)$ computational runtime and memory bottleneck. Nyströmformer (Xiong et al., 2021) bypasses this by leveraging the classic Nyström low-rank matrix approximation.

By selecting $m \ll N$ representative landmark tokens via adaptive sequence pooling, Nyströmformer factorizes the dense $N \times N$ attention matrix into three smaller matrices: $\hat{\mathbf{A}} = \mathbf{F}_1 \mathbf{F}_2^+ \mathbf{F}_3$, where $\mathbf{F}_1 \in \mathbb{R}^{N \times m}$ measures affinity between full queries and key landmarks, $\mathbf{F}_2 \in \mathbb{R}^{m \times m}$ represents landmark self-affinity, and $\mathbf{F}_3 \in \mathbb{R}^{m \times N}$ represents affinity between query landmarks and full keys. Computing the Moore-Penrose pseudo-inverse $\mathbf{F}_2^+$ via iterative Schulz updates enables matrix-associative evaluation, dropping computational complexity to strictly linear $\mathcal{O}(N m d + m^3)$.

---

## 2. Mathematical Derivations & Formulations

Given projected query, key, and value tensors $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times H \times N \times d}$, Nyströmformer first extracts landmark matrices $\tilde{\mathbf{Q}}, \tilde{\mathbf{K}} \in \mathbb{R}^{B \times H \times m \times d}$ using 1D adaptive average pooling across the sequence dimension $N$:

\[
\tilde{\mathbf{Q}} = \operatorname{AdaptiveAvgPool1D}(\mathbf{Q}, m), \quad \tilde{\mathbf{K}} = \operatorname{AdaptiveAvgPool1D}(\mathbf{K}, m)
\]

Three distinct cross-affinity matrices are constructed:

1. **Query-Landmark Key Matrix ($\mathbf{F}_1$)**:
\[
\mathbf{F}_1 = \operatorname{softmax}\left( \frac{\mathbf{Q} \tilde{\mathbf{K}}^\top}{\sqrt{d}}, \text{dim}=-1 \right) \in \mathbb{R}^{B \times H \times N \times m}
\]

2. **Landmark-Landmark Matrix ($\mathbf{F}_2$)**:
\[
\mathbf{F}_2 = \operatorname{softmax}\left( \frac{\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^\top}{\sqrt{d}}, \text{dim}=-1 \right) \in \mathbb{R}^{B \times H \times m \times m}
\]

3. **Landmark Query-Key Matrix ($\mathbf{F}_3$)**:
\[
\mathbf{S}_3 = \frac{\tilde{\mathbf{Q}} \mathbf{K}^\top}{\sqrt{d}} + \mathbf{M}_{\text{mask}}, \quad \mathbf{F}_3 = \operatorname{softmax}\left( \mathbf{S}_3, \text{dim}=-1 \right) \in \mathbb{R}^{B \times H \times m \times N}
\]

To resolve $\mathbf{F}_2^+ \in \mathbb{R}^{B \times H \times m \times m}$ without expensive SVD operations, we apply $K$ iterations of the Schulz iterative Newton-Raphson scheme:

\[
\alpha = \frac{1}{\|\mathbf{F}_2\|_1 \|\mathbf{F}_2\|_\infty + \epsilon}, \quad \mathbf{V}_0 = \alpha \mathbf{F}_2^\top
\]
\[
\mathbf{V}_{k+1} = \mathbf{V}_k (2 \mathbf{I}_m - \mathbf{F}_2 \mathbf{V}_k), \quad k = 0, \dots, K-1 \implies \mathbf{F}_2^+ = \mathbf{V}_K
\]

Using associative sequence multiplication, the final contextual output is computed without forming $N \times N$ tensors:

\[
\mathbf{P}_1 = \mathbf{F}_3 \mathbf{V} \in \mathbb{R}^{B \times H \times m \times d}
\]
\[
\mathbf{P}_2 = \mathbf{F}_2^+ \mathbf{P}_1 \in \mathbb{R}^{B \times H \times m \times d}
\]
\[
\mathbf{O} = \left( \mathbf{F}_1 \mathbf{P}_2 \right) \odot \mathbf{M}_{\text{valid}} \in \mathbb{R}^{B \times H \times N \times d}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input Hidden States X [B, N, D]
  │
  ├──> Linear Projections -> Q, K, V [B, H, N, d]
  │
  ├──> Adaptive Avg Pooling (over N)
  │    ├──> Q_landmarks = AdaptiveAvgPool1D(Q, m) ──> [B, H, m, d]
  │    └──> K_landmarks = AdaptiveAvgPool1D(K, m) ──> [B, H, m, d]
  │
  ├──> Compute Kernel Affinity Matrices:
  │    ├──> F1 = Softmax(Q @ K_landmarks^T / sqrt(d)) ────> [B, H, N, m]
  │    ├──> F2 = Softmax(Q_landmarks @ K_landmarks^T) ───> [B, H, m, m]
  │    └──> F3 = Softmax(Q_landmarks @ K^T + Mask) ──────> [B, H, m, N]
  │
  ├──> Schulz Iterative Pseudo-Inverse:
  │    └──> F2_pinv = IterativePinv(F2, iters=6) ────────> [B, H, m, m]
  │
  ├──> Associative Sequence Aggregation:
  │    ├──> Step 1: P1 = F3 @ V ─────────────────────────> [B, H, m, d]
  │    ├──> Step 2: P2 = F2_pinv @ P1 ───────────────────> [B, H, m, d]
  │    └──> Context = F1 @ P2 ───────────────────────────> [B, H, N, d]
  │
  └──> Re-mask Queries & Project Out:
       Context = Context * Valid_Mask ───────────────────> [B, N, D]
       Y = Context @ W_O ────────────────────────────────> [B, N, D]
```

### Detailed Shape Progression Table

| Step | Operation | Input Tensor Shapes | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| 1 | Projections ($W_Q, W_K, W_V$) | `X: [B, N, D]` | `Q, K, V: [B, H, N, d]` |
| 2 | Adaptive Pooling | `Q, K: [B, H, N, d]` | `Q_land, K_land: [B, H, m, d]` |
| 3 | Kernel Matrix $\mathbf{F}_1$ | `Q: [B, H, N, d]`, `K_land^T: [B, H, d, m]` | `F1: [B, H, N, m]` |
| 4 | Kernel Matrix $\mathbf{F}_2$ | `Q_land: [B, H, m, d]`, `K_land^T: [B, H, d, m]` | `F2: [B, H, m, m]` |
| 5 | Kernel Matrix $\mathbf{F}_3$ | `Q_land: [B, H, m, d]`, `K^T: [B, H, d, N]` | `F3: [B, H, m, N]` |
| 6 | Schulz Pseudo-Inverse | `F2: [B, H, m, m]` | `F2_pinv: [B, H, m, m]` |
| 7 | Step 1 Product ($\mathbf{F}_3 \mathbf{V}$) | `F3: [B, H, m, N]`, `V: [B, H, N, d]` | `P1: [B, H, m, d]` |
| 8 | Step 2 Inverse Projection | `F2_pinv: [B, H, m, m]`, `P1: [B, H, m, d]` | `P2: [B, H, m, d]` |
| 9 | Context Reconstruction | `F1: [B, H, N, m]`, `P2: [B, H, m, d]` | `Context: [B, H, N, d]` |
| 10 | Output Projection | `Context: [B, H, N, d]` | `Y: [B, N, D]` |

---

## 4. Analytical Complexity Profiling

### FLOP & MAC Counting
1. **Input & Output Linears**: $8 B N D^2$ FLOPs.
2. **Landmark Extraction**: $2 \times (B H N d)$ operations for adaptive pooling.
3. **Kernel Factor Computations**:
   - $\mathbf{F}_1 (\mathbf{Q} \tilde{\mathbf{K}}^\top)$: $2 B H N m d = 2 B N m D$ FLOPs.
   - $\mathbf{F}_2 (\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^\top)$: $2 B H m^2 d = 2 B m^2 D$ FLOPs.
   - $\mathbf{F}_3 (\tilde{\mathbf{Q}} \mathbf{K}^\top)$: $2 B H m N d = 2 B N m D$ FLOPs.
4. **Schulz Iterative Inversion**: $K \times (2 B H m^3)$ FLOPs (for $K=6$, $12 B H m^3$).
5. **Associative Sequence Multiply**:
   - $\mathbf{F}_3 \mathbf{V}$: $2 B N m D$ FLOPs.
   - $\mathbf{F}_2^+ \mathbf{P}_1$: $2 B m^2 D$ FLOPs.
   - $\mathbf{F}_1 \mathbf{P}_2$: $2 B N m D$ FLOPs.

**Total Computational Cost**:
\[
\text{Total FLOPs} = 8 B N D^2 + 8 B N m D + 4 B m^2 D + 12 B H m^3
\]

For settings where landmark count $m \ll N$ (e.g., $m=64, N=4096$), quadratic terms disappear.

### Memory Footprint & Tensor Store
- **Parameters**: $4 D^2$ floats.
- **Activation Memory**:
  - Landmark Tensors: $2 B H m d$
  - Factor Matrices: $\mathbf{F}_1 [B, H, N, m]$, $\mathbf{F}_2 [B, H, m, m]$, $\mathbf{F}_3 [B, H, m, N]$
  - Intermediate State Tensors: $\mathbf{P}_1, \mathbf{P}_2 [B, H, m, d]$
  - Peak Activation Memory: $\mathcal{O}(B N D + B H N m + B H m^2)$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

1. **Ill-Conditioning in Schulz Inversion**: The Schulz iterative update $\mathbf{V}_{k+1} = \mathbf{V}_k (2 \mathbf{I} - \mathbf{F}_2 \mathbf{V}_k)$ converges quadratically only if $\|\mathbf{I} - \mathbf{F}_2 \mathbf{V}_0\|_2 < 1$. If landmark affinity matrix $\mathbf{F}_2$ becomes rank-deficient or ill-conditioned, Schulz iterations diverge, outputting NaN activation values.
2. **Loss of Fine-Grained Token Dependencies**: Adaptive pooling mixes features across contiguous sequence windows to construct landmarks. This creates spatial blur, impairing performance on tasks requiring exact token-to-token indexing.
3. **Landmark Saturation at Small Sequence Lengths**: When sequence length $N \le m$, landmark pooling reduces to identity, adding unnecessary compute overhead compared to standard attention.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Memory Access & Vectorization Barriers
Naive PyTorch instantiates matrices $\mathbf{F}_1, \mathbf{F}_2, \mathbf{F}_3$ into HBM separately. For long sequence lengths, writing $\mathbf{F}_1 [B, H, N, m]$ and $\mathbf{F}_3 [B, H, m, N]$ stresses HBM bandwidth.

### High-Performance Triton Kernel Fusion Strategy
1. **Fused Landmark Pooling & Kernel Factor Compute**: Fuse adaptive average pooling directly into a Triton kernel. Load tiles of $\mathbf{Q}$ and $\mathbf{K}$ into SRAM registers, perform reduction to compute landmarks $\tilde{\mathbf{Q}}, \tilde{\mathbf{K}}$, and store them directly in Shared Memory (`smem`).
2. **Fused Schulz Inversion Kernel**: Invert matrix $\mathbf{F}_2$ entirely inside Shared Memory using register-level GEMM loops:

```python
# Triton Pseudocode for Fused Landmark Gemm & Inversion
@triton.jit
def _nystrom_fused_landmark_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    N, D, M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Load landmarks into SRAM [M, D]
    q_land = tl.load(Q_land_ptr + offsets_m[:, None] * D + offsets_d[None, :])
    k_land = tl.load(K_land_ptr + offsets_m[:, None] * D + offsets_d[None, :])
    
    # Compute F2 [M, M] inside registers
    f2 = tl.dot(q_land, tl.trans(k_land))
    f2_softmax = tl.softmax(f2)
    
    # On-chip Schulz Inversion Iterations inside SRAM
    v_inv = alpha * tl.trans(f2_softmax)
    for i in range(6):
        v_inv = tl.dot(v_inv, (2.0 * I - tl.dot(f2_softmax, v_inv)))
        
    # Stream sequence blocks Q [BLOCK_N, D], compute F1 tile on-the-fly
```

3. **Streamed Output Reconstruction**: Avoid storing $\mathbf{F}_1 [B, H, N, m]$ in DRAM. Instead, stream blocks of queries $\mathbf{Q}$, compute rows of $\mathbf{F}_1$ in SRAM, and immediately multiply by pre-computed state $\mathbf{P}_2 = \mathbf{F}_2^+ (\mathbf{F}_3 \mathbf{V})$.