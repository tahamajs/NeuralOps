# Primal Attention: Asymmetric Kernel SVD in Primal Representation

## 1. Abstract & Theoretical Motivation
Standard self-attention constructs an explicit $N \times N$ Gram matrix $\mathbf{A} = \operatorname{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}}\right)$, which incurs quadratic time and space complexities $\mathcal{O}(N^2)$. Primal Attention resolves this by projecting the full sequence attention operator onto an asymmetric primal landmark kernel space.

By extracting $r \ll N$ landmark query and key representations via segment-wise mean pooling or uniform downsampling, Primal Attention decomposes the dense score matrix into three lower-dimensional factor matrices: $\mathbf{F}_1 \in \mathbb{R}^{N \times r}$, $\mathbf{F}_2 \in \mathbb{R}^{r \times r}$, and $\mathbf{F}_3 \in \mathbb{R}^{r \times N}$. To guarantee stability, Moore-Penrose pseudo-inversion of the landmark kernel $\mathbf{F}_2$ is refined using $T$ iterations of Newton-Schulz matrix updates. This provides linear computational complexity $\mathcal{O}(N r d + T r^3)$.

---

## 2. Mathematical Derivations & Formulations

Given input sequence representations $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$, multi-head query, key, and value matrices are computed:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{B \times H \times N \times d}, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{B \times H \times N \times d}, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{B \times H \times N \times d}
\]

### 1. Landmark Token Extraction
To construct landmark matrices $\tilde{\mathbf{Q}}, \tilde{\mathbf{K}} \in \mathbb{R}^{B \times H \times r \times d}$, sequence length $N$ is partitioned into $r$ non-overlapping segments of size $S = N / r$ (if $N \pmod r = 0$):

\[
\tilde{\mathbf{Q}}_{b, h, k, :} = \frac{1}{S} \sum_{m=(k-1)S+1}^{k S} \mathbf{Q}_{b, h, m, :}, \quad \tilde{\mathbf{K}}_{b, h, k, :} = \frac{1}{S} \sum_{m=(k-1)S+1}^{k S} \mathbf{K}_{b, h, m, :}
\]

If $N \pmod r \neq 0$, landmark indices are selected via uniform linear spacing.

### 2. Tripartite Kernel Factorization
Three low-rank affinity factors are calculated:

\[
\mathbf{F}_1 = \frac{\mathbf{Q} \tilde{\mathbf{K}}^\top}{\sqrt{d}} \in \mathbb{R}^{B \times H \times N \times r}
\]
\[
\mathbf{F}_2 = \frac{\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^\top}{\sqrt{d}} \in \mathbb{R}^{B \times H \times r \times r}
\]
\[
\mathbf{F}_3 = \frac{\tilde{\mathbf{Q}} \mathbf{K}^\top}{\sqrt{d}} \in \mathbb{R}^{B \times H \times r \times N}
\]

### 3. Iterative Newton-Schulz Pseudoinverse Refinement
Starting from initial Moore-Penrose pseudo-inverse $\mathbf{X}_0 = \mathbf{F}_2^+ \in \mathbb{R}^{B \times H \times r \times r}$, $T$ iterations of Schulz matrix refinement are performed:

\[
\mathbf{X}_{t+1} = \mathbf{X}_t (2 \mathbf{I}_r - \mathbf{F}_2 \mathbf{X}_t), \quad t = 0, \dots, T-1 \implies \mathbf{F}_2^{\dagger} = \mathbf{X}_T
\]

### 4. Attention Score Reconstruction & Context Readout
The low-rank score matrix is reconstructed via:

\[
\mathbf{S} = \mathbf{F}_1 \mathbf{F}_2^{\dagger} \mathbf{F}_3 + \mathbf{M}_{\text{mask}} \in \mathbb{R}^{B \times H \times N \times N}
\]
\[
\mathbf{A} = \operatorname{softmax}(\mathbf{S}, \text{dim}=-1), \quad \mathbf{O} = (\mathbf{A} \mathbf{V}) \mathbf{W}_O \in \mathbb{R}^{B \times N \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input Hidden States X [B, N, D]
  │
  ├──> Linear Projections -> Q, K, V [B, H, N, d]
  │
  ├──> Segment Pooling Landmarks (if N % r == 0):
  │    ├──> Q_land = Reshape(Q, [B, H, r, seg, d]).mean(dim=3) ──> [B, H, r, d]
  │    └──> K_land = Reshape(K, [B, H, r, seg, d]).mean(dim=3) ──> [B, H, r, d]
  │
  ├──> Compute Low-Rank Affinity Factors:
  │    ├──> F1 = Q @ K_land^T / sqrt(d) ─────────────────────────> [B, H, N, r]
  │    ├──> F2 = Q_land @ K_land^T / sqrt(d) ────────────────────> [B, H, r, r]
  │    └──> F3 = Q_land @ K^T / sqrt(d) ─────────────────────────> [B, H, r, N]
  │
  ├──> Schulz Refinement:
  │    ├──> X_0 = pinv(F2) ──────────────────────────────────────> [B, H, r, r]
  │    └──> Loop T steps: X_{t+1} = X_t @ (2I - F2 @ X_t) ───────> [B, H, r, r]
  │
  ├──> Reconstruct Score Matrix & Softmax:
  │    ├──> Score = F1 @ (F2_pinv @ F3) + Mask ──────────────────> [B, H, N, N]
  │    └──> A = Softmax(Score, dim=-1) ──────────────────────────> [B, H, N, N]
  │
  └──> Context Aggregation & Out Projection:
       Context = A @ V ──────────────────────────────────────────> [B, H, N, d]
       Y = Reshape(Context) @ W_O ───────────────────────────────> [B, N, D]
```

### Detailed Shape Progression Table

| Step | Operation | Input Tensor Shapes | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| 1 | Projections ($W_Q, W_K, W_V$) | `X: [B, N, D]` | `Q, K, V: [B, N, D]` |
| 2 | Multi-Head Reshape | `Q, K, V: [B, N, D]` | `Q, K, V: [B, H, N, d]` |
| 3 | Segment Pooling Landmarks | `Q, K: [B, H, N, d]` | `Q_land, K_land: [B, H, r, d]` |
| 4 | Factor $\mathbf{F}_1$ Computation | `Q: [B, H, N, d]`, `K_land^T: [B, H, d, r]` | `F1: [B, H, N, r]` |
| 5 | Factor $\mathbf{F}_2$ Computation | `Q_land: [B, H, r, d]`, `K_land^T: [B, H, d, r]` | `F2: [B, H, r, r]` |
| 6 | Factor $\mathbf{F}_3$ Computation | `Q_land: [B, H, r, d]`, `K^T: [B, H, d, N]` | `F3: [B, H, r, N]` |
| 7 | Initial Pseudo-Inverse | `F2: [B, H, r, r]` | `X0: [B, H, r, r]` |
| 8 | Schulz Iterative Refinement | `X0, F2: [B, H, r, r]` | `F2_pinv: [B, H, r, r]` |
| 9 | Low-Rank Score Product | `F1: [B, H, N, r]`, `F2_pinv`, `F3: [B, H, r, N]` | `Score: [B, H, N, N]` |
| 10 | Softmax & Output Projection | `Score: [B, H, N, N]`, `V: [B, H, N, d]` | `Y: [B, N, D]` |

---

## 4. Analytical Complexity Profiling

### FLOP & MAC Counting
1. **Linear Projections ($W_Q, W_K, W_V, W_O$)**: $8 B N D^2$ FLOPs.
2. **Segment Pooling**: $2 \times (B H N d) = 2 B N D$ additions.
3. **Factor Computations**:
   - $\mathbf{F}_1 (\mathbf{Q} \tilde{\mathbf{K}}^\top)$: $2 B H N r d = 2 B N r D$ FLOPs.
   - $\mathbf{F}_2 (\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^\top)$: $2 B H r^2 d = 2 B r^2 D$ FLOPs.
   - $\mathbf{F}_3 (\tilde{\mathbf{Q}} \mathbf{K}^\top)$: $2 B H r N d = 2 B N r D$ FLOPs.
4. **Schulz Iterations ($T$ steps)**: $T \times (2 B H r^3 + 2 B H r^3) = 4 B H T r^3$ FLOPs.
5. **Reconstruction Product ($\mathbf{F}_1 \mathbf{F}_2^\dagger \mathbf{F}_3$)**: $2 B H N r^2 + 2 B H N^2 r$ FLOPs.

**Total Computational Cost**:
\[
\text{Total FLOPs} = 8 B N D^2 + 4 B N r D + 2 B r^2 D + 4 B H T r^3 + 2 B H N^2 r
\]

When $r \ll N$, landmark factorization reduces inner projection compute.

### Memory Footprint & Tensor Store
- **Parameters**: $4 D^2$ floats.
- **Activation Memory**:
  - Factor Matrices ($\mathbf{F}_1 [B, H, N, r]$, $\mathbf{F}_2 [B, H, r, r]$, $\mathbf{F}_3 [B, H, r, N]$): $B H N r + B H r^2 + B H r N$ floats.
  - Pseudo-Inverse State: $B H r^2$ floats.
  - Reconstructed Score Matrix: $B H N^2$ floats.
  - Peak Activation Memory: $\mathcal{O}(B N D + B H N r + B H N^2)$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

1. **Instability in Double Inversion**: Calling `torch.linalg.pinv(F2)` followed by $T$ Schulz iterations creates redundant inverse steps. If $\mathbf{F}_2$ is ill-conditioned, Schulz updates diverge exponentially, generating NaN logits.
2. **Persistent Quadratic Memory Footprint**: Because the code computes $\mathbf{S} = \mathbf{F}_1 \mathbf{F}_2^\dagger \mathbf{F}_3 \in \mathbb{R}^{B \times H \times N \times N}$ prior to softmax, the attention matrix remains $N \times N$, retaining quadratic space complexity $\mathcal{O}(N^2)$.
3. **Segment Mean Blur**: Mean pooling across sequence segments produces spatial blur, weakening fine-grained token-level cross-attention.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Memory Access & Vectorization Barriers
In standard PyTorch, materializing the full score matrix $\mathbf{S} = \mathbf{F}_1 \mathbf{F}_2^\dagger \mathbf{F}_3$ writes $B \times H \times N \times N$ elements to DRAM before applying softmax.

### High-Performance Triton Kernel Fusion Strategy
1. **Associative Sequence Multiplication without $N \times N$ Scores**: Avoid reconstructing $\mathbf{S} \in \mathbb{R}^{N \times N}$. If kernel linear attention is used, re-associate the product as $\mathbf{O} = \mathbf{F}_1 \left( \mathbf{F}_2^\dagger (\mathbf{F}_3 \mathbf{V}) \right)$, executing the pass in $\mathcal{O}(N r d)$ time and space.
2. **Fused Landmark Reduction & Schulz Kernel**: Implement a Triton kernel that loads query/key blocks, pools landmarks into Shared Memory (`smem`), performs Schulz inversion in registers, and streams sequence outputs:

```python
# Triton Pseudocode for Fused Primal Landmark Kernel
@triton.jit
def _primal_fused_landmark_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    N, D, r: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Load landmarks into SRAM [r, D]
    q_land = tl.load(Q_land_ptr + offsets_r[:, None] * D + offsets_d[None, :])
    k_land = tl.load(K_land_ptr + offsets_r[:, None] * D + offsets_d[None, :])
    
    # Compute F2 [r, r] and execute Schulz inverse inside registers
    f2 = tl.dot(q_land, tl.trans(k_land)) * scale
    f2_inv = tl.pinv(f2)
    for t in range(6):
        f2_inv = tl.dot(f2_inv, (2.0 * I - tl.dot(f2, f2_inv)))
        
    # Stream sequence blocks Q [BLOCK_N, D], multiply by (F2_inv @ F3 @ V)
```

3. **In-Register Schulz Inversion**: Keep landmark matrix $\mathbf{F}_2 \in \mathbb{R}^{r \times r}$ inside GPU Shared Memory (`smem`), enabling ultra-fast register-level Schulz GEMM updates.