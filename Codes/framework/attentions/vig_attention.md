# ViG Attention: 2D Grid Kernelized Linear Attention with Input-Conditioned Gated Fusion

## 1. Abstract & Theoretical Motivation
Standard 1D sequence attention computes full pairwise affinity matrices across sequence length $N$, leading to quadratic computational and memory scaling $\mathcal{O}(N^2)$. Vision GNN (ViG) inspired attention restructures 1D token sequences into a dynamic 2D spatial grid layout $H_g \times W_g \approx \sqrt{N} \times \sqrt{N}$.

By decomposing global 2D sequence interactions into decoupled 1D row-wise and 1D column-wise linear attention passes using positive ELU kernel feature maps $\phi(\mathbf{x}) = \operatorname{ELU}(\mathbf{x}) + 1$, ViG Attention models both horizontal and vertical contextual dependencies. Dynamic input-conditioned gating mechanisms $g_{\text{row}}, g_{\text{col}}$ fuse row and column representations adaptively, delivering linear time and memory complexity $\mathcal{O}(N \sqrt{N} d) \approx \mathcal{O}(N)$.

---

## 2. Mathematical Derivations & Formulations

Given an input representation $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$, sequence length $N$ is reshaped into a 2D spatial grid of dimensions $H_g = \lfloor \sqrt{N} \rfloor$ and $W_g = \lceil N / H_g \rceil$. If $H_g \times W_g > N$, zero-padding is appended to create $\mathbf{X}_{\text{pad}} \in \mathbb{R}^{B \times N_{\text{pad}} \times D}$.

Linear projections generate query, key, and value tensors:

\[
\mathbf{Q} = \mathbf{X}_{\text{pad}} \mathbf{W}_Q \in \mathbb{R}^{B \times H \times H_g \times W_g \times d}, \quad \mathbf{K} = \mathbf{X}_{\text{pad}} \mathbf{W}_K \in \mathbb{R}^{B \times H \times H_g \times W_g \times d}, \quad \mathbf{V} = \mathbf{X}_{\text{pad}} \mathbf{W}_V \in \mathbb{R}^{B \times H \times H_g \times W_g \times d}
\]

Simultaneously, input-conditioned dynamic fusion gates are derived via linear transformation and sigmoid activation:

\[
\mathbf{G} = \sigma\left( \mathbf{X}_{\text{pad}} \mathbf{W}_{\text{gate}} \right) \in \mathbb{R}^{B \times N_{\text{pad}} \times 2 \times H \times 1} \implies (\mathbf{g}_{\text{row}}, \mathbf{g}_{\text{col}})
\]

Non-negative kernel maps are applied to queries and keys:

\[
\phi(\mathbf{x}) = \operatorname{ELU}(\mathbf{x}) + 1 \in \mathbb{R}_+^d
\]

### Row-Wise Linear Attention Pass
Along each column slice $w \in \{1, \dots, W_g\}$, row tokens interact via linear associative kernel multiplication:

\[
\mathbf{K}\mathbf{V}_{\text{row}}^{(w)} = \sum_{h=1}^{H_g} \phi\left( \mathbf{k}_{h, w} \right) \mathbf{v}_{h, w}^\top \in \mathbb{R}^{d \times d}
\]
\[
\mathbf{z}_{\text{row}}^{(w)} = \sum_{h=1}^{H_g} \phi\left( \mathbf{k}_{h, w} \right) \in \mathbb{R}^d
\]
\[
\mathbf{O}_{\text{row}, h, w} = \frac{\phi\left( \mathbf{q}_{h, w} \right) \mathbf{K}\mathbf{V}_{\text{row}}^{(w)}}{\phi\left( \mathbf{q}_{h, w} \right) \mathbf{z}_{\text{row}}^{(w)} + \epsilon} \in \mathbb{R}^d
\]

### Column-Wise Linear Attention Pass
Along each row slice $h \in \{1, \dots, H_g\}$, column tokens interact symmetrically:

\[
\mathbf{K}\mathbf{V}_{\text{col}}^{(h)} = \sum_{w=1}^{W_g} \phi\left( \mathbf{k}_{h, w} \right) \mathbf{v}_{h, w}^\top \in \mathbb{R}^{d \times d}
\]
\[
\mathbf{z}_{\text{col}}^{(h)} = \sum_{w=1}^{W_g} \phi\left( \mathbf{k}_{h, w} \right) \in \mathbb{R}^d
\]
\[
\mathbf{O}_{\text{col}, h, w} = \frac{\phi\left( \mathbf{q}_{h, w} \right) \mathbf{K}\mathbf{V}_{\text{col}}^{(h)}}{\phi\left( \mathbf{q}_{h, w} \right) \mathbf{z}_{\text{col}}^{(h)} + \epsilon} \in \mathbb{R}^d
\]

### Gated Spatial Context Fusion
Row and column contexts are dynamically fused using the input-conditioned gating factors:

\[
\mathbf{O}_{\text{fused}, h, w} = \mathbf{g}_{\text{row}, h, w} \odot \mathbf{O}_{\text{row}, h, w} + \mathbf{g}_{\text{col}, h, w} \odot \mathbf{O}_{\text{col}, h, w} \in \mathbb{R}^d
\]

The grid tensor is flattened back to sequence dimension $N_{\text{pad}}$, unpadded to length $N$, and linearly projected:

\[
\mathbf{Y} = \operatorname{Concat}(\mathbf{O}_{\text{fused}}) \mathbf{W}_O \in \mathbb{R}^{B \times N \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input Hidden States X [B, N, D]
  │
  ├──> Pad Sequence to 2D Grid: N -> N_pad = H_g * W_g
  │    └──> X_pad [B, N_pad, D]
  │
  ├──> Linear Projections (W_Q, W_K, W_V, W_gate)
  │    ├──> Q, K, V [B, H, N_pad, d] ──> Reshape ──> [B, H, H_g, W_g, d]
  │    └──> Gate = Sigmoid(X @ W_gate) ────────────> [B, N_pad, 2, H, 1]
  │
  ├──> Row-Wise Linear Attention (along H_g dimension):
  │    ├──> Feature Map: phi(Q_row), phi(K_row) = ELU + 1
  │    ├──> KV_row = Einsum('b m n d, b m n e -> b m d e') ──> [B, H*W_g, d, d]
  │    └──> Out_row = (phi(Q) @ KV_row) / Denom ─────────────> [B, H, H_g, W_g, d]
  │
  ├──> Column-Wise Linear Attention (along W_g dimension):
  │    ├──> Feature Map: phi(Q_col), phi(K_col) = ELU + 1
  │    ├──> KV_col = Einsum('b n d, b n e -> b d e') ────────> [B*H*H_g, d, d]
  │    └──> Out_col = (phi(Q) @ KV_col) / Denom ─────────────> [B, H, H_g, W_g, d]
  │
  ├──> Gated Fusion: Out = g_row * Out_row + g_col * Out_col ──> [B, H, H_g, W_g, d]
  │
  └──> Reshape to Sequence & Unpad:
       Out_seq = Reshape(Out) [B, N_pad, D] -> Slice[:N] ─────> [B, N, D]
       Y = Out_seq @ W_O ────────────────────────────────────> [B, N, D]
```

### Detailed Shape Progression Table

| Step | Operation | Input Tensor Shapes | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| 1 | Sequence Grid Padding | `X: [B, N, D]` | `X_pad: [B, N_pad, D]` |
| 2 | Projections ($W_Q, W_K, W_V$) | `X_pad: [B, N_pad, D]` | `Q, K, V: [B, N_pad, D]` |
| 3 | 2D Grid Layout Reshape | `Q, K, V: [B, N_pad, D]` | `Q, K, V: [B, H, H_g, W_g, d]` |
| 4 | Dynamic Gating Computation | `X_pad: [B, N_pad, D]` | `gates: [B, N_pad, 2, H, 1]` |
| 5 | Row Feature Projection | `Q, K: [B, H*W_g, H_g, d]` | `Q_row, K_row: [B, H*W_g, H_g, d]` |
| 6 | Row State Outer Product | `K_row: [B, H*W_g, H_g, d]`, `V_row: ...` | `KV_row: [B, H*W_g, d, d]` |
| 7 | Row Linear Output | `Q_row: [B, H*W_g, H_g, d]`, `KV_row` | `Out_row: [B, H, H_g, W_g, d]` |
| 8 | Column State Outer Product | `K_col: [B*H*H_g, W_g, d]`, `V_col: ...` | `KV_col: [B*H*H_g, d, d]` |
| 9 | Column Linear Output | `Q_col: [B*H*H_g, W_g, d]`, `KV_col` | `Out_col: [B, H, H_g, W_g, d]` |
| 10 | Gated Combination | `Out_row, Out_col, gates` | `Out_fused: [B, H, H_g, W_g, d]` |
| 11 | Unpadding & Projection | `Out_fused: [B, H, H_g, W_g, d]` | `Y: [B, N, D]` |

---

## 4. Analytical Complexity Profiling

### FLOP & MAC Counting
1. **Linear Projections ($W_Q, W_K, W_V, W_O, W_{\text{gate}}$)**: $8 B N D^2 + 2 B N D (2 H)$ FLOPs.
2. **Row Linear Attention**:
   - $\mathbf{K}\mathbf{V}_{\text{row}}$ outer product: $2 B H W_g H_g d^2 = 2 B N D d$ FLOPs.
   - Context readout: $2 B H W_g H_g d^2 = 2 B N D d$ FLOPs.
3. **Column Linear Attention**:
   - $\mathbf{K}\mathbf{V}_{\text{col}}$ outer product: $2 B H H_g W_g d^2 = 2 B N D d$ FLOPs.
   - Context readout: $2 B H H_g W_g d^2 = 2 B N D d$ FLOPs.
4. **Gated Fusion & Normalization**: $4 B N D$ FLOPs.

**Total Computational Cost**:
\[
\text{Total FLOPs} = 8 B N D^2 + 8 B N D d + 4 B N D H
\]

Decoupling sequence attention into 2D orthogonal grid passes drops compute scaling from quadratic $\mathcal{O}(N^2)$ to linear $\mathcal{O}(N d)$.

### Memory Footprint & Tensor Store
- **Parameters**: $4 D^2 + 2 H D$ floats.
- **Activation Memory**:
  - Grid Tensors ($\mathbf{Q}, \mathbf{K}, \mathbf{V}$): $3 B H N_{\text{pad}} d = 3 B N_{\text{pad}} D$ floats.
  - State Tensors ($\mathbf{K}\mathbf{V}_{\text{row}}, \mathbf{K}\mathbf{V}_{\text{col}}$): $B H W_g d^2 + B H H_g d^2 = B D d (H_g + W_g)$ floats.
  - Gating Weights: $2 B N_{\text{pad}} H$ floats.
  - Peak Memory: $\mathcal{O}(B N D + B D d \sqrt{N})$.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

1. **Aspect Ratio Imbalance**: For non-square sequence lengths (e.g., $N=1000 \implies H_g=31, W_g=33$), zero-padding overhead increases memory allocation. Extreme aspect ratio mismatches weaken either horizontal or vertical context propagation.
2. **Diagonal Disconnect in Decoupled Pass**: Because row and column passes run in parallel without cross-diagonal interaction terms, information flow between top-left and bottom-right grid cells requires multi-layer stacking.
3. **Feature Map Non-negativity Noise**: Applying $\operatorname{ELU}(\mathbf{x}) + 1$ maps negative values to small positive numbers near $0$. For large inputs, this feature map suffers from numerical saturation, reducing linear attention selectivity.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Memory Access & Vectorization Barriers
Naive PyTorch evaluates row and column passes via multiple separate `torch.einsum` calls, writing intermediate $d \times d$ outer product state tensors back to DRAM multiple times.

### High-Performance Triton Kernel Fusion Strategy
1. **Fused 2D Grid Block Reduction Kernel**: Implement a Triton kernel that loads a 2D spatial tile $[BLOCK\_H_g, BLOCK\_W_g]$ into Shared Memory (`smem`). Compute both row and column linear attention passes simultaneously in SRAM registers.

```python
# Triton Pseudocode for Fused 2D Grid Linear Attention
@triton.jit
def _vig_fused_2d_grid_kernel(
    Q_ptr, K_ptr, V_ptr, Gate_ptr, Out_ptr,
    Hg, Wg, D, d: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Load 2D block of Q, K, V into SRAM
    q_tile = tl.load(Q_ptr + offsets_h[:, None] * Wg * d + offsets_w[None, :] * d)
    # Compute ELU+1 feature map on-the-fly in SRAM registers
    phi_q = tl.where(q_tile > 0, q_tile + 1.0, tl.exp(q_tile))
    
    # Compute Row State KV_row = phi_k.T @ v in registers
    # Compute Col State KV_col = phi_k.T @ v in registers
    
    # Fuse row readout, col readout, and sigmoid gating inside registers
    out = g_row * (phi_q @ KV_row) + g_col * (phi_q @ KV_col)
    tl.store(Out_ptr + offsets, out)
```

2. **Shared Memory Reuse for Gating Coefficients**: Keep gating weights $g_{\text{row}}, g_{\text{col}}$ inside SRAM registers during context readout, avoiding separate global memory fetches.