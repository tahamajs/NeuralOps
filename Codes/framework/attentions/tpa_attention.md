# Tensor Product Attention (TPA): Factorized Multiplicative Query-Key Subspace Expansion

## 1. Abstract & Theoretical Motivation

Standard Multi-Head Attention (MHA) computes attention scores using linear projections that capture basic inner products between query and key vectors. Tensor Product Attention (TPA) extends this formulation by constructing queries and keys through the tensor outer product of two lower-rank factorized sub-vectors:

\[
q_{h, i} = q_{1, h, i} \otimes q_{2, h, i} \in \mathbb{R}^{r^2}, \quad k_{h, j} = k_{1, h, j} \otimes k_{2, h, j} \in \mathbb{R}^{r^2}
\]

where $r$ is the factorized subspace rank ($r \ll d_k$).

By factorizing the projection parameters into dual linear streams ($W_{Q_1}, W_{Q_2}$ and $W_{K_1}, W_{K_2}$), TPA parameterizes an effective $r^2$-dimensional attention space using only $2 \times H \cdot r \cdot D$ parameters per stream. This outer-product expansion generates higher-order multiplicative polynomial similarity kernels between token states, providing richer feature interaction manifolds at reduced linear projection cost.

---

## 2. Mathematical Derivations & Formulations

Let $X \in \mathbb{R}^{B \times L \times D}$ be the input activation sequence, where $B$ is batch size, $L$ is sequence length, and $D$ is model dimension. Let $H$ be the number of attention heads, $d_k = D / H$ the head dimension, and $r$ the factorized rank per head.

### Dual Low-Rank Subspace Factorization Projections:
The input hidden states are mapped into paired rank-$r$ subspaces for queries and keys:

\[
q_1 = X W_{Q1} \in \mathbb{R}^{B \times L \times (H \cdot r)}, \quad W_{Q1} \in \mathbb{R}^{D \times (H \cdot r)}
\]
\[
q_2 = X W_{Q2} \in \mathbb{R}^{B \times L \times (H \cdot r)}, \quad W_{Q2} \in \mathbb{R}^{D \times (H \cdot r)}
\]
\[
k_1 = X W_{K1} \in \mathbb{R}^{B \times L \times (H \cdot r)}, \quad W_{K1} \in \mathbb{R}^{D \times (H \cdot r)}
\]
\[
k_2 = X W_{K2} \in \mathbb{R}^{B \times L \times (H \cdot r)}, \quad W_{K2} \in \mathbb{R}^{D \times (H \cdot r)}
\]

Value states use standard multi-head linear projection:

\[
V = X W_V \in \mathbb{R}^{B \times L \times D}, \quad W_V \in \mathbb{R}^{D \times D}
\]

Reshaping yields $q_1, q_2, k_1, k_2 \in \mathbb{R}^{B \times L \times H \times r}$ and $V \in \mathbb{R}^{B \times H \times L \times d_k}$.

### Tensor Outer-Product Subspace Synthesis:
For query head $h$ at token position $i$, the effective query vector $Q_{h, i} \in \mathbb{R}^{r^2}$ is synthesized via Kronecker / outer product:

\[
(Q_{h, i})_{a, b} = (q_{1, h, i})_a \cdot (q_{2, h, i})_b, \quad a, b \in \{1, \dots, r\}
\]
\[
Q_{h, i} = \text{Vec}(q_{1, h, i} \otimes q_{2, h, i}) \in \mathbb{R}^{r^2}
\]

Similarly, for key head $h$ at token position $j$:

\[
K_{h, j} = \text{Vec}(k_{1, h, j} \otimes k_{2, h, j}) \in \mathbb{R}^{r^2}
\]

In batched tensor notation using Einstein summation:

\[
Q = \text{Reshape}\left(\text{Einsum}("blhi, blhj \to blhij", q_1, q_2)\right) \in \mathbb{R}^{B \times H \times L \times r^2}
\]
\[
K = \text{Reshape}\left(\text{Einsum}("blhi, blhj \to blhij", k_1, k_2)\right) \in \mathbb{R}^{B \times H \times L \times r^2}
\]

### Factorized Kernel Inner-Product:
The unnormalized similarity score between query position $i$ and key position $j$ evaluates to:

\[
\langle Q_{h, i}, K_{h, j} \rangle = \sum_{a=1}^r \sum_{b=1}^r (q_{1, h, i})_a (q_{2, h, i})_b \cdot (k_{1, h, j})_a (k_{2, h, j})_b = \left( q_{1, h, i}^\top k_{1, h, j} \right) \cdot \left( q_{2, h, i}^\top k_{2, h, j} \right)
\]

This proves that the inner product of the $r^2$-dimensional outer products equals the scalar product of two rank-$r$ inner products.

Scaling by factor $\frac{1}{\sqrt{r}}$ and adding mask $M \in \mathbb{R}^{B \times 1 \times L \times L}$:

\[
S_{b, h, i, j} = \frac{1}{\sqrt{r}} \left( \sum_{m=1}^{r^2} Q_{b, h, i, m} \cdot K_{b, h, j, m} \right) + M_{b, 1, i, j}
\]
\[
A_{b, h, i, j} = \text{Softmax}_j (S_{b, h, i, j}), \quad \widehat{A} = \text{Dropout}(A, p_{\text{drop}})
\]
\[
O = \widehat{A} V \in \mathbb{R}^{B \times H \times L \times d_k} \implies \text{Reshaped Context} \in \mathbb{R}^{B \times L \times D}
\]
\[
Y = \text{Context} \cdot W_O \in \mathbb{R}^{B \times L \times D}, \quad W_O \in \mathbb{R}^{D \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input: hidden_states [B, L, D]
  │
  ├──> Linear(D, H * r) [q_proj1] ──> q1: [B, L, H, r]
  ├──> Linear(D, H * r) [q_proj2] ──> q2: [B, L, H, r]
  ├──> Linear(D, H * r) [k_proj1] ──> k1: [B, L, H, r]
  ├──> Linear(D, H * r) [k_proj2] ──> k2: [B, L, H, r]
  └──> Linear(D, D)     [v_proj]  ──> v:  [B, L, H, d_k]
                                │
                                ├──> Einsum("blhi,blhj->blhij", q1, q2) ──> view ──> Q: [B, H, L, r*r]
                                ├──> Einsum("blhi,blhj->blhij", k1, k2) ──> view ──> K: [B, H, L, r*r]
                                ├──> Permute v ──> v: [B, H, L, d_k]
                                │
                                ├──> MatMul: Q @ K.transpose(-2, -1) / sqrt(r) ──> scores: [B, H, L, L]
                                ├──> Add attention_mask ──> [B, H, L, L]
                                ├──> Softmax(dim=-1) & Dropout & Head Mask ──> attn: [B, H, L, L]
                                │
                                └──> MatMul: attn @ v ──> out: [B, H, L, d_k]
                                        │
                                        └──> permute(0, 2, 1, 3).contiguous().view([B, L, D])
                                        └──> Linear(D, D) [out_proj] ──> Output: [B, L, D]
```

---

## 4. Analytical Complexity Profiling

### Parameter & Computational FLOPs Breakdown

Let $B$ be batch size, $L$ sequence length, $D$ model dimension, $H$ heads, $d_k = D / H$, and $r$ factorized rank.

1. **Parameter Footprint**:
   - Standard Query/Key Projections: $2 D^2$.
   - TPA Factorized Query/Key Projections ($W_{Q1}, W_{Q2}, W_{K1}, W_{K2}$): $4 \times D \cdot (H \cdot r) = 4 D H r$.
   - Value & Output Projections ($W_V, W_O$): $2 D^2$.
   - Total Parameters: $2 D^2 + 4 D H r$.
   - For $D=768, H=12, r=8$:
     - Standard Q/K: $2 \times 768^2 = 1,179,648$ parameters.
     - TPA Q/K: $4 \times 768 \times 96 = 294,912$ parameters ($75\%$ reduction in Q/K parameter footprint).

2. **Computational FLOPs & MACs**:
   - Low-Rank Projections ($W_{Q1}, W_{Q2}, W_{K1}, W_{K2}$): $4 \times 2 B L D H r = 8 B L D H r$ FLOPs ($4 B L D H r$ MACs).
   - Outer-Product Einsum: $2 \times 2 B L H r^2 = 4 B L H r^2$ FLOPs.
   - Core Score Calculation ($Q K^\top$): $[B, H, L, r^2] \times [B, H, r^2, L] \implies 2 B H L^2 r^2$ FLOPs ($B H L^2 r^2$ MACs).
   - Context Aggregation ($A V$): $[B, H, L, L] \times [B, H, L, d_k] \implies 2 B H L^2 d_k = 2 B L^2 D$ FLOPs.
   - Value & Output Projections ($W_V, W_O$): $4 B L D^2$ FLOPs.

- **Total MACs**: $4 B L D H r + B H L^2 r^2 + 2 B L D^2 + B L^2 D$
- **Total FLOPs**: $8 B L D H r + 4 B L H r^2 + 2 B H L^2 r^2 + 4 B L D^2 + 2 B L^2 D$

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### Explicit Einsum Memory Expansion Overhead
In naive PyTorch, calling `torch.einsum("blhi,blhj->blhij", q1, q2)` explicitly materializes a 5D intermediate tensor of size $[B, L, H, r, r]$ before flattening to $[B, L, H, r^2]$. For $r=16$, this tensor allocates $256$ floats per token head, introducing intermediate memory bandwidth overhead.

### Score Variance Scaling Misalignment
TPA uses scale factor $\frac{1}{\sqrt{r}}$ instead of the canonical inner-product scale factor $\frac{1}{r} = \frac{1}{\sqrt{r^2}}$. Because $Q$ and $K$ vectors live in $\mathbb{R}^{r^2}$, the expected variance of unnormalized dot-products scale proportionally with $r^2$:

\[
\text{Var}(\langle Q, K \rangle) = \mathcal{O}(r^2)
\]

Scaling by $\frac{1}{\sqrt{r}}$ leaves residual variance of order $\mathcal{O}(r^{\frac{3}{2}})$, causing pre-softmax score saturation and vanishing gradients as rank $r$ increases.

### Rank Degeneracy in Paired Subspace Projections
If $W_{Q1}$ and $W_{Q2}$ collapse into parallel subspaces during training ($\text{cos}(W_{Q1}, W_{Q2}) \approx 1$), the outer product tensor degenerates into a symmetric rank-1 matrix, discarding polynomial expressiveness.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Fused Factorized Inner-Product Triton Kernel
Because $\langle q_{1, h, i} \otimes q_{2, h, i}, k_{1, h, j} \otimes k_{2, h, j} \rangle = (q_{1, h, i}^\top k_{1, h, j}) \cdot (q_{2, h, i}^\top k_{2, h, j})$, hardware kernels can compute scores directly from rank-$r$ vectors, bypassing the $r^2$ outer-product expansion entirely.

```
       Global Memory (HBM)
      ┌────────────────────────────────────────────────────────┐
      │ q1, q2 [B, H, L, r]                                    │
      │ k1, k2 [B, H, L, r]                                    │
      │ V [B, H, L, d_k]                                       │
      └───────────────────────────┬────────────────────────────┘
                                  │ Direct Tile Load (Rank r)
                                  ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                   Triton SRAM Scratchpad                         │
 │ 1. Load Rank-r Tiles: q1_tile, q2_tile, k1_tile, k2_tile         │
 │ 2. Compute Factorized Scores:                                    │
 │    S1_tile = q1_tile @ k1_tile.T  [BLOCK_M, BLOCK_N]            │
 │    S2_tile = q2_tile @ k2_tile.T  [BLOCK_M, BLOCK_N]            │
 │ 3. Hadamard Product Score Fusion:                                │
 │    S_ij = (S1_tile * S2_tile) / r                                │
 │ 4. Execute Online Softmax & Accumulate Output Tile O_tile        │
 └────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Output O [B, H, L, d_k]                                │
      └────────────────────────────────────────────────────────┘
```

### Triton Implementation Guidelines:
1. **Mathematical Hadamard Score Fusion**:
   Avoid materializing $r^2$-dimensional vectors in DRAM. Compute two smaller $[BLOCK\_M, BLOCK\_N]$ matrix products in SRAM registers and multiply them element-wise:
   ```python
   # Inside Triton Kernel
   scores = (tl.dot(q1, k1.T) * tl.dot(q2, k2.T)) * (1.0 / rank)
   ```
   This reduces score computation complexity from $\mathcal{O}(B H L^2 r^2)$ to $\mathcal{O}(B H L^2 r)$, achieving substantial speedups.
2. **Corrected Variance Scaling**:
   Enforce mathematically rigorous variance scaling $\frac{1}{r}$ inside the fused Triton kernel to guarantee numerical stability across arbitrary rank choices ($r \in [4, 32]$).
3. **Register-Level Vectorized Reduction**:
   Utilize 128-bit SIMD registers to execute parallel rank-$r$ dot products, keeping activation memory footprint at $\mathcal{O}(B \cdot H \cdot L \cdot r)$ throughout forward and backward passes.