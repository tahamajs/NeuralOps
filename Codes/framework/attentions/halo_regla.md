# HALO-ReGLA: Hardware-Aware Linear Optics & Rectified Gated Linear Attention

## 1. Abstract & Theoretical Motivation

Standard Multi-Head Scaled Dot-Product Attention scales quadratically $\mathcal{O}(L^2)$ with respect to sequence length $L$ due to explicit materialization of softmax attention matrices. HALO-ReGLA (Hardware-Aware Linear Optics with Rectified Gated Linear Attention) resolves this quadratic barrier by combining Rotary Position Embedding (RoPE) with non-negative kernel feature maps $\phi(x) = \text{ReLU}(x)$.

By applying associative matrix algebra to non-negative query and key states, HALO-ReGLA reformulates attention scoring as a continuous matrix state accumulator $KV = \sum_{t=1}^L \widetilde{K}_t^\top V_t \in \mathbb{R}^{d_k \times d_k}$. This reduces time and memory activation complexity from quadratic $\mathcal{O}(L^2)$ to strictly linear $\mathcal{O}(L \cdot d_k^2)$ in sequence length $L$, enabling long-context processing while preserving relative positional awareness through RoPE.

---

## 2. Mathematical Derivations & Formulations

Let $X \in \mathbb{R}^{B \times L \times D}$ denote the input activation sequence, where $B$ is batch size, $L$ is sequence length, and $D$ is model hidden dimension. Let $H$ be the number of heads and $d_k = D / H$ the head dimension.

### Linear Subspace Projections & RoPE Encoding:
Queries, Keys, and Values are projected via linear transformations:

\[
Q_{\text{raw}} = X W_Q \in \mathbb{R}^{B \times L \times D}, \quad K_{\text{raw}} = X W_K \in \mathbb{R}^{B \times L \times D}, \quad V = X W_V \in \mathbb{R}^{B \times L \times D}
\]

where $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$. Reshaping yields $Q_{\text{raw}}, K_{\text{raw}}, V \in \mathbb{R}^{B \times H \times L \times d_k}$.

Rotary Position Embedding (RoPE) is applied to raw queries and keys:

\[
Q = \text{RoPE}(Q_{\text{raw}}) = \left( Q_{\text{raw}} \odot \mathbf{C} \right) + \left( \mathcal{R}(Q_{\text{raw}}) \odot \mathbf{S} \right)
\]
\[
K = \text{RoPE}(K_{\text{raw}}) = \left( K_{\text{raw}} \odot \mathbf{C} \right) + \left( \mathcal{R}(K_{\text{raw}}) \odot \mathbf{S} \right)
\]

where $\mathbf{C} = \cos(m \boldsymbol{\theta})$, $\mathbf{S} = \sin(m \boldsymbol{\theta})$, and $\mathcal{R}(x) = [-x_{D/2:}, x_{:D/2}]$.

### Rectified Non-Negative Kernel Feature Map:
Non-negative feature maps $\phi(x) = \text{ReLU}(x) = \max(0, x)$ project vectors into the positive orthant:

\[
\widetilde{Q} = \text{ReLU}(Q) \in \mathbb{R}^{B \times H \times L \times d_k}
\]
\[
\widetilde{K} = \text{ReLU}(K) \in \mathbb{R}^{B \times H \times L \times d_k}
\]

### Associative Matrix-Vector State Accumulation (ReGLA):
Using non-negative inner products $\langle \widetilde{Q}_i, \widetilde{K}_j \rangle = \sum_{m=1}^{d_k} \widetilde{Q}_{i, m} \widetilde{K}_{j, m} \ge 0$, the unnormalized contextual numerator is factorized via matrix associativity:

\[
\text{Num}_{b, h, i, e} = \sum_{j=1}^L \widetilde{Q}_{b, h, i, d} \left( \widetilde{K}_{b, h, j, d} \cdot V_{b, h, j, e} \right) = \sum_{d=1}^{d_k} \widetilde{Q}_{b, h, i, d} \cdot (KV)_{b, h, d, e}
\]

where the global key-value associative matrix buffer $(KV) \in \mathbb{R}^{B \times H \times d_k \times d_k}$ is accumulated as:

\[
(KV)_{b, h, d, e} = \sum_{j=1}^L \widetilde{K}_{b, h, j, d} \cdot V_{b, h, j, e}
\]

The normalization denominator $Z \in \mathbb{R}^{B \times H \times L \times 1}$ is derived from the key sum vector:

\[
(Z_{\text{sum}})_{b, h, d} = \sum_{j=1}^L \widetilde{K}_{b, h, j, d} \in \mathbb{R}^{B \times H \times d_k}
\]
\[
\text{Den}_{b, h, i} = \left( \sum_{d=1}^{d_k} \widetilde{Q}_{b, h, i, d} \cdot (Z_{\text{sum}})_{b, h, d} \right) + \epsilon
\]

where $\epsilon = 10^{-8}$ prevents division by zero.

### Output Normalization & Projection:
\[
O_{b, h, i, e} = \frac{\text{Num}_{b, h, i, e}}{\text{Den}_{b, h, i}} \in \mathbb{R}^{B \times H \times L \times d_k}
\]
\[
Y = \text{Concat}(O_1, \dots, O_H) \cdot W_O + b_O \in \mathbb{R}^{B \times L \times D}, \quad W_O \in \mathbb{R}^{D \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input: hidden_states [B, L, D]
  │
  ├──> Linear(D, D) [q_proj] ──> Q: [B, L, D] ──> view & transpose ──> Q: [B, H, L, d_k]
  ├──> Linear(D, D) [k_proj] ──> K: [B, L, D] ──> view & transpose ──> K: [B, H, L, d_k]
  └──> Linear(D, D) [v_proj] ──> V: [B, L, D] ──> view & transpose ──> V: [B, H, L, d_k]
                                │
                                ├──> Apply RoPE: Q = rope(Q), K = rope(K) ──> [B, H, L, d_k]
                                ├──> Apply ReLU: Q = ReLU(Q), K = ReLU(K) ──> [B, H, L, d_k]
                                │
                                ├──> Einsum("b h n d, b h n e -> b h d e", K, V) ──> KV: [B, H, d_k, d_k]
                                ├──> Sum K over sequence: z = K.sum(dim=2) ──> Z_sum: [B, H, d_k]
                                │
                                ├──> Einsum("b h n d, b h d e -> b h n e", Q, KV) ──> Num: [B, H, L, d_k]
                                ├──> Einsum("b h n d, b h d -> b h n", Q, Z_sum) + 1e-8 ──> Den: [B, H, L, 1]
                                ├──> Output Context = Num / Den ──> [B, H, L, d_k]
                                │
                                └──> transpose(1, 2).contiguous().view([B, L, D])
                                └──> Linear(D, D) [out_proj] ──> Output: [B, L, D]
```

---

## 4. Analytical Complexity Profiling

### Operational & Computational Workload (FLOPs & MACs)

Let $B$ be batch size, $L$ sequence length, $D$ model dimension, $H$ heads, and $d_k = D / H$.

1. **Linear Projections**:
   - $W_Q, W_K, W_V, W_O$: $4 \times 2 B L D^2 = 8 B L D^2$ FLOPs ($4 B L D^2$ MACs).

2. **Rotary Position Embedding (RoPE)**:
   - $Q, K$ rotations: $2 \times 3 B H L d_k = 6 B L D$ FLOPs.

3. **ReGLA Linear Associative Computations**:
   - Key-Value Matrix Outer Product ($KV = K^\top V$): $[B, H, d_k, L] \times [B, H, L, d_k] \implies 2 B H L d_k^2 = 2 B L D d_k$ FLOPs ($B L D d_k$ MACs).
   - Key Sum Vector ($Z_{\text{sum}}$): $B H L d_k = B L D$ FLOPs.
   - Numerator Contracting Product ($\widetilde{Q} \cdot KV$): $[B, H, L, d_k] \times [B, H, d_k, d_k] \implies 2 B H L d_k^2 = 2 B L D d_k$ FLOPs ($B L D d_k$ MACs).
   - Denominator Normalization ($\widetilde{Q} \cdot Z_{\text{sum}}$): $2 B H L d_k = 2 B L D$ FLOPs.
   - Elementwise Division: $B L D$ FLOPs.

### Complexity Summary:
- **Total MACs**: $4 B L D^2 + 2 B L D d_k$
- **Total FLOPs**: $8 B L D^2 + 4 B L D d_k + 10 B L D$
- **Theoretical Speedup**: Completely eliminates sequence quadratic term $\mathcal{O}(L^2 D)$. Time complexity scales strictly linearly $\mathcal{O}(L)$ with sequence length.
- **Memory Buffer Footprint**: Stores state matrix $(KV) \in \mathbb{R}^{B \times H \times d_k \times d_k}$ ($B H d_k^2$ floats), avoiding $B H L^2$ attention maps.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### Non-Negative Sparsity & Gradient Dead-Zones
Because $\widetilde{Q} = \text{ReLU}(Q)$ and $\widetilde{K} = \text{ReLU}(K)$, any feature channel where $Q_{b, h, i, d} \le 0$ or $K_{b, h, j, d} \le 0$ has its gradient $\frac{\partial \text{ReLU}}{\partial x} = 0$ set to zero. If initialization pushes activations into negative regions, large portions of query and key channels enter permanent gradient dead-zones.

### Recurrent State Saturation under Causal Decoding
In long autoregressive decoding tasks ($L > 100,000$), un-decayed accumulation $(KV)_t = (KV)_{t-1} + \widetilde{K}_t^\top V_t$ causes state matrix norm $\|KV\|_F$ to grow monotonically. Numerical precision degradation occurs in floating-point representations (FP16 / BF16).

### Head State Capacity Bound
The associative memory buffer $(KV) \in \mathbb{R}^{d_k \times d_k}$ has a maximum rank of $d_k$. For sequence lengths $L \gg d_k$ (e.g., $L=16384, d_k=64$), the state matrix becomes heavily saturated, causing loss of fine-grained token position retrieval capacity.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Fused Linear State Accumulation Triton Kernel
To maximize hardware utilization, Triton fused kernels accumulate the matrix state $(KV) \in \mathbb{R}^{d_k \times d_k}$ in GPU shared memory without materializing sequence-length intermediate tensors in DRAM.

```
       Global Memory (HBM)
      ┌────────────────────────────────────────────────────────┐
      │ Q [B, H, L, d_k], K [B, H, L, d_k], V [B, H, L, d_k]   │
      └───────────────────────────┬────────────────────────────┘
                                  │ Direct Stream Block Loads
                                  ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                   Triton SRAM Scratchpad                         │
 │ 1. Apply RoPE & ReLU in SIMD Registers: Q_relu, K_relu           │
 │ 2. Accumulate Local Block KV Matrix:                             │
 │    KV_tile += K_relu_block.T @ V_block   [d_k, d_k]              │
 │ 3. Accumulate Local Key Sum: Z_tile += K_relu_block.sum(dim=0)   │
 │ 4. Compute Output Tile: Num = Q_relu @ KV_tile                   │
 │ 5. Compute Denominator: Den = Q_relu @ Z_tile.T + 1e-8           │
 │ 6. Output_tile = Num / Den                                       │
 └────────────────────────────────┬─────────────────────────────────┘
                                  │ Stream Context Output
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Context Output O [B, L, D]                             │
      └────────────────────────────────────────────────────────┘
```

### Key Triton Implementation Guidelines:
1. **Fused In-Register RoPE + ReLU Feature Mapping**:
   Execute RoPE rotation and ReLU clipping directly inside SIMD registers prior to matrix multiplication, avoiding DRAM read/write cycles.
2. **SRAM State Accumulation**:
   Maintain matrix state $(KV) \in \mathbb{R}^{d_k \times d_k}$ in SRAM L1 cache registers across tile loops over sequence blocks `BLOCK_N`, writing only final context vectors $O$ back to global HBM.
3. **Smooth Activation Replacement**:
   To prevent ReLU gradient dead-zones, replace standard ReLU with a smooth positive activation:
   \[
   \phi(x) = \text{ELU}(x) + 1.0 \quad \text{or} \quad \phi(x) = \text{Softplus}(x)
   \]
   guaranteeing continuous positive gradients across all feature channels.