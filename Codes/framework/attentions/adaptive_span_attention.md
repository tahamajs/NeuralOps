# Adaptive Span Attention: Dynamic Per-Head Context Window Truncation

## 1. Abstract & Theoretical Motivation

Standard Multi-Head Attention (MHA) calculates full dense quadratic pairwise interactions $\mathcal{O}(L^2)$ across the entire sequence length $L$ for all attention heads. Empirical research (Sukhbaatar et al., 2019) reveals that different attention heads specialize in distinct temporal scales: while a subset of heads requires global contextual access, many heads primarily attend to local, short-range syntactic clusters. 

Adaptive Span Attention dynamically caps the receptive field of each individual head $h$ to a learnable span boundary $S_h \in (0, S_{\max}]$. By parameterizing span boundaries using differentiable soft sigmoid gating, the module enforces head-specific context truncation. This dynamic masking reduces memory bandwidth overhead during long-sequence processing, facilitates structural head specialization, and eliminates unneeded long-range attention computation.

---

## 2. Mathematical Derivations & Formulations

Let $X \in \mathbb{R}^{B \times L \times D}$ be the input activation sequence, where $B$ is batch size, $L$ is sequence length, and $D$ is model dimension. Let $H$ be the number of attention heads, $d_k = D / H$ the head dimension, and $S_{\max}$ the maximum allowable attention span.

### Multi-Head Subspace Projections:
Query, Key, and Value representations are projected via linear transformations:

\[
Q = X W_Q \in \mathbb{R}^{B \times L \times D}, \quad K = X W_K \in \mathbb{R}^{B \times L \times D}, \quad V = X W_V \in \mathbb{R}^{B \times L \times D}
\]

where $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$. Reshaping into multi-head layouts yields $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_k}$.

The unnormalized, scaled dot-product attention score tensor $S \in \mathbb{R}^{B \times H \times L \times L}$ per head $h$ is:

\[
S_{b, h, i, j} = \frac{1}{\sqrt{d_k}} \sum_{m=1}^{d_k} Q_{b, h, i, m} \cdot K_{b, h, j, m}
\]

### Differentiable Receptive Field Span Parameterization:
Each attention head $h \in \{1, \dots, H\}$ owns a trainable scalar parameter logit $\theta_h \in \mathbb{R}$. The active sequence span $S_h$ is computed via:

\[
S_h = \sigma(\theta_h) \cdot S_{\max} = \frac{S_{\max}}{1 + \exp(-\theta_h)} \in (0, S_{\max}]
\]

### Relative Positional Distance & Span Masking:
For sequence query index $i$ and key index $j$, the absolute relative distance is:

\[
\Delta_{i, j} = |i - j|
\]

The span mask tensor $M^{\text{span}} \in \{0, 1\}^{H \times L \times L}$ evaluates whether key position $j$ lies within the active receptive window of query position $i$ for head $h$:

\[
M^{\text{span}}_{h, i, j} = \begin{cases} 
1 & \text{if } \Delta_{i, j} \le S_h \\ 
0 & \text{if } \Delta_{i, j} > S_h 
\end{cases}
\]

Applying the span mask to raw attention scores:

\[
\widetilde{S}_{b, h, i, j} = \begin{cases} 
S_{b, h, i, j} & \text{if } M^{\text{span}}_{h, i, j} = 1 \\ 
-\infty & \text{if } M^{\text{span}}_{h, i, j} = 0 
\end{cases}
\]

### Softmax Normalization & Context Aggregation:
Incorporating optional padding mask $M^{\text{pad}} \in \mathbb{R}^{B \times 1 \times 1 \times L}$:

\[
\widehat{S}_{b, h, i, j} = \widetilde{S}_{b, h, i, j} + M^{\text{pad}}_{b, 1, 1, j}
\]
\[
A_{b, h, i, j} = \text{Softmax}_j (\widehat{S}_{b, h, i, j}) = \frac{\exp(\widehat{S}_{b, h, i, j})}{\sum_{l=1}^L \exp(\widehat{S}_{b, h, i, l})}
\]

Applying dropout probability $p_{\text{drop}}$:

\[
\overline{A} = \text{Dropout}(A, p_{\text{drop}})
\]
\[
O_{b, h, i, m} = \sum_{j=1}^L \overline{A}_{b, h, i, j} \cdot V_{b, h, j, m} \implies O \in \mathbb{R}^{B \times H \times L \times d_k}
\]
\[
Y = \text{Concat}(O_1, \dots, O_H) \cdot W_O \in \mathbb{R}^{B \times L \times D}, \quad W_O \in \mathbb{R}^{D \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

```
Input: hidden_states [B, L, D]
  │
  ├──> Linear(D, D) [W_q] ──> Q: [B, L, D] ──> view & transpose ──> Q: [B, H, L, d_k]
  ├──> Linear(D, D) [W_k] ──> K: [B, L, D] ──> view & transpose ──> K: [B, H, L, d_k]
  └──> Linear(D, D) [W_v] ──> V: [B, L, D] ──> view & transpose ──> V: [B, H, L, d_k]
                                │
                                ├──> MatMul: Q @ K.transpose(-2, -1) / sqrt(d_k) ──> scores: [B, H, L, L]
                                │
                                ├──> Compute Span: S_h = Sigmoid(span_logits) * max_span ──> span: [H]
                                ├──> Compute Distance Matrix: dist = |i - j| ──> [L, L]
                                ├──> Compare: (dist <= span) ──> mask_span: [H, L, L]
                                ├──> scores.masked_fill(mask_span == 0, -1e9) ──> [B, H, L, L]
                                │
                                ├──> Add attention_mask (padding) ──> [B, H, L, L]
                                ├──> Softmax(dim=-1) & Dropout ──> attn_weights: [B, H, L, L]
                                │
                                └──> MatMul: attn_weights @ V ──> context: [B, H, L, d_k]
                                        │
                                        └──> transpose(1, 2).contiguous().view([B, L, D])
                                        └──> Linear(D, D) [W_o] ──> Output: [B, L, D]
```

---

## 4. Analytical Complexity Profiling

### FLOPs, MACs & Memory Footprint

Let $B$ be batch size, $L$ sequence length, $D$ model dimension, $H$ heads, $d_k = D / H$, and $S_h$ head span.

1. **Linear Projection FLOPs**:
   - $W_Q, W_K, W_V, W_O$: $4 \times 2 B L D^2 = 8 B L D^2$ FLOPs ($4 B L D^2$ MACs).

2. **Attention Score Calculation ($Q K^\top$)**:
   - Dense Score Matrix FLOPs: $2 B H L^2 d_k = 2 B L^2 D$ FLOPs ($B L^2 D$ MACs).

3. **Span Mask Evaluation Overhead**:
   - Distance matrix broadcast comparison: $H \cdot L^2$ comparisons $+ H$ Sigmoid evaluations.
   - Operations: $\mathcal{O}(H L^2)$ (Negligible compared to GEMM).

4. **Context Matrix Product ($A V$)**:
   - Ideal Sparse Computation FLOPs (when enforced via sparse layout):
     \[
     \text{FLOPs}_{\text{sparse}} = 2 B \sum_{h=1}^H L \cdot \min(L, S_h) \cdot d_k = 2 B L d_k \sum_{h=1}^H \min(L, S_h)
     \]
   - Standard Unfused Dense Matrix FLOPs: $2 B L^2 D$.

### Memory Footprint:
- **Parameters**: $4 D^2 + H$ floats (Adding $H$ scalar span logits).
- **Activation Storage**: $\mathcal{O}(B \cdot H \cdot L^2)$ for dense score matrices in standard PyTorch implementations.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### Hard Threshold Gradient Mask Discontinuity
In naive step-thresholding implementations (`dist <= span`), the indicator function $I(\Delta_{ij} \le S_h)$ has a derivative of zero everywhere except at the boundary $\Delta_{ij} = S_h$. Consequently, standard backpropagation cannot update span logits $\theta_h$, trapping heads at their initial $S_{\max}$ values.

```
Sub-Differentiable Boundary Problem:
∂ Mask / ∂ θ_h  =  0  everywhere (Hard Mask Discontinuity)
```

### Materialized Dense Matrix Memory Bottleneck
Although Adaptive Span reduces the theoretical computational complexity to $\mathcal{O}(B \cdot L \cdot S_{\text{avg}} \cdot D)$, standard PyTorch instantiates full $[B, H, L, L]$ dense matrices before applying `masked_fill`, failing to realize DRAM memory footprint reductions during execution.

### Short-Span Entropy Saturation
If $\theta_h$ decays excessively ($\sigma(\theta_h) \to 0, S_h < 1$), head $h$ attends exclusively to the self-token ($i = j$). This turns the attention head into an identity map, causing representation capacity collapse in that subspace.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

### Fused Band-Diagonal Triton Attention Kernel Architecture
To translate theoretical span savings into hardware execution speedups, Triton fused kernels must restrict block reads to active diagonal bands defined by head span $S_h$.

```
       Global Memory (HBM)
      ┌────────────────────────────────────────────────────────┐
      │ Q [B, H, L, d_k], K [B, H, L, d_k], V [B, H, L, d_k]   │
      │ Head Spans S_h [H]                                     │
      └───────────────────────────┬────────────────────────────┘
                                  │ Load Block Tiles Within Span Boundary
                                  ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                   Triton SRAM Scratchpad                         │
 │ 1. Evaluate Tile Distance: |tile_m - tile_n|                     │
 │ 2. IF |tile_m - tile_n| * BLOCK_SIZE > S_h:                       │
 │       SKIP LOAD & COMPUTATION (Early Block Pruning)              │
 │ 3. ELSE:                                                         │
 │       Load K_tile, V_tile                                        │
 │       Compute Scores S_tile = Q_tile @ K_tile.T / sqrt(d_k)      │
 │       Execute Online Softmax & Accumulate Output Tile            │
 └────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Context Output O [B, H, L, d_k]                        │
      └────────────────────────────────────────────────────────┘
```

### Key Triton Implementation Guidelines:
1. **Piecewise Differentiable Soft-Ramping**:
   Replace hard binary masks with a continuous soft-ramp function during training to ensure non-zero gradient flow:
   \[
   M^{\text{soft}}_{h, i, j} = \text{Clamp}\left(1 - \frac{\Delta_{i, j} - S_h}{R}, 0, 1\right)
   \]
   where $R$ is a fixed ramp length parameter (e.g., $R=16$).
2. **Early Block Pruning in SRAM**:
   In the Triton inner loop over key blocks `tile_n`, skip key block processing entirely if the block range $[n \cdot \text{BLOCK\_N}, (n+1) \cdot \text{BLOCK\_N}]$ falls completely outside the window $[m \cdot \text{BLOCK\_M} - S_h, m \cdot \text{BLOCK\_M} + S_h]$.
3. **Dynamic Banded KV Cache Decoding**:
   During autoregressive decoding, store key-value tokens in circular ring-buffers of capacity $S_h$, capping decoding KV cache memory to $\sum_{h=1}^H S_h \cdot d_k$ bytes instead of $H \cdot L_{\text{past}} \cdot d_k$.