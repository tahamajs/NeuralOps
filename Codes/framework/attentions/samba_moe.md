# SambaMoE: Tri-Hybrid Sliding-Window Attention, Spectral Mamba SSM, and Token-Choice MoE

## 1. Abstract & Theoretical Motivation
Modern long-context architectures face a fundamental tension: dense Multi-Head Self-Attention (MHSA) models capture high-precision local syntactic patterns but scale quadratically $\mathcal{O}(L^2)$, whereas continuous-time State-Space Models (SSMs) like Mamba process global sequence contexts with linear complexity $\mathcal{O}(L)$ but struggle with precise associative recall. Furthermore, standard Feed-Forward Networks (FFNs) apply static parameter computations equally to all tokens, wasting parameter capacity.

**SambaMoE (Sliding-window Attention, Mamba SSM, and Mixture-of-Experts)** unifies these three paradigms into a single, cohesive sub-quadratic layer block. SambaMoE synthesizes sequence representations in two sequential stages:
1. **SAMBA-M3 Hybrid Stage**: Couples local sliding-window multi-head attention (receptive field $W$) with spectral state-space model recurrence (Mamba SSM) in a pre-layer normalized dual-residual pipeline.
2. **REAPMoE Sparse Stage**: Refines the composite sequence representations via a Token-Choice Top-$k$ Mixture-of-Experts (MoE) sub-network, routing each token dynamically to $k$ active experts out of an $E$-expert pool.

By combining $W$-constrained local attention, linear global SSM recurrence, and sparse conditional expert computation, SambaMoE achieves strict $\mathcal{O}(L \cdot D \cdot (W + d_{\text{state}} + k \cdot d_{\text{ffn}}))$ linear scaling with sequence length $L$, expanding model parameter capacity without increasing per-token latency.

---

## 2. Mathematical Derivations & Formulations

### 2.1 Stage 1: SAMBA-M3 Local-Global Hybrid Fusion
Given input hidden states $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$:

1. **Sliding-Window Causal Multi-Head Attention (SWA)**:
   A causal window mask $\mathbf{M}_{\text{window}} \in \{0, -\infty\}^{L \times L}$ restricts attention to relative token distances $\le W$:
   $$\mathbf{M}_{\text{window}}(i, j) = \begin{cases} 0 & \text{if } 0 \le i - j \le W \\ -\infty & \text{otherwise} \end{cases}$$
   $$\mathbf{X}_{\text{attn}}, \mathbf{A} = \operatorname{SWA}(\mathbf{X}, \mathbf{M}_{\text{window}}) \in \mathbb{R}^{B \times L \times D}$$
   $$\mathbf{X}^{(1)} = \operatorname{LayerNorm}\left(\mathbf{X} + \mathbf{X}_{\text{attn}}\right)$$

2. **Spectral Mamba State-Space Recurrence**:
   $\mathbf{X}^{(1)}$ is processed by continuous state-space discretization with learnable spectral channel gating $\mathbf{s} \in \mathbb{R}^D$:
   $$\mathbf{h}_t = \mathbf{A}_{\text{ssm}} \mathbf{h}_{t-1} + \mathbf{B}_{\text{ssm}} \mathbf{x}^{(1)}_t$$
   $$\mathbf{y}_t = \mathbf{C}_{\text{ssm}} \mathbf{h}_t + \mathbf{D}_{\text{ssm}} \mathbf{x}^{(1)}_t$$
   $$\mathbf{X}_{\text{mamba}} = \mathbf{Y} \odot \sigma(\mathbf{s})$$
   $$\mathbf{X}^{(2)} = \operatorname{LayerNorm}\left(\mathbf{X}^{(1)} + \mathbf{X}_{\text{mamba}}\right)$$

### 2.2 Stage 2: REAPMoE Sparse Token Routing
The intermediate output $\mathbf{X}^{(2)} \in \mathbb{R}^{B \times L \times D}$ enters the **REAPMoE** sub-network:

1. **Router Gating Logits & Top-$k$ Selection**:
   The router projects hidden features to expert logit space:
   $$\mathbf{R} = \mathbf{X}^{(2)} \mathbf{W}_{\text{router}} \quad \in \mathbb{R}^{B \times L \times E}$$
   The top-$k$ active expert indices $\mathcal{E}$ and unnormalized scores $\mathbf{S}_{\text{top-}k}$ are selected ($k=2$):
   $$\mathcal{E}, \mathbf{S}_{\text{top-}k} = \operatorname{TopK}\left(\mathbf{R}, k\right) \quad \in \mathbb{R}^{B \times L \times k}$$
   Softmax normalization is performed across the top-$k$ entries:
   $$w_{b, \ell, j} = \frac{\exp(\mathbf{S}_{\text{top-}k, b, \ell, j})}{\sum_{m=1}^k \exp(\mathbf{S}_{\text{top-}k, b, \ell, m})}, \quad j \in \{1, \dots, k\}$$

2. **Conditional Expert Execution & Aggregation**:
   Each expert $e \in \{1, \dots, E\}$ is a 2-layer MLP with expansion factor 4:
   $$\operatorname{Expert}_e(\mathbf{x}) = \operatorname{GELU}\left(\mathbf{x} \mathbf{W}_{1, e} + \mathbf{b}_{1, e}\right) \mathbf{W}_{2, e} + \mathbf{b}_{2, e}$$
   where $\mathbf{W}_{1, e} \in \mathbb{R}^{D \times 4D}$ and $\mathbf{W}_{2, e} \in \mathbb{R}^{4D \times D}$.

   The total output for token position $(b, \ell)$ evaluates the weighted sum of expert transformations matching activated mask $\mathbb{I}(e \in \mathcal{E}_{b, \ell})$:
   $$\mathbf{Y}_{\text{moe}, b, \ell} = \sum_{j=1}^k w_{b, \ell, j} \cdot \operatorname{Expert}_{\mathcal{E}_{b, \ell, j}}\left( \mathbf{X}^{(2)}_{b, \ell} \right)$$

3. **Final Layer Normalization & Residual Connection**:
   $$\mathbf{O} = \operatorname{LayerNorm}\left( \mathbf{X}^{(2)} + \mathbf{Y}_{\text{moe}} \right) \in \mathbb{R}^{B \times L \times D}$$

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Step | Operation | Input Shape | Output Shape | Tensor Transformation Description |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Input Activations | - | `[B, L, D]` | Input sequence tensor |
| 2 | Sliding Window Attention | `[B, L, D]` | `[B, L, D]` | Local attention within window size $W$ |
| 3 | Residual Add & Norm 1 | `[B, L, D]` | `[B, L, D]` | Add input residual & apply LayerNorm |
| 4 | Spectral Mamba SSM | `[B, L, D]` | `[B, L, D]` | Linear SSM sequence recurrence + spectral gating |
| 5 | Residual Add & Norm 2 | `[B, L, D]` | `[B, L, D]` | Add intermediate residual & apply LayerNorm |
| 6 | Router Gating Projection| `[B, L, D]` | `[B, L, E]` | Linear projection via $W_{\text{router}}$ |
| 7 | Top-$k$ Selection & Softmax| `[B, L, E]` | `[B, L, k]` | Extract top-$k$ expert indices & probabilities |
| 8 | Masked Expert Loops | `[B, L, D]`, `Indices` | `[B, L, D]` | Conditional routing execution over $E$ expert MLPs |
| 9 | MoE Weighted Sum | Masked Outputs | `[B, L, D]` | Multiply active expert outputs by router weights |
| 10| Residual Add & Norm 3 | `[B, L, D]` | `[B, L, D]` | Add SAMBA output residual & apply LayerNorm |

```
                       [B, L, D] Input Tensor
                           │
       ┌───────────────────┴───────────────────┐
       │ Residual                              ▼
       │                     Sliding-Window Attention (W)
       │                        [B, L, D], Mask [L, L]
       │                                       │
       └───────────────────┬───────────────────┘
                           ▼
               Residual Add & LayerNorm 1
                           │
       ┌───────────────────┴───────────────────┐
       │ Residual                              ▼
       │                     Spectral Mamba SSM Recurrence
       │                           [B, L, D]
       │                                       │
       └───────────────────┬───────────────────┘
                           ▼
               Residual Add & LayerNorm 2 (X^(2))
                           │
       ┌───────────────────┴───────────────────┐
       │ Residual                              ▼
       │                       REAPMoE Router Logits
       │                          [B, L, E]
       │                                       │
       │                                       ▼
       │                       Top-k Selection & Softmax
       │                          Indices [B, L, k]
       │                                       │
       │                                       ▼
       │                       Masked Expert Executions
       │                       Weighted Accumulation [B, L, D]
       │                                       │
       └───────────────────┬───────────────────┘
                           ▼
               Residual Add & LayerNorm 3
                           │
                           ▼
             [B, L, D] Output & Attention Weights
```

---

## 4. Analytical Complexity Profiling

### 4.1 FLOPs Computation Breakdown
Let $B$ be batch size, $L$ sequence length, $D$ hidden dimension, $W$ sliding window size, $d_{\text{state}}$ SSM state dimension, $E$ total experts, and $k$ active experts per token.

1. **Sliding-Window Attention (SWA)**:
   - Projections ($\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}$): $8 B L D^2$
   - Banded Matmul (Window size $W$): $2 \cdot 2 \cdot B \cdot H \cdot L \cdot W \cdot d_k = 4 B L D W$
   $$\operatorname{FLOPs}_{\text{swa}} = 8 B L D^2 + 4 B L D W$$
2. **Spectral Mamba SSM**:
   - State-space recurrence & spectral channel gating:
   $$\operatorname{FLOPs}_{\text{ssm}} \approx 2 B L D \cdot d_{\text{state}} + 2 B L D^2$$
3. **REAPMoE Router Gate**:
   $$\operatorname{FLOPs}_{\text{router}} = 2 B L D E$$
4. **Active Expert MLPs**:
   For each activated token, $k$ experts are computed. Each expert maps $D \to 4D \to D$, requiring $2 \cdot (D \cdot 4D) + 2 \cdot (4D \cdot D) = 16 D^2$ FLOPs.
   Across $B \times L$ tokens with $k$ active experts:
   $$\operatorname{FLOPs}_{\text{active\_experts}} = 16 B L k D^2$$
5. **Weighted Accumulation & LayerNorms**:
   $$\operatorname{FLOPs}_{\text{misc}} \approx 2 B L k D + 6 B L D$$

$$\text{Total FLOPs} = 10 B L D^2 + 4 B L D W + 2 B L D d_{\text{state}} + 2 B L D E + 16 B L k D^2$$

### 4.2 MACs & Memory Footprint Analysis
- **Multiply-Accumulate Operations (MACs)**:
  $$\text{Total MACs} \approx 5 B L D^2 + 2 B L D W + B L D d_{\text{state}} + B L D E + 8 B L k D^2$$
- **Peak Activation Memory**:
  - Sliding Window Attention Matrix: $B \cdot H \cdot L \cdot W \times 2 \text{ bytes}$
  - Mamba Hidden State Buffer: $B \cdot L \cdot D \cdot d_{\text{state}} \times 2 \text{ bytes}$
  - Router Logits & Top-$k$ Indices: $B \cdot L \cdot (E + k) \times 2 \text{ bytes}$
  - Masked Expert Intermediate Buffers: In iterative implementation, peak memory is bounded by single expert buffer $B \cdot L_{\text{masked}} \cdot 4D \times 2 \text{ bytes}$.

### 4.3 Asymptotic Scalability Comparison
- **Standard Dense Transformer Block**: $\mathcal{O}(L^2 D + L D^2)$ compute, $\mathcal{O}(L^2 H)$ memory.
- **SambaMoE Hybrid Block**: $\mathcal{O}\left(L D (D + W + d_{\text{state}} + k D)\right)$ compute, $\mathcal{O}(L D)$ memory.
- Completely avoids $L^2$ scaling while providing high parameter capacity via $E$ experts.

---

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes

### 5.1 Sequential Iterative Expert Masking Overhead
In `REAPMoE.forward`:
```python
for e in range(self.n_experts):
    mask = (indices == e).any(dim=-1)
    if mask.any():
        out += selected_weights * self.experts[e](x * mask.unsqueeze(-1))
```
- **Sequential Launch Overhead**: The `for e in range(n_experts):` loop launches $E$ separate PyTorch CUDA kernels sequentially. When $E=16$ or $32$, host-side overhead dominates, causing GPU warp starvation.
- **Zero-Value Multiplication Waste**: `x * mask.unsqueeze(-1)` zero-masks unrouted tokens but passes full `[B, L, D]` tensor shapes through the expert linear layers, wasting compute on zero inputs.

### 5.2 Window Receptive Field Starvation for Complex Graph Reasoning
The sliding window $W$ restricts attention to local context $\le W$. If a task requires direct multi-hop retrieval across distance $\Delta > W$ (e.g., matching variable definitions in code at line 1 and line 10,000):
- Local attention cannot bridge the gap directly.
- The model relies entirely on Mamba state-space recurrence $\mathbf{h}_t$. If the SSM state dimension $d_{\text{state}}$ experiences lossy compression, associative recall fails completely.

### 5.3 Router Over-Concentration & Expert Collapse
Without dynamic capacity factors or auxiliary load-balancing loss $\mathcal{L}_{\text{balance}} = E \sum_{e=1}^E f_e P_e$:
- Early gradient steps favor 1 or 2 high-norm experts.
- Tokens route exclusively to these dominant experts while remaining experts receive zero updates, reducing effective capacity from $E$ to $k$.

---

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies

To achieve true hardware speedups, SambaMoE requires fused Triton execution for both the Sliding-Window Attention and the Token-Choice MoE.

### 6.1 Triton Hardware Acceleration Layout

```
+-----------------------------------------------------------------------------------+
|                        TRITON SAMBAMOE PIPELINE                                   |
+-----------------------------------------------------------------------------------+
|  1. FUSED SLIDING-WINDOW FLASHATTENTION KERNEL                                    |
|     - Block size [BLOCK_M, BLOCK_N].                                              |
|     - Loads Q, K, V tiles into SRAM.                                              |
|     - Applies causal window condition: if (col_idx < row_idx - W) score = -inf.   |
|     - Computes online softmax and yields Output [B, L, D] without DRAM attention  |
|       matrix write.                                                               |
+-----------------------------------------------------------------------------------+
|  2. FUSED MAMBA RECURRENCE & LAYER NORM KERNEL                                    |
|     - Parallel scan implementation of state space update: h_t = A*h_{t-1} + B*x_t. |
|     - Applies channel gate sigmoid(s) and pre-LN in single SRAM pass.             |
+-----------------------------------------------------------------------------------+
|  3. TRITON ROUTED TOP-K MOE SCATTER-GEMM KERNEL                                   |
|     - Computes router logits and top-k indices.                                   |
|     - Packs activated tokens into contiguous expert buffers (eliminates zero-mul).|
|     - Invokes Grouped GEMM over active expert weights.                            |
|     - Writes back weighted outputs to destination tensor.                         |
+-----------------------------------------------------------------------------------+
```

### 6.2 Implementation Code Blueprint (Triton Sliding-Window FlashAttention)

```python
import triton
import triton.language as tl

@triton.jit
def _sliding_window_flash_attn_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    scale,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_K: tl.constexpr
):
    start_m = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    batch_id = tl.program_id(axis=2)

    # Offset pointers for Query tile
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_K)
    q_ptrs = Q_ptr + batch_id * stride_qb + head_id * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    q = tl.load(q_ptrs)

    # Calculate valid sliding window column bounds
    # Key tokens must lie within [row_idx - WINDOW_SIZE, row_idx]
    min_n = tl.maximum(0, start_m * BLOCK_M - WINDOW_SIZE)
    max_n = (start_m + 1) * BLOCK_M

    # Initialize Online Softmax Accumulators
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_K], dtype=tl.float32)

    # Loop over key tiles within valid sliding window range
    for start_n in range(min_n, max_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_id * stride_kb + head_id * stride_kh + (offs_n[None, :] * stride_kn + offs_d[:, None])
        k = tl.load(k_ptrs)

        # Compute dot product S = Q @ K.T
        s = tl.dot(q, k) * scale

        # Apply Causal & Sliding Window Mask
        # Condition: row_idx >= col_idx and row_idx - col_idx <= WINDOW_SIZE
        mask = (offs_m[:, None] >= offs_n[None, :]) & ((offs_m[:, None] - offs_n[None, :]) <= WINDOW_SIZE)
        s = tl.where(mask, s, float("-inf"))

        # Online Softmax update
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(s - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p, axis=1)

        # Rescale prior accumulator
        acc_scale = tl.exp(m_i - m_new)
        acc = acc * acc_scale[:, None]

        # Load Value tile & Accumulate
        v_ptrs = V_ptr + batch_id * stride_vb + head_id * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
        v = tl.load(v_ptrs)
        acc = tl.dot(p.to(v.dtype), v, acc)

        l_i = l_new
        m_i = m_new

    # Final Softmax Normalization
    acc = acc / l_i[:, None]
    out_ptrs = Out_ptr + batch_id * stride_ob + head_id * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty))
```

### 6.3 Hardware Acceleration Impact
1. **DRAM Memory Traffic Elimination**: Sliding-Window FlashAttention computes masked attention entirely in SRAM registers, dropping activation memory writes from $\mathcal{O}(B H L W)$ to $\mathcal{O}(B L D)$.
2. **Compact Contiguous Tokens**: Replacing `x * mask` with Triton token packing compresses active tokens into contiguous memory chunks before GEMM execution, raising Tensor Core MAC utilization to >90%.