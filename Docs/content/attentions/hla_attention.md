# HLA Attention: Hadamard Linear Attention via Outer-Product Associative Recurrence

## 1. Executive Abstract & Architectural Motivation

Standard Multi-Head Attention scales quadratically $\mathcal{O}(L^2)$ in sequence length $L$, creating a bottleneck for long-context execution. Hadamard Linear Attention (HLA Attention) eliminates the explicit $L \times L$ attention matrix by maintaining a running associative 2D state matrix $H_t \in \mathbb{R}^{d_k \times d_k}$ per head across time steps.

At each token position $t$, HLA Attention computes the outer product of Key vectors $K_t$ and Value vectors $V_t$, accumulating them into the state matrix $H_t = H_{t-1} + K_t \otimes V_t$. Query vectors $Q_t$ interact directly with $H_t$ via matrix-vector multiplication. By using matrix associativity, HLA reduces time complexity to $\mathcal{O}(L \cdot d_k^2)$ and retains a constant $\mathcal{O}(d_k^2)$ memory footprint during causal autoregressive decoding.

---

## 2. Mathematical Formulations & Derivations

Given input sequence representations $X \in \mathbb{R}^{B \times L \times D}$, queries, keys, and values are projected as:

\[
Q = X W_Q \in \mathbb{R}^{B \times L \times D}, \quad K = X W_K \in \mathbb{R}^{B \times L \times D}, \quad V = X W_V \in \mathbb{R}^{B \times L \times D}
\]

Multi-head formatting yields $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_k}$ with $d_k = D / H$.

### Outer-Product State Recurrence:
The hidden associative state matrix $H_t \in \mathbb{R}^{B \times H \times d_k \times d_k}$ is initialized to zero ($H_0 = \mathbf{0}$) and updated recursively:

\[
H_t = H_{t-1} + K_t \otimes V_t = H_{t-1} + K_t^\top V_t \in \mathbb{R}^{B \times H \times d_k \times d_k}
\]

### Query Context Extraction:
The contextualized output vector $O_t \in \mathbb{R}^{B \times H \times d_k}$ for query $Q_t$ is evaluated as:

\[
O_t = Q_t H_t = \sum_{j=1}^t \left( Q_t K_j^\top \right) V_j \in \mathbb{R}^{B \times H \times d_k}
\]

Stacking outputs across all sequence steps $t = 1, \dots, L$ produces $O \in \mathbb{R}^{B \times H \times L \times d_k}$.

### Final Output Projection:
Multi-head outputs are concatenated and projected to model dimension $D$:

\[
Y = \text{Concat}(O_1, \dots, O_H) W_O \in \mathbb{R}^{B \times L \times D}
\]

---

## 3. Step-by-Step Tensor Execution Trace

```
Input: hidden_states [B, L, D]
  │
  ├──> Q = Linear(D, D)(hidden_states) ──> [B, H, L, d_k]
  ├──> K = Linear(D, D)(hidden_states) ──> [B, H, L, d_k]
  └──> V = Linear(D, D)(hidden_states) ──> [B, H, L, d_k]
         │
         ├──> Recurrent Loop over sequence steps t = 1 ... L:
         │      ├──> Outer Product: K_t.transpose(-2, -1) @ V_t ──> [B, H, d_k, d_k]
         │      ├──> State Recurrence: H_t = H_{t-1} + OuterProduct ──> [B, H, d_k, d_k]
         │      └──> Query Interaction: O_t = Q_t @ H_t ──> [B, H, d_k]
         │
         ├──> Stack outputs across L ──> Context: [B, H, L, d_k]
         ├──> transpose(1, 2) ──> contiguous() ──> [B, L, D]
         └──> Linear(D, D)(Context) ──> Output: [B, L, D]
```

### Detailed Tensor Transformations:

| Step | Operation | Tensor Shape | Description |
| :--- | :--- | :--- | :--- |
| 1 | Input | $[B, L, D]$ | Input hidden state sequence |
| 2 | Linear Projections | $[B, H, L, d_k]$ | $Q, K, V$ states |
| 3 | Outer Product ($K_t \otimes V_t$) | $[B, H, d_k, d_k]$ | Rank-1 outer-product memory update per step |
| 4 | State Matrix Accumulation | $[B, H, d_k, d_k]$ | Cumulative memory state $H_t = H_{t-1} + K_t^\top V_t$ |
| 5 | Query State Projection | $[B, H, d_k]$ | Per-step context extraction $O_t = Q_t H_t$ |
| 6 | Sequence Stacking | $[B, H, L, d_k]$ | Concatenated multi-head context tensor |
| 7 | Out Projection | $[B, L, D]$ | Projection $W_O$ to model dimension $D$ |

---

## 4. Analytical Complexity & Hardware Profiling

### FLOP and MAC Breakdown ($B$: Batch, $L$: Length, $D$: Dim, $H$: Heads, $d_k$: Head Dim):
- **Input Projections ($W_Q, W_K, W_V$)**: $6 B L D^2$ FLOPs
- **Outer Product Memory Updates ($K_t^\top V_t$)**: $L \times (2 B H d_k^2) = 2 B L H d_k^2 = 2 B L D d_k$ FLOPs
- **State Accumulation ($H_{t-1} + \text{Outer}$)**: $L \times (B H d_k^2) = B L D d_k$ FLOPs
- **Query Matrix Multiplication ($Q_t H_t$)**: $L \times (2 B H d_k^2) = 2 B L D d_k$ FLOPs
- **Output Projection ($W_O$)**: $2 B L D^2$ FLOPs

**Total Complexity**:
- **Total MACs**: $4 B L D^2 + 2 B L D d_k$
- **Total FLOPs**: $8 B L D^2 + 5 B L D d_k$ ($\mathcal{O}(L)$ strictly linear sequence scaling)

### Memory Footprint:
- **Static Weights**: $4 D^2$ parameters.
- **Transient Activation Memory**: Bounded by $\mathcal{O}(B \cdot H \cdot d_k^2)$ per step during decoding, completely bypassing quadratic $L^2$ memory storage.

---

## 5. Failure Modes, Rank Collapse & Edge Cases

1. **Unbounded Memory Magnitude Explosion**: Because $H_t = H_{t-1} + K_t^\top V_t$ lacks decay or normalization factors, $H_t$ norms grow monotonically over long sequence horizons ($L > 65536$), causing floating-point overflow.
2. **Lack of Recency Bias**: Without exponential decay factors, early tokens contribute equal weight to $H_t$ as recent tokens, impairing performance on localized context tasks.
3. **Sequential Loop Overhead in PyTorch**: Python `for t in range(L)` loops incur CPU launch latency overhead unless replaced with parallel chunk scan kernels.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

A Triton kernel for HLA Attention implements parallel chunk scan algorithms, dividing sequence length $L$ into blocks and executing intra-chunk GEMM with inter-chunk state accumulation.

```python
# Triton Parallel Scan Kernel Blueprint for HLA Attention
import triton
import triton.language as tl

@triton.jit
def _hla_chunk_scan_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    L, d_k,
    BLOCK_L: tl.constexpr
):
    pid_bh = tl.program_id(0)
    
    # Persistent State Memory in SRAM [d_k, d_k]
    h_state = tl.zeros((d_k, d_k), dtype=tl.float32)
    
    for t in range(L):
        q = tl.load(Q_ptr + pid_bh * stride_qh + t * stride_ql + tl.arange(0, d_k))
        k = tl.load(K_ptr + pid_bh * stride_qh + t * stride_ql + tl.arange(0, d_k))
        v = tl.load(V_ptr + pid_bh * stride_qh + t * stride_ql + tl.arange(0, d_k))
        
        # Accumulate outer product
        h_state += k[:, None] * v[None, :]
        out_t = tl.sum(q[:, None] * h_state, axis=0)
        
        tl.store(Out_ptr + pid_bh * stride_qh + t * stride_ql + tl.arange(0, d_k), out_t)
```

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Gated HLA (Hadamard DeltaNet)**: Introduce learnable dynamic decay gates $f_t = \sigma(W_f x_t)$ to modulate memory accumulation: $H_t = f_t \odot H_{t-1} + K_t^\top V_t$.
2. **HLA-Mamba SSM Integration**: Combine HLA associative matrix states with continuous-time State Space Model discretization parameters ($\Delta t$).
3. **Low-Rank Factorized State Matrices**: Represent 2D state matrix $H_t$ as factorized low-rank vectors $U_t V_t^\top$ to further reduce FLOPs for large head dimensions ($d_k > 128$).
