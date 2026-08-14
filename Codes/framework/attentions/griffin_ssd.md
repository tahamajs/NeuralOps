# Griffin SSD: State Space Duality Recurrent Linear Attention Kernel

## 1. Executive Abstract & Architectural Motivation

Griffin SSD (State Space Duality, De et al., 2024 / Dao & Gu, 2024) establishes the formal mathematical equivalence between State Space Models (SSMs) and Recurrent Linear Attention (RLA). Standard attention incurs $\mathcal{O}(L^2)$ time and space complexity, while SSMs operate via 1D state recurrences. Griffin SSD unifies these two paradigms.

By framing state space updates as an associative matrix state recurrence $H_t = H_{t-1} + k_t \otimes v_t$, Griffin SSD computes linear sequence contextualization in $\mathcal{O}(L \cdot D^2)$ time and $\mathcal{O}(D^2)$ space. This matrix state duality enables parallel training via matrix multiplications (attention view) and constant-memory autoregressive decoding via state recurrences (SSM view).

---

## 2. Mathematical Formulations & Derivations

Given input sequence representations $X \in \mathbb{R}^{B \times L \times D}$, queries, keys, and values are projected as:

\[
Q = X W_Q \in \mathbb{R}^{B \times L \times D}, \quad K = X W_K \in \mathbb{R}^{B \times L \times D}, \quad V = X W_V \in \mathbb{R}^{B \times L \times D}
\]

### 1. Recurrent Matrix State Accumulation (SSM Dual View):
An associative 2D memory matrix $H_t \in \mathbb{R}^{B \times D \times D}$ is initialized to zero ($H_0 = \mathbf{0}$) and updated recursively at step $t$:

\[
H_t = H_{t-1} + k_t \otimes v_t = H_{t-1} + k_t v_t^\top \in \mathbb{R}^{B \times D \times D}
\]

### 2. Context Retrieval & Matrix Contraction:
Query vector $q_t \in \mathbb{R}^{B \times D}$ interacts with state matrix $H_t$ via matrix-vector multiplication:

\[
y_t = q_t H_t = \sum_{j=1}^t \left( q_t^\top k_j \right) v_j \in \mathbb{R}^{B \times D}
\]

### 3. Parallel Matrix Multiplication Dual View (Attention View):
During parallel training over sequence length $L$, the unnormalized causal attention matrix $A_{\text{causal}} \in \mathbb{R}^{B \times L \times L}$ is evaluated directly:

\[
A_{\text{causal}} = \text{Tril}\left( Q K^\top \right) \in \mathbb{R}^{B \times L \times L}
\]
\[
Y = A_{\text{causal}} V \in \mathbb{R}^{B \times L \times D}
\]

The mathematical duality guarantees that $y_t$ evaluated recursively equals row $t$ of $A_{\text{causal}} V$.

Final output projection produces $O = Y W_O \in \mathbb{R}^{B \times L \times D}$.

---

## 3. Step-by-Step Tensor Execution Trace

```
Input: hidden_states [B, L, D]
  │
  ├──> Q = Linear(D, D)(hidden_states) ──> [B, L, D]
  ├──> K = Linear(D, D)(hidden_states) ──> [B, L, D]
  └──> V = Linear(D, D)(hidden_states) ──> [B, L, D]
         │
         ├──> Recurrent Path (Autoregressive Decoding):
         │      ├──> Outer Product: k_t.unsqueeze(-1) @ v_t.unsqueeze(-2) ──> [B, D, D]
         │      ├──> State Recurrence: H_t = H_{t-1} + OuterProduct ──> [B, D, D]
         │      └──> Output Query Contraction: y_t = (q_t.unsqueeze(-2) @ H_t).squeeze(-2) ──> [B, D]
         │
         ├──> Parallel Path (Training):
         │      ├──> Causal Matrix: A = Tril(Q @ K.T) ──> [B, L, L]
         │      └──> Parallel Context: Y = A @ V ──> [B, L, D]
         │
         └──> Linear(D, D)(Y) ──> Output: [B, L, D]
```

### Detailed Tensor Transformations:

| Step | Operation | Tensor Shape | Description |
| :--- | :--- | :--- | :--- |
| 1 | Input | $[B, L, D]$ | Input sequence representation |
| 2 | Linear Projections | $[B, L, D]$ | $Q, K, V$ states |
| 3 | Outer Product ($k_t \otimes v_t$) | $[B, D, D]$ | Per-step rank-1 outer product matrix update |
| 4 | State Matrix Accumulation | $[B, D, D]$ | Cumulative memory state $H_t = H_{t-1} + k_t v_t^\top$ |
| 5 | Query State Projection | $[B, D]$ | Output extraction $y_t = q_t H_t$ |
| 6 | Parallel Training Matrix ($Q K^\top$) | $[B, L, L]$ | Causal lower-triangular score matrix |
| 7 | Out Projection | $[B, L, D]$ | Final projection $W_O$ to model dimension $D$ |

---

## 4. Analytical Complexity & Hardware Profiling

### FLOP and MAC Breakdown ($B$: Batch, $L$: Length, $D$: Model Dim):
- **Input Projections ($W_Q, W_K, W_V$)**: $6 B L D^2$ FLOPs
- **Recurrent State Updates ($k_t v_t^\top$)**: $L \times (2 B D^2) = 2 B L D^2$ FLOPs
- **State Query Projection ($q_t H_t$)**: $L \times (2 B D^2) = 2 B L D^2$ FLOPs
- **Output Projection ($W_O$)**: $2 B L D^2$ FLOPs

**Total Complexity**:
- **Total MACs**: $6 B L D^2$
- **Total FLOPs**: $12 B L D^2$ ($\mathcal{O}(L)$ strictly linear sequence scaling)

### Memory Footprint:
- **Static Weights**: $4 D^2$ parameters.
- **Decoding Memory Footprint**: Bounded by $\mathcal{O}(B \cdot D^2)$ state memory per sequence, eliminating KV cache growth.

---

## 5. Failure Modes, Rank Collapse & Edge Cases

1. **Unbounded Memory Magnitude Accumulation**: Without decay factors, $H_t = H_{t-1} + k_t v_t^\top$ grows monotonically over long context horizons ($L > 65536$), leading to FP16 overflow.
2. **Lack of Recency Weighting**: Equal weighting across all past positions impairs performance on localized context retrieval.
3. **High FLOP Cost for Large $D$**: Matrix state size $D \times D$ requires $2 D^2$ FLOPs per token step.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

A Triton kernel for Griffin SSD executes parallel chunk scan operations, partitioning sequence length $L$ into blocks and executing intra-chunk GEMM with inter-chunk state accumulation.

```python
# Triton Parallel Scan Blueprint for Griffin SSD
import triton
import triton.language as tl

@triton.jit
def _griffin_ssd_chunk_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_ql, stride_qd,
    L, D,
    BLOCK_L: tl.constexpr
):
    pid_b = tl.program_id(0)
    
    # Persistent matrix state memory in SRAM [D, D]
    h_state = tl.zeros((D, D), dtype=tl.float32)
    
    for t in range(L):
        q = tl.load(Q_ptr + pid_b * stride_qb + t * stride_ql + tl.arange(0, D))
        k = tl.load(K_ptr + pid_b * stride_qb + t * stride_ql + tl.arange(0, D))
        v = tl.load(V_ptr + pid_b * stride_qb + t * stride_ql + tl.arange(0, D))
        
        # Outer product accumulation
        h_state += k[:, None] * v[None, :]
        y_t = tl.sum(q[:, None] * h_state, axis=0)
        
        tl.store(Out_ptr + pid_b * stride_qb + t * stride_ql + tl.arange(0, D), y_t)
```

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Decay-Gated SSD (Mamba-SSD Dual)**: Add input-dependent decay parameters $\gamma_t = \sigma(W_{\gamma} x_t)$ to modulate matrix updates: $H_t = \gamma_t H_{t-1} + k_t v_t^\top$.
2. **Griffin-Mamba Hybrid Models**: Interleave Griffin SSD linear recurrence blocks with Mamba selective scan layers to achieve optimal long-context language modeling.
3. **FP8 Quantized Matrix States**: Quantize 2D matrix state $H_t$ to FP8 format for 2x memory reduction during long streaming decoding.
