# Infini-Attention: Compressive Memory & Local Sliding-Window Hybrid Attention

## 1. Executive Abstract & Architectural Motivation

Standard Multi-Head Attention cannot scale to infinitely long context streams due to the quadratic growth of KV cache memory ($\mathcal{O}(L^2)$). Infini-Attention (MHA with Compressive Memory, Mhar et al., 2024) solves this bottleneck by combining local masked sliding-window attention with persistent compressive memory tensors $M \in \mathbb{R}^{H \times d_k \times d_v}$.

Infini-Attention processes inputs in bounded streaming segments of length $N$. Within each segment, standard masked sliding-window attention computes fine-grained local dependencies. Simultaneously, historical key-value states from past segments are compressed into a non-bounded associative memory matrix $M$ using linear feature updates. Fusing local sliding-window context with compressive memory retrieval enables infinite sequence length processing with constant bounded memory overhead.

---

## 2. Mathematical Formulations & Derivations

Given input sequence segment representations $X \in \mathbb{R}^{B \times N \times D}$, query, key, and value states are projected as:

\[
Q = X W_Q \in \mathbb{R}^{B \times N \times D}, \quad K = X W_K \in \mathbb{R}^{B \times N \times D}, \quad V = X W_V \in \mathbb{R}^{B \times N \times D}
\]

Multi-head formatting yields $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_k}$ with $d_k = D / H$.

### 1. Local Sliding-Window Attention Branch:
Local attention scores are computed within segment length $N$ (or sliding window $W \le N$):

\[
S_{\text{local}} = \frac{Q K^\top}{\sqrt{d_k}} + M_{\text{window}} \in \mathbb{R}^{B \times H \times N \times N}
\]
\[
A_{\text{local}} = \text{Softmax}\left( S_{\text{local}} \right) \in \mathbb{R}^{B \times H \times N \times N}
\]
\[
O_{\text{local}} = A_{\text{local}} V \in \mathbb{R}^{B \times H \times N \times d_k}
\]

### 2. Compressive Memory Retrieval Branch:
Historical segment context is retrieved from persistent memory matrix $M \in \mathbb{R}^{B \times H \times d_k \times d_k}$ and normalization vector $z \in \mathbb{R}^{B \times H \times d_k \times 1}$:

\[
\phi(Q) = \text{ELU}(Q) + 1.0 > 0 \in \mathbb{R}^{B \times H \times N \times d_k}
\]
\[
O_{\text{mem}} = \frac{\phi(Q) M}{\phi(Q) z + \epsilon} \in \mathbb{R}^{B \times H \times N \times d_k}
\]

### 3. Fused Output & Compressive Memory State Update:
Local and compressive memory context outputs are fused using a learnable gating parameter $\beta \in \mathbb{R}^H$:

\[
O_{\text{fused}} = \sigma(\beta) \odot O_{\text{local}} + (1 - \sigma(\beta)) \odot O_{\text{mem}} \in \mathbb{R}^{B \times H \times N \times d_k}
\]

After processing the segment, compressive memory states $M$ and $z$ are updated with current segment keys and values:

\[
\phi(K) = \text{ELU}(K) + 1.0 > 0 \in \mathbb{R}^{B \times H \times N \times d_k}
\]
\[
M_{\text{new}} = M + \phi(K)^\top V \in \mathbb{R}^{B \times H \times d_k \times d_k}
\]
\[
z_{\text{new}} = z + \sum_{i=1}^N \phi(K)_i^\top \in \mathbb{R}^{B \times H \times d_k \times 1}
\]

Final output projection produces $Y = \text{Concat}(O_{\text{fused}, 1}, \dots, O_{\text{fused}, H}) W_O \in \mathbb{R}^{B \times N \times D}$.

---

## 3. Step-by-Step Tensor Execution Trace

```
Input Segment: hidden_states [B, N, D]
  │
  ├──> Q, K, V Projections ──> [B, H, N, d_k]
  │
  ├──> Local Branch:
  │      ├──> Local Softmax Scores S_local = Q @ K.T / sqrt(d_k) + M_win ──> [B, H, N, N]
  │      └──> Local Context O_local = Softmax(S_local) @ V ──> [B, H, N, d_k]
  │
  ├──> Memory Retrieval Branch:
  │      ├──> phi(Q) = ELU(Q) + 1.0 ──> [B, H, N, d_k]
  │      └──> Memory Context O_mem = phi(Q) @ M / (phi(Q) @ z + eps) ──> [B, H, N, d_k]
  │
  ├──> Gated Fusion: O_fused = sigmoid(beta) * O_local + (1 - sigmoid(beta)) * O_mem
  │
  ├──> Memory State Update:
  │      ├──> M_new = M + phi(K).T @ V ──> [B, H, d_k, d_k]
  │      └──> z_new = z + phi(K).sum(dim=2) ──> [B, H, d_k, 1]
  │
  └──> transpose(1, 2) ──> contiguous() ──> [B, N, D]
  └──> Linear(D, D)(O_fused) ──> Segment Output: [B, N, D]
```

### Detailed Tensor Transformations:

| Step | Operation | Tensor Shape | Description |
| :--- | :--- | :--- | :--- |
| 1 | Input Segment | $[B, N, D]$ | Hidden state segment |
| 2 | Projections | $[B, H, N, d_k]$ | Query, Key, Value states |
| 3 | Local Sliding Softmax | $[B, H, N, N]$ | Local segment score matrix |
| 4 | Local Context $O_{\text{local}}$ | $[B, H, N, d_k]$ | High-resolution local context |
| 5 | Compressive Memory Retrieval | $[B, H, N, d_k]$ | Long-range context retrieved from persistent state $M$ |
| 6 | Sigmoidal Gated Fusion | $[B, H, N, d_k]$ | Convex combination $\sigma(\beta) O_{\text{local}} + (1-\sigma(\beta)) O_{\text{mem}}$ |
| 7 | Memory State Update | $[B, H, d_k, d_k]$ | Cumulative ELU key-value update $M \leftarrow M + \phi(K)^\top V$ |
| 8 | Output Projection | $[B, N, D]$ | Segment output transformed to model dimension $D$ |

---

## 4. Analytical Complexity & Hardware Profiling

### FLOP and MAC Breakdown ($B$: Batch, $N$: Segment Len, $D$: Dim, $H$: Heads, $d_k$: Head Dim):
- **Input Projections ($W_Q, W_K, W_V$)**: $6 B N D^2$ FLOPs
- **Local Sliding Window Attention ($Q K^\top, A V$)**: $4 B H N^2 d_k = 4 B N^2 D$ FLOPs
- **Compressive Memory Retrieval ($\phi(Q) M$)**: $2 B H N d_k^2 = 2 B N D d_k$ FLOPs
- **Compressive Memory State Update ($\phi(K)^\top V$)**: $2 B H N d_k^2 = 2 B N D d_k$ FLOPs
- **Output Projection ($W_O$)**: $2 B N D^2$ FLOPs

**Total Complexity per Segment**:
- **Total MACs**: $4 B N D^2 + 2 B N^2 D + 2 B N D d_k$
- **Total FLOPs**: $8 B N D^2 + 4 B N^2 D + 4 B N D d_k$ ($\mathcal{O}(N)$ linear sequence length scaling per segment)

### Memory Footprint:
- **Static Weights**: $4 D^2 + H$ parameters.
- **Persistent Compressive State Memory**: Bounded by $\mathcal{O}(B \cdot H \cdot d_k^2)$ persistent tensor memory, independent of total past stream length $L$.

---

## 5. Failure Modes, Rank Collapse & Edge Cases

1. **Unbounded Compressive Memory Saturation**: Updating $M \leftarrow M + \phi(K)^\top V$ without decay causes memory matrix norm growth over millions of tokens, leading to FP16 overflow.
2. **Feature Smearing in Compressive Retrieval**: Linear ELU feature maps $\phi(K)$ aggregate historical tokens additively, causing fine details from 100k+ past steps to over-smooth.
3. **Gating Parameter Collapse**: If gate logit $\beta \gg 0$, $\sigma(\beta) \to 1$, completely deactivating the compressive memory branch and reducing the model to a standard sliding-window Transformer.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

A Triton kernel for Infini-Attention fuses local sliding window FlashAttention with persistent compressive memory retrieval in SRAM.

```python
# Triton Fused Infini-Attention Blueprint
import triton
import triton.language as tl

@triton.jit
def _infini_fused_kernel(
    Q_ptr, K_ptr, V_ptr, M_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    N, d_k,
    BLOCK_N: tl.constexpr
):
    pid_bh = tl.program_id(0)
    
    # Load persistent compressive memory state M [d_k, d_k] into SRAM
    m_state = tl.load(M_ptr + pid_bh * d_k * d_k + tl.arange(0, d_k)[:, None] * d_k + tl.arange(0, d_k)[None, :])
    
    # Compute local FlashAttention tile + compressive memory retrieval tile
    # Fuse gated combination directly in SRAM
```

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Decay-Regularized Infini Memory**: Incorporate learnable exponential decay factors $\gamma \in (0, 1)$ into state updates: $M_{\text{new}} = \gamma M + \phi(K)^\top V$ to prevent long-term memory saturation.
2. **Infini-Mamba Hybrid**: Interleave Infini-Attention compressive memory layers with Mamba selective scan modules to unify continuous state space recurrences with long-context memory retrieval.
3. **Quantized FP8 Compressive Memory**: Quantize persistent memory matrix $M$ to FP8 (E4M3 format) for 2x memory reduction during long streaming inference sessions.
