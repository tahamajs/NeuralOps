# AdaSplash Attention: Adaptive Sparse-Splash Top-k Dynamic Context Routing

## 1. Executive Abstract & Architectural Motivation

Standard Softmax attention assigns non-zero probability mass to every sequence position, causing attention leakage across irrelevant context tokens and inducing computational inefficiency. AdaSplash Attention addresses this limitation by replacing the standard Softmax operator with Tsallis 1.5-entmax sparse activation mapping (Peters et al., 2019; Correia et al., 2019).

The 1.5-entmax activation function explicitly project attention score distributions onto the simplex boundary, yielding exact zero probabilities for low-scoring tokens. By integrating an iterative fixed-point algorithm to dynamically solve the entmax thresholding multiplier $\tau$, AdaSplash creates sparse, highly interpretable context routing maps. This eliminates redundant key-value memory access overhead and sharpens attention focus on high-relevance tokens.

---

## 2. Mathematical Formulations & Derivations

Given input activation sequence $X \in \mathbb{R}^{B \times L \times D}$, queries, keys, and values are projected as:

\[
Q = X W_Q \in \mathbb{R}^{B \times L \times D}, \quad K = X W_K \in \mathbb{R}^{B \times L \times D}, \quad V = X W_V \in \mathbb{R}^{B \times L \times D}
\]

Multi-head formatting yields $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_k}$ with $d_k = D / H$.

Unnormalized scaled dot-product logits with optional attention mask $M$ are computed as:

\[
S = \frac{Q K^\top}{\sqrt{d_k}} + M_{\text{mask}} \in \mathbb{R}^{B \times H \times L \times L}
\]

### Tsallis 1.5-Entmax Activation Mapping:
For score logit vector $z \in \mathbb{R}^L$, the 1.5-entmax transformation is defined as:

\[
\text{entmax}_{1.5}(z) = \left[ \frac{1}{2} (z - \tau \mathbf{1}) \right]_+^2
\]

where $[a]_+ = \max(a, 0)$ enforces exact non-negativity and sparsity, and $\tau \in \mathbb{R}$ is the unique Lagrange multiplier satisfying the normalization constraint $\sum_{j=1}^L p_j = 1$.

### Fixed-Point Iterative Normalization:
Because $\tau$ cannot be solved in closed form, AdaSplash employs fixed-point iterations starting from $p^{(0)} = \text{Softmax}(z)$:

\[
q^{(t)} = \sqrt{p^{(t)}} \in \mathbb{R}^{B \times H \times L \times L}
\]
\[
\tau^{(t)} = \sum_{j=1}^L q_{i, j}^{(t)} p_{i, j}^{(t)} \in \mathbb{R}^{B \times H \times L \times 1}
\]
\[
p^{(t+1)} = \frac{q^{(t)}}{\tau^{(t)} + \epsilon}
\]

Upon convergence ($|p^{(t+1)} - p^{(t)}|_\infty < \epsilon$), sparse probability matrix $A = p^{(T)}$ is obtained.

### Context Aggregation & Output Projection:
Sparse attention probabilities aggregate value representations:

\[
O = A V \in \mathbb{R}^{B \times H \times L \times d_k}, \quad Y = \text{Concat}(O_1, \dots, O_H) W_O \in \mathbb{R}^{B \times L \times D}
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
         ├──> MatMul: Q @ K.T / sqrt(d_k) ──> Logits S: [B, H, L, L]
         ├──> Additive Masking (S + M_mask) ──> [B, H, L, L]
         │
         ├──> Fixed-Point 1.5-Entmax Loop (t = 0 ... T-1):
         │      ├──> p_0 = Softmax(S, dim=-1)
         │      ├──> q_t = sqrt(p_t)
         │      ├──> tau_t = sum(q_t * p_t, dim=-1)
         │      └──> p_{t+1} = q_t / tau_t
         │
         ├──> Sparse Attention Matrix A = p_T ──> [B, H, L, L] (Exact Zeros)
         ├──> Dropout(A) ──> [B, H, L, L]
         ├──> MatMul: A @ V ──> Context: [B, H, L, d_k]
         │
         └──> transpose(1, 2) ──> contiguous() ──> [B, L, D]
         └──> Linear(D, D)(Context) ──> Output: [B, L, D]
```

### Detailed Tensor Transformations:

| Step | Operation | Tensor Shape | Description |
| :--- | :--- | :--- | :--- |
| 1 | Input | $[B, L, D]$ | Input hidden state sequence |
| 2 | $W_Q, W_K, W_V$ Projections | $[B, H, L, d_k]$ | Query, Key, Value states |
| 3 | Scaled Logits Computation | $[B, H, L, L]$ | Dense pairwise dot-product score matrix $S$ |
| 4 | Softmax Initialization | $[B, H, L, L]$ | Initial dense probabilities $p^{(0)}$ |
| 5 | Fixed-Point Entmax Iterations | $[B, H, L, L]$ | Fixed-point update $p^{(t+1)} = \sqrt{p^{(t)}} / \tau$ |
| 6 | Sparse Thresholding | $[B, H, L, L]$ | Probability matrix with exact zeros |
| 7 | Value Context Aggregation | $[B, H, L, d_k]$ | Sparse matrix multiplication $A V$ |
| 8 | Output Projection | $[B, L, D]$ | Projection $W_O$ to model dimension $D$ |

---

## 4. Analytical Complexity & Hardware Profiling

### FLOP and MAC Breakdown ($B$: Batch, $L$: Length, $D$: Dim, $H$: Heads, $T_{\text{iter}}$: Entmax Iterations):
- **Input Projections ($W_Q, W_K, W_V$)**: $6 B L D^2$ FLOPs
- **Score Matrix Logits ($Q K^\top$)**: $2 B H L^2 d_k = 2 B L^2 D$ FLOPs
- **Fixed-Point Entmax Loop ($T_{\text{iter}}$ iterations)**: $T_{\text{iter}} \times (6 B H L^2) = 6 T_{\text{iter}} B H L^2$ FLOPs
- **Context Aggregation ($A V$)**: $2 B H L^2 d_k = 2 B L^2 D$ FLOPs
- **Output Projection ($W_O$)**: $2 B L D^2$ FLOPs

**Total Complexity**:
- **Total MACs**: $4 B L D^2 + 2 B L^2 D + 3 T_{\text{iter}} B H L^2$
- **Total FLOPs**: $8 B L D^2 + 4 B L^2 D + 6 T_{\text{iter}} B H L^2$

### Memory Footprint:
- **Static Weights**: $4 D^2$ parameters.
- **Transient Activation Memory**: $\mathcal{O}(B \cdot H \cdot L^2)$ for attention score logit buffers.

---

## 5. Failure Modes, Rank Collapse & Edge Cases

1. **Complete Sparsity Collapse (Zero Gradient)**: If score logit initialization assigns negative values with high dispersion, entmax can truncate all attention probabilities to zero for specific queries, causing gradient disconnection.
2. **Fixed-Point Non-Convergence**: Near uniform probability distributions, fixed-point iterations $p^{(t+1)} = \sqrt{p^{(t)}} / \tau$ converge slowly, exceeding default iteration bounds ($T_{\text{iter}} = 10$).
3. **FP16 Square-Root Underflow**: Computing $\sqrt{p}$ under FP16 for small non-zero probabilities ($p < 10^{-7}$) underflows to zero, prematurely altering sparsity boundaries.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

Standard PyTorch entmax materializes intermediate matrices during each fixed-point iteration in HBM. A Triton implementation fuses the iterative loop into SRAM, loading score logits once and storing only active sparse entries.

```python
# Triton Fused 1.5-Entmax Kernel Blueprint
import triton
import triton.language as tl

@triton.jit
def _entmax15_fused_kernel(
    Logits_ptr, Out_ptr,
    stride_b, stride_h, stride_l,
    L, N_ITER: tl.constexpr,
    BLOCK_L: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    offs_l = tl.arange(0, BLOCK_L)
    logits = tl.load(Logits_ptr + pid_bh * stride_h + pid_m * stride_l + offs_l)
    
    # Initialize p = softmax(logits)
    max_logit = tl.max(logits, axis=0)
    p = tl.exp(logits - max_logit)
    p = p / tl.sum(p, axis=0)
    
    # Fixed-Point Iterative Loop in SRAM
    for _ in range(N_ITER):
        q = tl.sqrt(p)
        tau = tl.sum(q * p, axis=0)
        p = q / (tau + 1e-12)
        
    tl.store(Out_ptr + pid_bh * stride_h + pid_m * stride_l + offs_l, p)
```

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Hardware-Accelerated Block-Sparse Entmax**: Combine 1.5-entmax thresholding with FlashAttention block-tiling to drop entire zero tiles, enabling $\mathcal{O}(L \log L)$ speedups.
2. **Adaptive Learnable Alpha Entmax ($\alpha(x)$)**: Dynamically project $\alpha \in [1.0, 2.0]$ per token to transition smoothly between dense Softmax ($\alpha = 1.0$) and ultra-sparse entmax ($\alpha = 1.5$).
3. **AdaSplash-Mamba Dynamic Gating**: Use entmax sparse masks to gate Mamba selective state updates, preventing noise accumulation from low-relevance tokens.
