# CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel Fourier Features

## 1. Executive Abstract & Architectural Motivation

Standard Multi-Head Attention requires materializing an $L \times L$ Softmax matrix, scaling quadratically $\mathcal{O}(L^2)$ in time and memory. CHELA-SLAY (Spherical Linearized Attention with Yat-Kernel) reformulates query-key contextualization by combining unit-hypersphere projection with trigonometric random Fourier features (Yat-Kernel).

By normalizing Queries and Keys onto the unit sphere $\mathbb{S}^{d-1}$ ($\widehat{Q} = Q / \|Q\|_2, \widehat{K} = K / \|K\|_2$), CHELA-SLAY eliminates vector magnitude distortions. It then applies a positive trigonometric random feature mapping $\phi(x) = \frac{1}{\sqrt{m}} [\cos(W x), \sin(W x)]^\top > 0$, enabling linear associative matrix multiplication $\phi(\widehat{Q}) (\phi(\widehat{K})^\top V)$. This reduces time and memory complexity to $\mathcal{O}(L \cdot m \cdot d)$, guaranteeing stable linear scaling with respect to sequence length $L$.

---

## 2. Mathematical Formulations & Derivations

Given input sequence representations $X \in \mathbb{R}^{B \times L \times D}$, query, key, and value states are projected as:

\[
Q = X W_Q \in \mathbb{R}^{B \times L \times D}, \quad K = X W_K \in \mathbb{R}^{B \times L \times D}, \quad V = X W_V \in \mathbb{R}^{B \times L \times D}
\]

Multi-head formatting yields $Q, K, V \in \mathbb{R}^{B \times H \times L \times d}$ with $d = D / H$.

### 1. Spherical Unit Normalization:
Query and key vectors are projected onto the unit hypersphere $\mathbb{S}^{d-1}$:

\[
\widehat{Q} = \frac{Q}{\|Q\|_2 + \epsilon} \in \mathbb{R}^{B \times H \times L \times d}, \quad \widehat{K} = \frac{K}{\|K\|_2 + \epsilon} \in \mathbb{R}^{B \times H \times L \times d}
\]

### 2. Trigonometric Yat-Kernel Random Fourier Mapping ($\phi$):
Given a non-trainable random Gaussian projection matrix $W \in \mathbb{R}^{m \times d}$ with $W_{i, j} \sim \mathcal{N}(0, \sigma^2)$, the Yat-Kernel feature map $\phi(x) \in \mathbb{R}^{2m}$ is defined as:

\[
\phi(x) = \frac{1}{\sqrt{m}} \left[ \cos(W x); \; \sin(W x) \right] \in \mathbb{R}^{B \times H \times L \times 2m}
\]

Applying $\phi$ to spherical queries and keys yields feature matrices $\phi(\widehat{Q}), \phi(\widehat{K}) \in \mathbb{R}^{B \times H \times L \times 2m}$.

### 3. Associative Linear Attention Recurrence:
Using matrix associativity, the unnormalized context is evaluated via matrix multiplication:

\[
KV = \phi(\widehat{K})^\top V = \sum_{t=1}^L \phi(\widehat{K}_t) \otimes V_t \in \mathbb{R}^{B \times H \times 2m \times d}
\]
\[
K_{\text{sum}} = \sum_{t=1}^L \phi(\widehat{K}_t) \in \mathbb{R}^{B \times H \times 1 \times 2m}
\]

The normalized context $O \in \mathbb{R}^{B \times H \times L \times d}$ is evaluated per sequence position:

\[
O = \frac{\phi(\widehat{Q}) KV}{\phi(\widehat{Q}) K_{\text{sum}}^\top + \epsilon} \in \mathbb{R}^{B \times H \times L \times d}
\]

Final projection yields $Y = \text{Concat}(O_1, \dots, O_H) W_O \in \mathbb{R}^{B \times L \times D}$.

---

## 3. Step-by-Step Tensor Execution Trace

```
Input: hidden_states [B, L, D]
  │
  ├──> Q, K, V Projections ──> [B, H, L, d]
  │
  ├──> Spherical Normalization:
  │      ├──> Q_hat = Q / ||Q||_2 ──> [B, H, L, d]
  │      └──> K_hat = K / ||K||_2 ──> [B, H, L, d]
  │
  ├──> Yat-Kernel Fourier Mapping:
  │      ├──> cos_feat = cos(Q_hat @ W.T) / sqrt(m) ──> [B, H, L, m]
  │      ├──> sin_feat = sin(Q_hat @ W.T) / sqrt(m) ──> [B, H, L, m]
  │      └──> phi(Q_hat) = Concat([cos_feat, sin_feat], dim=-1) ──> [B, H, L, 2m]
  │
  ├──> Associative Memory KV: phi(K_hat).T @ V ──────────> [B, H, 2m, d]
  ├──> Key Sum Vector K_sum: phi(K_hat).sum(dim=-2) ─────> [B, H, 1, 2m]
  │
  ├──> Numerator: phi(Q_hat) @ KV ─────────────────────────> [B, H, L, d]
  ├──> Denominator: phi(Q_hat) @ K_sum.T ──────────────────> [B, H, L, 1]
  ├──> Output Context: Numerator / Denominator ─────────> [B, H, L, d]
  │
  └──> transpose(1, 2) ──> contiguous() ──> [B, L, D]
  └──> Linear(D, D)(Context) ──> Output: [B, L, D]
```

### Detailed Tensor Transformations:

| Step | Operation | Tensor Shape | Description |
| :--- | :--- | :--- | :--- |
| 1 | Input | $[B, L, D]$ | Input sequence representation |
| 2 | Linear Projections | $[B, H, L, d]$ | Multi-head $Q, K, V$ states |
| 3 | L2 Spherical Normalization | $[B, H, L, d]$ | Unit-sphere projection $\widehat{Q}, \widehat{K}$ |
| 4 | Trigonometric Yat Mapping | $[B, H, L, 2m]$ | Trigonometric Fourier features $[\cos(W x); \sin(W x)]$ |
| 5 | Associative Matrix Product ($KV$) | $[B, H, 2m, d]$ | Outer-product aggregation over sequence length $L$ |
| 6 | Key Sum Partition Function | $[B, H, 1, 2m]$ | Summed key feature normalization vector |
| 7 | Query Context Numerator | $[B, H, L, d]$ | Query projection against accumulated $KV$ state |
| 8 | Output Projection | $[B, L, D]$ | Projection $W_O$ to model dimension $D$ |

---

## 4. Analytical Complexity & Hardware Profiling

### FLOP and MAC Breakdown ($B$: Batch, $L$: Length, $D$: Dim, $H$: Heads, $m$: Fourier Features):
- **Input Projections ($W_Q, W_K, W_V$)**: $6 B L D^2$ FLOPs
- **L2 Spherical Normalization**: $6 B H L d = 6 B L D$ FLOPs
- **Trigonometric Random Projections ($W \widehat{Q}, W \widehat{K}$)**: $2 \times (2 B H L d m) = 4 B L m D$ FLOPs
- **Sine / Cosine Evaluators**: $4 B H L m$ FLOPs
- **Associative State Matrix ($KV = \phi(\widehat{K})^\top V$)**: $2 B H (2m) L d = 4 B L m D$ FLOPs
- **Context Numerator ($\phi(\widehat{Q}) KV$)**: $2 B H L (2m) d = 4 B L m D$ FLOPs
- **Output Projection ($W_O$)**: $2 B L D^2$ FLOPs

**Total Complexity**:
- **Total MACs**: $4 B L D^2 + 6 B L m D$
- **Total FLOPs**: $8 B L D^2 + 12 B L m D$ ($\mathcal{O}(L)$ linear complexity)

### Memory Footprint:
- **Static Weights**: $4 D^2$ model parameters $+ m \cdot d$ random weight matrix $W$.
- **Transient Activation Memory**: Bounded by $\mathcal{O}(B \cdot H \cdot m \cdot d + B \cdot H \cdot L \cdot m)$.

---

## 5. Failure Modes, Rank Collapse & Edge Cases

1. **Trigonometric Cancellation in FP16**: Under FP16 execution, small angular differences $\Delta \theta = W (q - k)$ can cause catastrophic cancellation in $\cos(\cdot)$ and $\sin(\cdot)$, leading to zero inner products.
2. **Frequency Hyperparameter Sensitivity**: The standard deviation $\sigma$ of Gaussian matrix $W$ acts as a bandwidth hyperparameter. If $\sigma$ is miscalibrated, feature maps become either over-concentrated or uniform.
3. **Causal Scan Underflow**: In causal decoding, small feature values in $\phi(\widehat{K})$ can accumulate underflow errors across long sequence horizons ($L > 32768$).

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

A Triton kernel for CHELA-SLAY fuses L2 normalization, trigonometric projection, and linear matrix state accumulation directly into SRAM.

```python
# Triton Fused CHELA-SLAY Linear Attention Blueprint
import triton
import triton.language as tl

@triton.jit
def _chela_slay_fused_kernel(
    Q_ptr, K_ptr, V_ptr, W_ptr, Out_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    L, m, d,
    BLOCK_M: tl.constexpr
):
    pid_bh = tl.program_id(0)
    
    # Initialize persistent state memory in SRAM [2m, d]
    kv_state = tl.zeros((2 * m, d), dtype=tl.float32)
    k_sum = tl.zeros((2 * m,), dtype=tl.float32)
    
    for t in range(L):
        q = tl.load(Q_ptr + pid_bh * stride_qh + t * stride_ql + tl.arange(0, d))
        k = tl.load(K_ptr + pid_bh * stride_qh + t * stride_ql + tl.arange(0, d))
        v = tl.load(V_ptr + pid_bh * stride_qh + t * stride_ql + tl.arange(0, d))
        
        # Unit sphere normalize
        q_norm = q / (tl.sqrt(tl.sum(q * q)) + 1e-6)
        k_norm = k / (tl.sqrt(tl.sum(k * k)) + 1e-6)
        
        # Trigonometric projection
        angle_k = tl.dot(k_norm[None, :], W_ptr)
        phi_k = tl.cat([tl.cos(angle_k), tl.sin(angle_k)], axis=1) / tl.sqrt(m.to(tl.float32))
        
        kv_state += tl.dot(tl.trans(phi_k), v[None, :])
        k_sum += tl.sum(phi_k, axis=0)
```

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Orthogonal Spherical Fourier Features (ORF-SLAY)**: Replace i.i.d. random matrix $W$ with Gram-Schmidt orthogonalized Hadamard matrices to minimize Fourier feature estimation variance.
2. **CHELA-Mamba State Space Hybrid**: Combine spherical unit-norm Yat-Kernels with selective state-space discretization parameters ($\Delta t$) for continuous-time sequence modeling.
3. **FP8 Quantized Trigonometric Feature Maps**: Quantize Fourier feature maps $[\cos(W x); \sin(W x)]$ to FP8 (E4M3 format) for 2x memory throughput on NVIDIA Hopper architectures.
