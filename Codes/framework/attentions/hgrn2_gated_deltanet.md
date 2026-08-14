# HGRN-2 Gated DeltaNet: Interleaved Hierarchical Linear Recurrence & Dense Attention Architecture

## 1. Executive Abstract & Architectural Motivation

Pure linear recurrent networks provide $\mathcal{O}(L)$ time and constant decoding memory, but can struggle with exact long-range pairwise token retrieval. Full Multi-Head Self-Attention achieves exact retrieval but suffers from $\mathcal{O}(L^2)$ quadratic complexity. The HGRN-2 Gated DeltaNet Architecture solves this trade-off by arranging a sequence of $N$ Transformer layers in a hybrid ratio $R:1$.

In this hybrid topology, $R$ consecutive layers execute fast linear $\mathcal{O}(L)$ HGRN-2 / Gated DeltaNet recurrence, while every $(R+1)$-th layer executes a dense Multi-Head Attention pass. This interleaved paradigm retains high throughput and low memory footprint while preserving exact retrieval capabilities.

---

## 2. Mathematical Formulations & Derivations

Given input sequence $X^{(0)} \in \mathbb{R}^{B \times L \times D}$, layer $l \in \{0, \dots, N-1\}$ executes based on hybrid ratio $R$:

\[
X^{(l+1)} = \begin{cases}
    \text{MHA-Block}\left( X^{(l)} \right), & \text{if } l \pmod{R+1} = R \\
    \text{HGRN2-Block}\left( X^{(l)} \right), & \text{otherwise}
\end{cases}
\]

### Layer Type 1: HGRN-2 Linear Recurrence Layer ($l \pmod{R+1} \ne R$):
\[
f_t = \sigma(W_f X_t^{(l)}), \quad i_t = \sigma(W_i X_t^{(l)})
\]
\[
h_t = \text{Clamp}\left( f_t \odot h_{t-1} + i_t \odot X_t^{(l)}, -1.0, 1.0 \right)
\]
\[
X^{(l+1)} = X^{(l)} + W_O h
\]

### Layer Type 2: Dense Multi-Head Attention Layer ($l \pmod{R+1} = R$):
\[
Q = X^{(l)} W_Q, \quad K = X^{(l)} W_K, \quad V = X^{(l)} W_V \in \mathbb{R}^{B \times H \times L \times d_k}
\]
\[
A = \text{Softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} + M_{\text{causal}} \right) \in \mathbb{R}^{B \times H \times L \times L}
\]
\[
X^{(l+1)} = X^{(l)} + (A V) W_O
\]

---

## 3. Step-by-Step Tensor Execution Trace

```
Input: X_0 [B, L, D]
  │
  ├──> Layer 0 (HGRN-2 Linear): X_1 = HGRN2_Block(X_0) ──> [B, L, D]
  ├──> Layer 1 (HGRN-2 Linear): X_2 = HGRN2_Block(X_1) ──> [B, L, D]
  │    ... (R linear layers)
  ├──> Layer R (Dense MHA Bypass): X_{R+1} = MHA_Block(X_R) ──> [B, L, D]
  │
  └──> Repeat hybrid stack for N layers ──> Final Output: [B, L, D]
```

### Detailed Tensor Transformations:

| Step | Operation | Tensor Shape | Description |
| :--- | :--- | :--- | :--- |
| 1 | Input | $[B, L, D]$ | Input hidden state sequence |
| 2 | HGRN-2 Linear Layers | $[B, L, D]$ | $R$ consecutive linear recurrence layers ($\mathcal{O}(L)$) |
| 3 | Dense MHA Layer | $[B, L, D]$ | 1 dense Multi-Head Attention layer ($\mathcal{O}(L^2)$) |
| 4 | Final Stack Output | $[B, L, D]$ | Hybrid representation sequence |

---

## 4. Analytical Complexity & Hardware Profiling

### FLOP and MAC Breakdown ($N$ Total Layers, $R$ Hybrid Ratio):
- **Number of HGRN-2 Layers**: $N \cdot \frac{R}{R+1}$ layers
- **Number of MHA Layers**: $N \cdot \frac{1}{R+1}$ layers
- **Total FLOPs**:
\[
\text{Total FLOPs} = N \cdot \left( \frac{R}{R+1} \cdot 12 B L D^2 + \frac{1}{R+1} \cdot (16 B L D^2 + 4 B L^2 D) \right)
\]

### Memory Footprint:
- **Static Weights**: $N \times 4 D^2$ parameters.
- **Decoding KV Cache Footprint**: Reduced by factor of $(R+1)$, storing KV cache only for the $\frac{N}{R+1}$ MHA layers.

---

## 5. Failure Modes, Rank Collapse & Edge Cases

1. **Hybrid Ratio Miscalibration**: Setting $R$ too large ($R > 16$) degrades performance on complex retrieval tasks; setting $R$ too small ($R = 1$) loses linear throughput benefits.
2. **Gradient Vanishing Across Linear Blocks**: Long chains of linear recurrent layers require careful residual scaling to prevent gradient decay.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

A Triton pipeline executes HGRN-2 linear layers using fused SRAM scans, invoking FlashAttention only on designated MHA layers.

```python
# Triton Hybrid Pipeline Architecture
# Executes HGRN-2 linear scans for R blocks, then triggers FlashAttention kernel
```

---

## 7. Next-Generation Innovations & Synthesis Opportunities

1. **Dynamic Adaptive Hybrid Ratio $R(x)$**: Learn an adaptive router that dynamically allocates MHA layers for complex inputs and linear layers for simple inputs.
2. **HGRN-2 + Mamba + MHA Tri-Hybrid**: Stack HGRN-2 element-wise layers, Mamba selective scan layers, and sparse MHA layers for ultimate LLM efficiency.
