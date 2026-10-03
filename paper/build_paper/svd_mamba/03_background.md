<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Multi-Head Self-Attention

For $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{3.2}
$$

$$
\text{MHSA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\tag{3.3}
$$

**Complexity:** $\mathcal{O}(L^2 d)$.

## 3.2 Selective State Space Models (Mamba)

### 3.2.1 Continuous SSM

A continuous SSM is defined by:

$$
h'(t) = A h(t) + B x(t), \qquad
y(t) = C h(t) + D x(t)
\tag{3.4}
$$

with state $h \in \mathbb{R}^N$, input $x \in \mathbb{R}$, output $y \in \mathbb{R}$.

### 3.2.2 Discretization

Using zero-order hold with step size $\Delta$:

$$
\bar{A} = \exp(\Delta A), \qquad
\bar{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot \Delta B
\tag{3.5}
$$

For diagonal $A$, this simplifies to:

$$
\bar{A}_t = \exp(\Delta_t A), \qquad
\bar{B}_t = \Delta_t B_t
\tag{3.6}
$$

### 3.2.3 Selective SSM (Mamba)

The **selective** aspect is that $\Delta_t, B_t, C_t$ are functions of the 
input:

$$
\Delta_t = \text{softplus}(W_\Delta x_t + b_\Delta), \qquad
B_t = W_B x_t, \qquad
C_t = W_C x_t
\tag{3.7}
$$

This makes the SSM **content-dependent**, enabling input-based selection 
of what to remember.

### 3.2.4 Recurrence

The SSM updates:

$$
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \qquad
y_t = C_t h_t + D x_t
\tag{3.8}
$$

**Parallel scan.** With diagonal $A$, the recurrence can be computed with 
a parallel prefix scan, achieving $\mathcal{O}(\log L)$ depth.

## 3.3 Chunked Associative Scan

For long sequences, a chunked scan is used. For chunk size $C$:

$$
h_{c, i} = \bar{A}_{c, i} h_{c, i-1} + \bar{B}_{c, i} x_{c, i}
\tag{3.9}
$$

Within a chunk, intra-chunk scan is used. Between chunks, the state is 
carried:

$$
h_{c+1, 0} = h_{c, C}
\tag{3.10}
$$

SVD-Mamba uses chunk size 64.

## 3.4 SVD Factorization

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^\top
\tag{3.11}
$$

Truncating to rank $r$:

$$
W \approx U_{:, :r} \Sigma_{:r} V_{:r, :}^\top
\tag{3.12}
$$

Parameter count: $r(m + n + 1)$ vs $mn$.

## 3.5 Gated Fusion

Two paths $Y_a, Y_b$ fused by:

$$
Y = \sigma(W_g X) \odot Y_a + (1 - \sigma(W_g X)) \odot Y_b
\tag{3.13}
$$

**Initialization.** Bias $b_g = -7$ gives $\sigma(-7) \approx 0.0009$, so 
$Y \approx Y_b$ at start.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden size |
| $d_h$ | Per-head dimension |
| $N$ | SSM state dimension |
| $\Delta$ | Discretization step |
| $\bar{A}$ | Discretized transition |
| $\bar{B}$ | Discretized input |
| $C$ | Output map |
| $D$ | Skip connection |
| $U, \Sigma, V$ | SVD factors |
| $r$ | SVD rank |

---