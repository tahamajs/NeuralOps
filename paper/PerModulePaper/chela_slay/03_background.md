# 3. Background

## 3.1 Softmax Attention

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.1}
$$

with $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_h}$.

## 3.2 Linear Attention

Linear attention replaces softmax with a kernel $\phi$:

$$
\text{Attn}(Q, K, V)_i = \frac{\sum_j \phi(q_i)^\top \phi(k_j) v_j}{\sum_j \phi(q_i)^\top \phi(k_j)}
\tag{3.2}
$$

The **associative** trick:

$$
\text{Attn}(Q, K, V)_i = \frac{\phi(q_i)^\top \left[ \sum_j \phi(k_j) v_j^\top \right]}{\phi(q_i)^\top \left[ \sum_j \phi(k_j) \right]}
\tag{3.3}
$$

Computing the bracketed terms is $\mathcal{O}(L \cdot 2F \cdot d_h)$, independent of $L^2$.

## 3.3 Random Fourier Features

**Bochner's theorem** (Rahimi & Recht, 2007) states that a shift-invariant 
kernel $k(x, y) = k(x - y)$ can be represented as the Fourier transform of 
a probability distribution $p(\omega)$:

$$
k(x - y) = \mathbb{E}_\omega [e^{i \omega^\top (x - y)}]
\tag{3.4}
$$

Approximating this expectation with $F$ samples:

$$
k(x - y) \approx \frac{1}{F} \sum_{i=1}^{F} e^{i \omega_i^\top (x - y)}
\tag{3.5}
$$

Using Euler's formula, the real part is:

$$
k(x - y) \approx \frac{1}{F} \sum_{i=1}^{F} \cos(\omega_i^\top (x - y))
= \phi(x)^\top \phi(y)
\tag{3.6}
$$

where $\phi(x) = \frac{1}{\sqrt{F}} [\cos(\omega_i^\top x), \sin(\omega_i^\top x)]_{i=1}^F$.

## 3.4 The Yat-Kernel

The **Yat-Kernel** (Zhang et al., 2024) modifies the RFF to:

$$
\phi(x) = \frac{1}{F^{1/4}} [\cos(x \Omega) \,\|\, \sin(x \Omega)]
\tag{3.7}
$$

where $\Omega \in \mathbb{R}^{d_h \times F}$ is a random projection matrix.

**Properties:**

- $2F$ output dimension.
- $\phi(x)^\top \phi(y) = \frac{1}{\sqrt{F}} [\cos(x\Omega)^\top \cos(y\Omega) + \sin(x\Omega)^\top \sin(y\Omega)]$.
- Using $\cos(a)\cos(b) + \sin(a)\sin(b) = \cos(a - b)$, this equals 
  $\frac{1}{\sqrt{F}} \sum_i \cos(\omega_i^\top (x - y))$.

The Yat-Kernel is thus a scaled cosine-similarity kernel, which is 
**positive** (needed for stable linear attention).

## 3.5 Spherical Normalization

Given Q, K, we project them onto the unit sphere:

$$
\hat{Q} = \frac{Q}{\|Q\|_2 + \epsilon}, \quad
\hat{K} = \frac{K}{\|K\|_2 + \epsilon}
\tag{3.8}
$$

**Properties:**

- $\|\hat{Q}\|_2 = 1$ for all tokens and heads.
- Attention scores depend only on **direction**, not magnitude.
- Stabilizes the linear kernel: $\phi(\hat{K})$ has bounded magnitude.

## 3.6 Adaptive Gating

A learnable gate blends two streams:

$$
g = \sigma(W_g X + b_g)
\tag{3.9}
$$

$$
O = g \odot O^\text{dense} + (1 - g) \odot O^\text{linear}
\tag{3.10}
$$

**Initialization.** $W_g = 0$, $b_g = 5.0$ gives $g \approx 
0.993$ at step 0.

## 3.7 LoRA Recovery

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.11}
$$

with $B = 0$ initially.

## 3.8 Notation

| Symbol | Meaning |
|---|---|
| $Q, K, V$ | Query, Key, Value |
| $\hat{Q}, \hat{K}$ | Spherical-normalized Q, K |
| $\phi$ | Yat-Kernel feature map |
| $\Omega$ | RFF projection |
| $F$ | Number of RFF features |
| $g$ | Adaptive gate |
| $L, d_h$ | Sequence length, head dim |

---