<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Sliding-Window Attention

Given $Q, K, V \in \mathbb{R}^{L \times d_h}$ and window size $w$:

$$
S_{ij} = \frac{Q_i^\top K_j}{\sqrt{d_h}}, \quad
\text{mask}_{ij} = \begin{cases} 1 & |i - j| \leq w \\ 0 & \text{otherwise} \end{cases}
\tag{3.1}
$$

$$
O_i^\text{local} = \frac{\sum_j \text{mask}_{ij} \exp(S_{ij}) V_j}{\sum_j \text{mask}_{ij} \exp(S_{ij})}
\tag{3.2}
$$

**Complexity.** $\mathcal{O}(L w d_h)$ time, $\mathcal{O}(L w)$ memory.

**Causal version.** Only $j \le i$ and $i - j \leq w$ are attended.

## 3.2 Random Fourier Features

Rahimi & Recht (2007) show that:

$$
k(q, k) = \exp\!\left(-\frac{\|q - k\|^2}{2}\right) \approx \phi(q)^\top \phi(k)
\tag{3.3}
$$

where $\phi(x) = \frac{1}{\sqrt{m}} \left[\cos(\omega_1^\top x), \sin(\omega_1^\top x), \ldots\right]$ 
with $\omega_i \sim \mathcal{N}(0, I)$.

**For attention.** Performer uses $\phi(x) = \frac{1}{\sqrt{m}} \exp(\omega_i^\top x)$ 
with positive orthogonal features.

**NHA uses a simplified non-negative variant:**

$$
\phi(x) = \frac{1}{\sqrt{m}} (\mathrm{elu}(\Omega^\top x) + 1)
\tag{3.4}
$$

with $\Omega \in \mathbb{R}^{d_h \times m}$ orthogonal.

## 3.3 Orthogonal Projection Matrix

We construct $\Omega$ by block QR:

$$
\Omega = [Q_1, Q_2, \ldots, Q_b] \cdot \sqrt{d_h}
\tag{3.5}
$$

where each $Q_i \in \mathbb{R}^{d_h \times d_h}$ is from `torch.linalg.qr` 
on a random Gaussian matrix, and $b = \lceil m / d_h \rceil$.

**Why orthogonal?** Orthogonal projections give **lower variance** kernel 
approximation than i.i.d. Gaussians, at no extra cost.

## 3.4 Kernel Linear Attention

Given $\phi$, linear attention is:

$$
O_i^\text{global} = \frac{\phi(Q_i)^\top \left(\sum_j \phi(K_j) V_j^\top\right)}{\phi(Q_i)^\top \left(\sum_j \phi(K_j)\right)}
\tag{3.6}
$$

**Complexity.** $\mathcal{O}(L m d_h)$ time, $\mathcal{O}(m d_h)$ memory.

## 3.5 Causal Prefix Scan

For causal models, the cumulative state is:

$$
S^{(i)} = \sum_{j \leq i} \phi(K_j) V_j^\top \in \mathbb{R}^{m \times d_h}
\tag{3.7}
$$

$$
Z^{(i)} = \sum_{j \leq i} \phi(K_j) \in \mathbb{R}^m
\tag{3.8}
$$

**NHA's chunked implementation.** We process in chunks of 64:

$$
S \leftarrow S + \sum_{j \in \text{chunk}} \phi(K_j) V_j^\top
\tag{3.9}
$$

This bounds memory to $\mathcal{O}(m d_h)$ regardless of $L$.

## 3.6 Sigmoid Gate

Per-token gate:

$$
g_i = \sigma(w^\top x_i + b) \in (0, 1)
\tag{3.10}
$$

**Initialization.** $w = 0$, $b = 3.0$ gives $g_i = \sigma(3) \approx 0.95$ 
for all tokens. The model starts close to the local stream.

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $H$ | Number of heads |
| $d_h$ | Per-head dimension |
| $w$ | Window size |
| $m$ | RFF feature count |
| $\Omega$ | RFF projection matrix |
| $\phi$ | Feature map |
| $S, Z$ | Cumulative RFF state |
| $g_i$ | Gate at position $i$ |

---