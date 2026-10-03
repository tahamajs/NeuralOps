<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Softmax Attention

For $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_h}$:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.1}
$$

**Complexity:** $\mathcal{O}(N^2 d_h)$ in time, $\mathcal{O}(N^2)$ in memory.

## 3.2 Kernelized Linear Attention

Replace the softmax kernel with a **positive feature map**:

$$
\text{Attn}(Q, K, V)_t = \frac{\phi(Q_t) \sum_{j \leq t} \phi(K_j) V_j^\top}{\phi(Q_t) \sum_{j \leq t} \phi(K_j)}
\tag{3.2}
$$

with $\phi(x) = \text{ELU}(x) + 1 \in \mathbb{R}_{>0}^{d_h}$.

**Associative form.** Define $S_t = \sum_{j \leq t} \phi(K_j) V_j^\top$ and 
$Z_t = \sum_{j \leq t} \phi(K_j)$. Then:

$$
\text{Attn}_t = \phi(Q_t) S_t / (\phi(Q_t) Z_t)
\tag{3.3}
$$

**Complexity:** $\mathcal{O}(N d_h^2)$ — linear in $N$.

## 3.3 Exponential Decay Gating

Add a per-position retention factor:

$$
S_t = \lambda_t S_{t-1} + \phi(K_t) V_t^\top
\tag{3.4}
$$

with $\lambda_t \in (0, 1]$ computed from the input:

$$
\lambda_t = \exp(-\text{softplus}(W_g x_t + b_g))
\tag{3.5}
$$

**Interpretation.** $\lambda_t$ controls how much of the past is retained. 
When $\lambda_t \approx 1$, the memory is preserved; when $\lambda_t \approx 0$, 
the memory is reset.

**Initialization.** We initialize $b_g$ so that the initial decay times 
$\tau = -\log \lambda$ are log-spaced between 1 and 1024 tokens across heads.

## 3.4 Chunked Parallel Scan

The recurrence $S_t = \lambda_t S_{t-1} + u_t$ can be computed sequentially 
in $\mathcal{O}(N)$. For parallelism, we **chunk** the sequence:

Within a chunk of size $C$:

$$
S_{c, i}^{\text{intra}} = e^{\lambda_{c, i}} \sum_{j \leq i} e^{-\lambda_{c, j}} u_{c, j}
\tag{3.6}
$$

where $\lambda_{c, i} = \sum_{j \leq i} \log \lambda_{c, j}$.

Between chunks:

$$
S_{c, i} = S_{c, i}^{\text{intra}} + e^{\lambda_{c, i}} S_{c-1, C}
\tag{3.7}
$$

This gives $\mathcal{O}(N)$ complexity with $\mathcal{O}(\log C)$ parallel 
depth within each chunk.

## 3.5 SVD Factorization

For $W \in \mathbb{R}^{m \times n}$:

$$
W \approx U_{:, :r} \Sigma_{:r} V_{:r, :}^\top
\tag{3.8}
$$

Parameter count: $r(m + n + 1)$ vs $mn$. For $m = n = 768$, $r = 32$: 
$\sim$8× reduction.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $d_h$ | Per-head dimension |
| $r$ | SVD rank |
| $r_L$ | LoRA rank |
| $\phi(x)$ | Feature map |
| $\lambda_t$ | Decay at position $t$ |
| $S_t, Z_t$ | Memory state |
| $C$ | Chunk size |
| $\tau$ | Decay time = $-\log \lambda$ |

---