<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Singular Value Decomposition

For any $W \in \mathbb{R}^{m \times n}$, the SVD is:

$$
W = U \Sigma V^\top
\tag{3.1}
$$

with $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ 
orthogonal, and $\Sigma \in \mathbb{R}^{m \times n}$ diagonal with 
$\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD.** Keeping the top-$r$ singular values gives:

$$
W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top
\tag{3.2}
$$

with error $\|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}$.

**Rank-$r$ approximation.** For a matrix of rank $r$, the factorization 
requires $r(m + n + 1)$ parameters, vs $mn$ for the full matrix. For 
$m = n = 768$ and $r = 64$: 98,368 vs 589,824 — a **6×** reduction.

## 3.2 Activation-Aware Weighting

For a linear layer $y = Wx + b$, the reconstruction error is:

$$
\mathbb{E}_x[\|Wx - W_r x\|_2^2] = \text{Tr}[(W - W_r) C_X (W - W_r)^\top]
\tag{3.3}
$$

where $C_X = \mathbb{E}[x x^\top]$ is the input covariance.

Under a **diagonal** assumption $C_X = \text{diag}(F_1, \dots, F_n)$:

$$
\mathbb{E}_x[\|Wx - W_r x\|_2^2] = \|(W - W_r) \text{diag}(\sqrt{F})\|_F^2
\tag{3.4}
$$

So the weighted problem reduces to unweighted SVD on the transformed matrix:

$$
W_\text{weighted} = W \cdot \text{diag}(\sqrt{F})
\tag{3.5}
$$

**Energy proxy.** We estimate $F_i = \mathbb{E}[x_i^2]$ from calibration 
activations. This is the "activation energy" per input dimension.

## 3.3 Energy-Based Rank Selection

Given singular values $\sigma_1 \geq \sigma_2 \geq \dots$, the cumulative 
energy at rank $r$ is:

$$
E(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(m, n)} \sigma_i^2}
\tag{3.6}
$$

**Energy thresholding.** Choose the minimal $r$ such that $E(r) \geq \varepsilon$. 
Typical $\varepsilon \in [0.90, 0.99]$.

**Trade-off.** Higher $\varepsilon$ gives more accurate reconstruction but 
less compression.

## 3.4 Frobenius Reconstruction Error

For validation:

$$
\text{err}(W, U, \Sigma, V) = \frac{\|W - U \Sigma V^\top\|_F}{\|W\|_F + \epsilon}
\tag{3.7}
$$

Threshold $\tau$: typically 0.10 (tight) to 0.50 (permissive). ASVD 
defaults to $\tau = 0.20$.

## 3.5 LoRA Recovery

LoRA (Hu et al., 2021) parameterizes weight updates as:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.8}
$$

with $A \in \mathbb{R}^{r_L \times n}$, $B \in \mathbb{R}^{m \times r_L}$, 
$B = 0$ at initialization.

**In ASVD.** The factorized layer with LoRA computes:

$$
y = (x V^\top \odot \Sigma) U^\top + (x A^\top) B^\top \cdot s + b
\tag{3.9}
$$

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $W$ | Weight matrix |
| $U, \Sigma, V$ | SVD factors |
| $r$ | Rank |
| $\varepsilon$ | Energy threshold |
| $\tau$ | Error threshold |
| $\sigma_i$ | $i$-th singular value |
| $F_i$ | Activation energy per dimension |
| $C_X$ | Input covariance |
| $r_L$ | LoRA rank |

---