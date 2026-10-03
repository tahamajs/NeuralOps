<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 SVD Decomposition

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^\top
\tag{3.1}
$$

with $U \in \mathbb{R}^{m \times m}$, $\Sigma \in \mathbb{R}^{m \times n}$, 
$V \in \mathbb{R}^{n \times n}$.

**Truncated:** $W_r = U_r \Sigma_r V_r^\top$. Error $\|W - W_r\|_F = \sqrt{\sum_{i > r} \sigma_i^2}$.

**Symmetric factorization:**

$$
W_r = (U_r \Sigma_r^{1/2})(\Sigma_r^{1/2} V_r^\top)
\tag{3.2}
$$

with $W_\text{up} = U_r \Sigma_r^{1/2} \in \mathbb{R}^{m \times r}$, 
$W_\text{down} = \Sigma_r^{1/2} V_r^\top \in \mathbb{R}^{r \times n}$.

**Parameter count:** $r(m + n)$ vs $mn$.

## 3.2 Energy-Based Rank Selection

Given singular values $\sigma_1 \geq \sigma_2 \geq \dots$, the energy 
captured by the top-$r$:

$$
E(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_i \sigma_i^2}
\tag{3.3}
$$

ORION selects the smallest $r$ such that $E(r) \geq \eta$ (energy threshold, 
default 0.95).

## 3.3 Random Fourier Features

Approximate $\exp(q^\top k)$ via:

$$
\exp(q^\top k) \approx \phi(q)^\top \phi(k), \quad
\phi(x) = \frac{1}{\sqrt{2F}} [\cos(\Omega x), \sin(\Omega x)]
\tag{3.4}
$$

with $\Omega \in \mathbb{R}^{d_h \times F}$.

## 3.4 Linear Attention with RFF

$$
O_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{3.5}
$$

Cost: $\mathcal{O}(L F d)$.

## 3.5 Softpick (Simplified)

ORION's softpick:

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{x_i} - 1)}{\sum_j \text{ReLU}(e^{x_j} - 1) + \epsilon}
\tag{3.6}
$$

**Key property:** For $x_i = -10^4$ (masked), $e^{x_i} - 1 \approx -1$, 
$\text{ReLU}(-1) = 0$. Masked positions contribute zero.

## 3.6 LoRA

Standard (Hu et al., 2021).

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $m, n$ | Matrix dimensions |
| $r$ | SVD rank |
| $\sigma_i$ | Singular values |
| $\eta$ | Energy threshold |
| $F$ | RFF features |
| $\Omega$ | RFF projection |
| $r_L$ | LoRA rank |

---