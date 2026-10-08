<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Softmax Attention

For $Q, K, V \in \mathbb{R}^{L \times d_h}$:

$$
A = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) \in \mathbb{R}^{L \times L}
\tag{3.1}
$$

$$
\text{Attn}(Q, K, V) = A V \in \mathbb{R}^{L \times d_h}
\tag{3.2}
$$

**Complexity.** $A$ requires $\mathcal{O}(L^2)$ memory; the two matmuls 
require $\mathcal{O}(L^2 d_h)$ time.

## 3.2 Nyström Approximation

For a symmetric PSD kernel matrix $K \in \mathbb{R}^{L \times L}$, the Nyström 
method selects $m$ landmark rows/columns and approximates:

$$
K \approx C W^{-1} C^\top
\tag{3.3}
$$

where:

- $C \in \mathbb{R}^{L \times m}$: query-landmark kernel
- $W \in \mathbb{R}^{m \times m}$: landmark-landmark kernel (PSD)

The approximation is exact when $K$ has rank $\le m$.

**Application to attention.** With $K_\text{attn} = \text{softmax}(QK^\top / \sqrt{d_h})$, 
the Nyströmform is:

$$
A \approx \tilde{A} = \begin{pmatrix} A_{Q, L_m} \\ A_{L_m, L_m} \\ A_{L_m, Q} \end{pmatrix}
\tag{3.4}
$$

where $L_m$ indexes $m$ landmarks.

## 3.3 Asymmetric Kernel

When the kernel is **not symmetric** — e.g., $K(q_i, k_j) \ne K(k_i, q_j)$ 
— Nyström approximation in the standard form does not apply.

**Observation.** The attention kernel $k(q, k) = \exp(q^\top k / \sqrt{d_h})$ 
**is** symmetric in $q, k$ for the *score* function, but the matrices $Q$ and 
$K$ themselves contain different information. The **row and column spaces** 
of $A$ may differ.

Primal Attention handles this by allowing $Q_\text{land} \ne K_\text{land}$.

## 3.4 Primal vs Dual Representation

Given a kernel matrix $K$ with rank $r$:

- **Primal representation:** $K = \Phi \Phi^\top$ where $\Phi \in \mathbb{R}^{L \times r}$
- **Dual representation:** $K = U \Sigma U^\top$ where $U \in \mathbb{R}^{L \times r}$ is orthogonal

Primal is cheaper when $r \ll L$ and $K$ is low-rank. Primal Attention uses 
the primal form with asymmetric factors.

## 3.5 Moore–Penrose Pseudoinverse

For a matrix $A \in \mathbb{R}^{m \times n}$, the Moore–Penrose pseudoinverse 
$A^+ \in \mathbb{R}^{n \times m}$ satisfies:

$$
A A^+ A = A, \qquad A^+ A A^+ = A^+, \qquad (A A^+)^\top = A A^+, \qquad (A^+ A)^\top = A^+ A
\tag{3.5}
$$

**Computation.** For $A = U \Sigma V^\top$, $A^+ = V \Sigma^+ U^\top$ where 
$\Sigma^+_{ii} = 1/\sigma_i$ if $\sigma_i > \epsilon$, else $0$.

## 3.6 Schulz Iteration

Schulz (1933) proposed the iteration:

$$
X_{t+1} = X_t (2I - A X_t)
\tag{3.6}
$$

which converges to $A^{-1}$ **quadratically** if $\|I - A X_0\| < 1$.

**Convergence.** Let $E_t = I - A X_t$. Then:

$$
E_{t+1} = (I - A X_{t+1}) = (I - A X_t (2I - A X_t)) = E_t^2
\tag{3.7}
$$

so $\|E_t\| \le \|E_0\|^{2^t}$, giving quadratic convergence.

**Initialization.** Common choices: $X_0 = A^\top / \|A\|_F^2$ or $X_0 = A^+$ 
(precomputed).

## 3.7 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable low-rank adapters:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A
\tag{3.8}
$$

We use LoRA to recover quality after attention replacement.

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $m$ | Landmark count |
| $d_h$ | Per-head dimension |
| $A$ | Attention matrix |
| $A^+$ | Pseudoinverse |
| $F_1, F_2, F_3$ | Primal factors |
| $X_t$ | $t$-th Schulz iterate |
| $r$ | LoRA rank |
| $\sigma$ | Softmax |

---
