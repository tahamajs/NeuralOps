<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Weighted Low-Rank Approximation

### A.1.1 Problem Statement

Given $W \in \mathbb{R}^{m \times n}$ and PSD weight matrix $C_X \in \mathbb{R}^{n \times n}$:

$$
W_r^* = \arg\min_{\text{rank}(W_r) = r} \mathrm{Tr}\!\left[(W - W_r) C_X (W - W_r)^\top\right]
\tag{A.1}
$$

### A.1.2 Transform

Since $C_X = C_X^{1/2} C_X^{1/2}$:

$$
\mathrm{Tr}[(W - W_r) C_X (W - W_r)^\top] = \|(W - W_r) C_X^{1/2}\|_F^2
\tag{A.2}
$$

Define $\tilde{W} = W C_X^{1/2}$, $\tilde{W}_r = W_r C_X^{1/2}$. Since 
$\text{rank}(\tilde{W}_r) \leq r$, the optimal $\tilde{W}_r$ is the rank-$r$ 
SVD truncation of $\tilde{W}$:

$$
\tilde{W}_r = U_r \Sigma_r V_r^\top
\tag{A.3}
$$

Therefore:

$$
W_r^* = U_r \Sigma_r V_r^\top C_X^{-1/2}
\tag{A.4}
$$

Setting $U = U_r \Sigma_r^{1/2}$, $V = \Sigma_r^{1/2} V_r^\top C_X^{-1/2}$ 
gives the AASVD factorization. $\square$

## A.2 Reconstruction Error Bound

**Proposition.** Under AASVD at rank $r$:

$$
\|W - U_r V_r\|_F \leq \frac{\sqrt{\sum_{i > r} \sigma_i^2}}{\sqrt{\lambda_\text{min}(C_X)}}
\tag{A.5}
$$

where $\sigma_i$ are the singular values of $W C_X^{1/2}$.

**Proof.** From the transformed problem:

$$
\|W C_X^{1/2} - U_r V_r C_X^{1/2}\|_F^2 = \sum_{i > r} \sigma_i^2
\tag{A.6}
$$

Meanwhile:

$$
\|W C_X^{1/2} - U_r V_r C_X^{1/2}\|_F^2 \geq \lambda_\text{min}(C_X) \|W - U_r V_r\|_F^2
\tag{A.7}
$$

since $C_X \succeq \lambda_\text{min}(C_X) I$. Combining:

$$
\|W - U_r V_r\|_F^2 \leq \frac{\sum_{i > r} \sigma_i^2}{\lambda_\text{min}(C_X)}
\tag{A.8}
$$

Taking square roots gives (A.5). $\square$

## A.3 EMA Convergence

**Proposition.** Under constant true covariance $\mathbb{E}[C_k] = \bar{C}$, 
the EMA estimate satisfies:

$$
\mathbb{E}[\hat{C}^{(k)}] - \bar{C} = \beta^{k-1} (C_1 - \bar{C})
\tag{A.9}
$$

**Proof.** The EMA recursion:

$$
\hat{C}^{(k)} = \beta \hat{C}^{(k-1)} + (1-\beta) C_k
$$

has closed form (unrolled):

$$
\hat{C}^{(k)} = \beta^{k-1} C_1 + (1-\beta) \sum_{j=2}^{k} \beta^{k-j} C_j
$$

Taking expectations:

$$
\mathbb{E}[\hat{C}^{(k)}] = \beta^{k-1} C_1 + (1-\beta) \sum_{j=2}^{k} \beta^{k-j} \bar{C}
$$

The sum $\sum_{j=2}^{k} \beta^{k-j} = \frac{1 - \beta^{k-1}}{1 - \beta}$. So:

$$
\mathbb{E}[\hat{C}^{(k)}] = \beta^{k-1} C_1 + (1 - \beta^{k-1}) \bar{C}
$$

Rearranging:

$$
\mathbb{E}[\hat{C}^{(k)}] - \bar{C} = \beta^{k-1} (C_1 - \bar{C}) \qquad \square
$$

## A.4 Eigendecomposition for Square Root

For $C_X = Q \Lambda Q^\top$ with $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_n)$:

$$
C_X^{1/2} = Q \mathrm{diag}(\sqrt{\lambda_1}, \dots, \sqrt{\lambda_n}) Q^\top
\tag{A.10}
$$

**Verification:**

$$
C_X^{1/2} \cdot C_X^{1/2} = Q \Lambda^{1/2} Q^\top Q \Lambda^{1/2} Q^\top = Q \Lambda Q^\top = C_X
\tag{A.11}
$$

Similarly for $C_X^{-1/2}$. $\square$

## A.5 Numerical Stability

Regularization threshold:

$$
\epsilon_\text{reg} = 10^{-6} \cdot \max_i \lambda_i + 10^{-8}
\tag{A.12}
$$

Clamping:

$$
\lambda_i \leftarrow \max(\lambda_i, \epsilon_\text{reg})
\tag{A.13}
$$

This ensures $\lambda_i \geq \epsilon_\text{reg} > 0$, so $1/\sqrt{\lambda_i}$ is bounded.
