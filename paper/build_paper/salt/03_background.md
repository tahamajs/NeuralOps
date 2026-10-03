<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Singular Value Decomposition

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^\top
\tag{3.1}
$$

where $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ are 
orthogonal, and $\Sigma$ is diagonal with $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD.**

$$
W_r = U_{:, :r} \Sigma_{:r} V_{:r, :}^\top
\tag{3.2}
$$

This is the best rank-$r$ approximation in Frobenius norm (Eckart-Young).

## 3.2 Low-Rank Adaptation (LoRA)

LoRA:

$$
W_\text{eff} = W + \frac{\alpha}{r} BA
\tag{3.3}
$$

with $A \in \mathbb{R}^{r \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r}$.

## 3.3 Singular Value Modulation

Given fixed $U, \Sigma, V$, one can modulate the singular values via a 
diagonal scale:

$$
W(\gamma) = U \text{diag}(\Sigma \odot \gamma) V^\top
\tag{3.4}
$$

with $\gamma \in \mathbb{R}^r$. Setting $\gamma = \mathbf{1}$ recovers the 
original.

**Effect.** Increasing $\gamma_i$ amplifies the $i$-th singular direction. 
Decreasing suppresses it.

## 3.4 Parameter Counting

For a weight $W \in \mathbb{R}^{m \times n}$ at rank $r$:

| Representation | Parameters |
|---|---|
| Full W | $mn$ |
| Truncated SVD ($U_r, \Sigma_r, V_r$) | $r(m + n + 1)$ |
| LoRA ($A, B$) | $r(m + n)$ |
| SALT ($U_r, \Sigma_r, V_r, \gamma, A, B$) | $2r(m + n) + 2r$ |

**Example (m = n = 768, r = 16):**

- Full: 589,824
- Truncated SVD: 24,592
- LoRA: 24,576
- SALT: 49,184 → **~12× reduction vs full**

## 3.5 Forward Pass Efficiency

At inference, computing $y = Wx$ requires:

- **Full W:** one matmul $Wx$ — cost $\mathcal{O}(mn)$
- **Low-rank W = UV:** two matmuls $Vx$ then $U(\cdot)$ — cost $\mathcal{O}(r(m+n))$
- **SALT (naive):** reconstruct W then one matmul — cost $\mathcal{O}(mn + r(m+n))$, **slower**
- **SALT (fused):** three matmuls — cost $\mathcal{O}(r(m+n))$

**Observation.** Realizing the inference benefit requires **fused kernels** 
or a forward pass that avoids materializing W.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $W$ | Weight matrix |
| $U, \Sigma, V$ | SVD factors |
| $r$ | Rank |
| $\gamma$ | Singular value scale |
| $A, B$ | Low-rank residual |
| $d_\text{in}, d_\text{out}$ | Input/output dims |
</parameter>

---