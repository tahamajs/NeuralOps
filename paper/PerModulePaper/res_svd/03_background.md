# 3. Background

## 3.1 Singular Value Decomposition

For $W \in \mathbb{R}^{m \times n}$:
$$
W = U \Sigma V^\top
\tag{3.1}
$$
with $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD:** $W_r = U_{:, :r} \Sigma_{:r} V_{:r, :}^\top$, error 
$\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2$.

## 3.2 LoRA

LoRA (Hu et al., 2021) parameterizes weight updates as:
$$
\Delta W = \frac{\alpha}{r} B A, \quad A \in \mathbb{R}^{r \times d_\text{in}}, \quad B \in \mathbb{R}^{d_\text{out} \times r}
\tag{3.2}
$$
with $B = 0$ at init, so $\Delta W = 0$.

**Parameter count:** $r(d_\text{in} + d_\text{out})$.

## 3.3 Spectral Initialization

Standard LoRA: $B = 0$. This makes the initial model identical to the 
base model but with no residual capacity.

**PiSSA (Meng et al., 2024):** $B, A$ are initialized from the **top** 
singular directions of $W$, and the base is set to the residual:
$$
W = W_\text{residual} + B A \quad \text{where} \quad B A = U_{:, :r} \Sigma_{:r} V_{:r, :}^\top
\tag{3.3}
$$

**Res-SVD:** the **complement** — base is the top singular directions, 
residual is the next ones:
$$
W = W_b + R \quad \text{where} \quad W_b = U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top, \quad R = U_{:, r_b:r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top
\tag{3.4}
$$

## 3.4 Two-Level Decomposition

The key observation: $W_b + R$ with $r_b + r_r$ components is the same as 
$W_{r_b + r_r}$ (SVD of rank $r_b + r_r$), because SVD is additive on 
orthogonal directions.

**Proof sketch.** $U$ has orthogonal columns, so:
$$
U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top + U_{:, r_b:r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top = U_{:, :r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top
\tag{3.5}
$$

## 3.5 Notation Summary

| Symbol | Meaning |
|---|---|
| $W$ | Full weight |
| $W_b$ | Base SVD approximation |
| $R$ | Residual $W - W_b$ |
| $r_b, r_r$ | Base, residual ranks |
| $U_b, V_b$ | Base SVD factors |
| $B, A$ | Residual LoRA factors |
| $\alpha$ | Residual scaling |

---