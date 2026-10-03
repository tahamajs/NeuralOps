<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 SVD Optimality (Eckart–Young)

**Theorem.** For any $W \in \mathbb{R}^{m \times n}$ and any $r \leq \min(m, n)$:

$$
W_r^* = \arg\min_{\text{rank}(W_r) \leq r} \|W - W_r\|_F = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top
\tag{A.1}
$$

**Proof.** This is classical. The minimum Frobenius error is 
$\sqrt{\sum_{i > r} \sigma_i^2}$. Any rank-$r$ matrix can be written 
$W_r = A B$ with $A \in \mathbb{R}^{m \times r}$, $B \in \mathbb{R}^{r \times n}$. 
The optimal solution is given by SVD truncation. $\square$

## A.2 Energy Criterion Bound

**Proposition.** Let $\eta \in (0, 1)$ and $r_\text{eff}$ be the smallest 
rank such that $E(r_\text{eff}) \geq \eta$. Then:

$$
\frac{\|W - W_{r_\text{eff}}\|_F}{\|W\|_F} \leq \sqrt{1 - \eta}
\tag{A.2}
$$

**Proof.** $\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2 = \|W\|_F^2 - \sum_{i \leq r} \sigma_i^2$. 
Dividing by $\|W\|_F^2$:

$$
\frac{\|W - W_r\|_F^2}{\|W\|_F^2} = 1 - E(r)
\tag{A.3}
$$

Since $E(r_\text{eff}) \geq \eta$:

$$
\frac{\|W - W_{r_\text{eff}}\|_F^2}{\|W\|_F^2} \leq 1 - \eta
\tag{A.4}
$$

Taking square roots gives (A.2). $\square$

## A.3 Symmetric Factorization Equivalence

**Claim.** $U_r \Sigma_r^{1/2} \cdot \Sigma_r^{1/2} V_r^\top = U_r \Sigma_r V_r^\top$.

**Proof.** Direct computation. $\square$

## A.4 Parameter Count Analysis

For a linear layer with $d_\text{in}$ inputs and $d_\text{out}$ outputs:

| Configuration | Parameters |
|---|---|
| Original `Linear` | $d_\text{in} \cdot d_\text{out}$ |
| `FactorisedLinear` at rank $r$ | $r \cdot d_\text{in} + r \cdot d_\text{out} = r(d_\text{in} + d_\text{out})$ |
| + LoRA($r_L$) | $(r + r_L)(d_\text{in} + d_\text{out})$ |

**Compression ratio:**

$$
\rho = \frac{d_\text{in} \cdot d_\text{out}}{r(d_\text{in} + d_\text{out})}
\tag{A.5}
$$

For square layers ($d_\text{in} = d_\text{out} = d$):

$$
\rho = \frac{d^2}{2rd} = \frac{d}{2r}
\tag{A.6}
$$

## A.5 GQA Dimension Handling

For GQA with $H_{kv} < H$ query heads:

- Query projection: $\mathbb{R}^{d \times (H \cdot d_h)}$
- Key projection: $\mathbb{R}^{d \times (H_{kv} \cdot d_h)}$
- Value projection: $\mathbb{R}^{d \times (H_{kv} \cdot d_h)}$

Factorized parameter count:

$$
|W_Q|_F + |W_K|_F + |W_V|_F = r(d + H d_h) + r(d + H_{kv} d_h) + r(d + H_{kv} d_h)
\tag{A.7}
$$

## A.6 Weight Transfer Correctness

**Claim.** After `copy_weights`, the factorized module approximates the 
pretrained attention with rank-$r$ truncation error.

**Proof.** For each projection:
1. Extract $W$ (architecture-dependent).
2. Compute SVD: $W = U \Sigma V^\top$.
3. Truncate at $r_\text{eff}$: $W_{r_\text{eff}} = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top$.
4. Store $U_r = U_{:, :r} \Sigma^{1/2}$ in `up`, $V_r = \Sigma^{1/2} V^\top$ in `down`.

Forward: $y = x V_r^\top U_r^\top = x (U_r V_r)^\top = x W_{r_\text{eff}}^\top$. 
This is the rank-$r_\text{eff}$ approximation of the original projection. $\square$

## A.7 Causality Preservation

**Claim.** MinimalNOVA preserves causal masking for decoder models.

**Proof.** The causal mask is applied after computing scores:

```python
if self.is_decoder and N > 1:
    causal_mask = torch.triu(...)
    scores = scores + causal_mask
```

Positions $j > i$ are set to $-\infty$ before softmax, so attention at 
position $i$ depends only on positions $\leq i$. This does not depend on the 
attention implementation. $\square$

---