<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 SVD Optimality (Eckart–Young)

**Theorem.** For any $W \in \mathbb{R}^{m \times n}$ and $r \leq \min(m, n)$:

$$
W_r^* = \arg\min_{\text{rank}(W_r) \leq r} \|W - W_r\|_F = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top
\tag{A.1}
$$

**Proof.** Classical. The minimum Frobenius error is $\sqrt{\sum_{i > r} \sigma_i^2}$. 
$\square$

## A.2 Energy Criterion Bound

**Proposition.** Let $\eta \in (0, 1)$ and $r_\text{eff}$ be the smallest 
rank with $E(r_\text{eff}) \geq \eta$. Then:

$$
\frac{\|W - W_{r_\text{eff}}\|_F}{\|W\|_F} \leq \sqrt{1 - \eta}
\tag{A.2}
$$

**Proof.** $\|W - W_r\|_F^2 = \|W\|_F^2 (1 - E(r))$. Since $E(r_\text{eff}) \geq \eta$, 
the bound follows. $\square$

## A.3 Symmetric Factorization Equivalence

**Claim.** $U_r \Sigma_r^{1/2} \cdot \Sigma_r^{1/2} V_r^\top = U_r \Sigma_r V_r^\top$.

**Proof.** Direct. $\square$

## A.4 Parameter Count Analysis

| Configuration | Parameters |
|---|---|
| Original | $d_\text{in} \cdot d_\text{out}$ |
| Factorized at rank $r$ | $r(d_\text{in} + d_\text{out})$ |
| + LoRA($r_L$) | $(r + r_L)(d_\text{in} + d_\text{out})$ |

**Compression ratio for square layers ($d_\text{in} = d_\text{out} = d$):**

$$
\rho = \frac{d^2}{2rd} = \frac{d}{2r}
\tag{A.3}
$$

For $d = 768$:
- $r = 64$: $\rho = 6$
- $r = 128$: $\rho = 3$
- $r = 192$: $\rho = 2$

## A.5 GQA Dimension Handling

For GQA with $H_{kv} < H$:

- $W_Q \in \mathbb{R}^{d \times H d_h}$
- $W_K, W_V \in \mathbb{R}^{d \times H_{kv} d_h}$

Factorized parameter count:

$$
|W_Q| + |W_K| + |W_V| = r(d + H d_h) + r(d + H_{kv} d_h) + r(d + H_{kv} d_h)
\tag{A.4}
$$

## A.6 Weight Transfer Correctness

**Claim.** After `copy_weights`, the factorized module approximates the 
pretrained attention with rank-$r$ truncation error.

**Proof.** For each projection:
1. Extract $W$.
2. Compute SVD: $W = U \Sigma V^\top$.
3. Truncate at $r_\text{eff}$.
4. Store $U_r$ in `up`, $V_r$ in `down`.

Forward: $y = x V_r^\top U_r^\top = x W_{r_\text{eff}}^\top$. This is the 
rank-$r_\text{eff}$ approximation. $\square$

## A.7 Causality Preservation

**Claim.** MinimalPRISM preserves causal masking for decoder models.

**Proof.** The causal mask is applied after computing scores:

```python
if self.is_decoder and N > 1:
    causal_mask = torch.triu(...)
    scores = scores + causal_mask
```

Positions $j > i$ are set to $-\infty$ before softmax. This is independent 
of the attention implementation. $\square$

## A.8 Energy Bound with GQA

For GQA models with $H_{kv} < H$, the effective rank $r_\text{eff}$ may 
differ per projection. The bound (A.2) applies per projection.

## A.9 Numerical Stability

The symmetric factorization (4.6) distributes $\Sigma^{1/2}$ evenly, keeping 
both factors at similar magnitudes:

$$
\|U_r\|_2 \approx \|\Sigma_r^{1/2}\|_2 = \sqrt{\sigma_1}, \quad
\|V_r\|_2 \approx \sqrt{\sigma_1}
\tag{A.5}
$$

This avoids FP16 overflow when one factor dominates.

---