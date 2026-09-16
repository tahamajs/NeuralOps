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

**Proof.** Classical. $\square$

## A.2 Energy Criterion Bound

**Proposition.** Let $\eta \in (0, 1)$ and $r_\text{eff}$ be the smallest 
rank with $E(r_\text{eff}) \geq \eta$. Then:

$$
\frac{\|W - W_{r_\text{eff}}\|_F}{\|W\|_F} \leq \sqrt{1 - \eta}
\tag{A.2}
$$

**Proof.** $\|W - W_r\|_F^2 = \|W\|_F^2(1 - E(r))$. $\square$

## A.3 Symmetric Factorization Equivalence

**Claim.** $U_r \Sigma_r^{1/2} \cdot \Sigma_r^{1/2} V_r^\top = U_r \Sigma_r V_r^\top$.

**Proof.** Direct. $\square$

## A.4 Parameter Count Analysis

| Configuration | Parameters |
|---|---|
| Original | $d_\text{in} \cdot d_\text{out}$ |
| Factorized at rank $r$ | $r(d_\text{in} + d_\text{out})$ |
| + LoRA($r_L$) | $(r + r_L)(d_\text{in} + d_\text{out})$ |

**Compression ratio for square layers:**

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

**Proof.** For each projection: extract $W$, compute SVD, truncate at 
$r_\text{eff}$, store $U_r$ in `up`, $V_r$ in `down`. Forward yields 
$x V_r^\top U_r^\top = x W_{r_\text{eff}}^\top$. $\square$

## A.7 Causality Preservation

**Claim.** MinimalSVD preserves causal masking.

**Proof.** Causal mask is applied after computing scores:

```python
if self.is_decoder and N > 1:
    causal_mask = torch.triu(...)
    scores = scores + causal_mask
```

Positions $j > i$ are set to $-\infty$ before softmax. This is independent 
of the attention implementation. $\square$

## A.8 Comparison to Companion Variants

The energy-adaptive rank selection used by MinimalSVD is **identical** to 
that used by MinimalNOVA and MinimalPRISM. The only differences are in the 
injection protocol:

| Variant | Injection protocol |
|---|---|
| MinimalSVD | Single-layer (caller handles traversal) |
| MinimalNOVA | Whole-model traversal + pre-test |
| MinimalPRISM | Whole-model + injection-time nested detection |

The core algorithm is identical. $\square$

## A.9 Numerical Stability

The symmetric factorization distributes $\Sigma^{1/2}$ evenly:

$$
\|U_r\|_2 \approx \|\Sigma_r^{1/2}\|_2 = \sqrt{\sigma_1}, \quad
\|V_r\|_2 \approx \sqrt{\sigma_1}
\tag{A.5}
$$

This avoids FP16 overflow when one factor dominates.

---