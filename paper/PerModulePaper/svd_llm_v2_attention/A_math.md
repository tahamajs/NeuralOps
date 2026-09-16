<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 SVD Optimality (Eckart–Young)

**Theorem.** For any $W \in \mathbb{R}^{m \times n}$ and $r \leq \min(m, n)$:

$$
W_r^* = \arg\min_{\text{rank}(W_r) \leq r} \|W - W_r\|_F = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^H
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

**Proof.** $\|W - W_r\|_F^2 = \|W\|_F^2 (1 - E(r))$. $\square$

## A.3 Explicit vs Symmetric Equivalence

**Claim.** $U \Sigma V^H = (U \Sigma^{1/2}) (\Sigma^{1/2} V^H)$.

**Proof.** Direct. $\square$

## A.4 Parameter Count Analysis

For explicit storage:

| Configuration | Parameters |
|---|---|
| Original | $d_\text{in} \cdot d_\text{out}$ |
| Explicit $U, \Sigma, V^H$ | $r \cdot d_\text{out} + r + r \cdot d_\text{in}$ |
| + bias | $+ d_\text{out}$ |
| + LoRA | $+ r_L(d_\text{in} + d_\text{out})$ |

**Compression ratio (ignoring bias and Σ):**

$$
\rho \approx \frac{d^2}{2rd} = \frac{d}{2r}
\tag{A.3}
$$

## A.5 GQA Pooling Derivation

Given $W_K \in \mathbb{R}^{H d_h \times d}$ with $H > H_{kv}$, we 
mean-pool:

$$
W_K^\text{pooled}_{h, :} = \frac{1}{|G_h|} \sum_{g \in G_h} W_{g, :}
\tag{A.4}
$$

where $G_h$ partitions the $H$ source heads into $H_{kv}$ groups.

**Interpretation.** The pooled weight retains the average behavior of each 
head group, which is the natural reduction under the GQA assumption.

## A.6 Weight Transfer Correctness

**Claim.** After `copy_weights`, the factorized module approximates the 
pretrained attention with rank-$r$ truncation error.

**Proof.** For each projection, extract $W$, compute SVD, truncate at 
$r_\text{eff}$, store $U_r$, $\Sigma_r$, $V_r^H$. Forward yields 
$x (U_r \Sigma_r V_r^H)^\top = x (V_r^{H\top} \Sigma_r U_r^\top)$. $\square$

## A.7 Causality Preservation

**Claim.** SVD-LLM V2 preserves causal masking.

**Proof.** Causal mask applied after scores:

```python
if self.is_decoder and seq_len > 1:
    causal_mask = torch.triu(...)
    scores = scores + causal_mask
```

Positions $j > i$ set to $-\infty$ before softmax. $\square$

## A.8 Numerical Stability of Explicit Storage

Explicit storage keeps $\Sigma$ separate. In FP16:

- If $\sigma_\text{min} < 6 \times 10^{-8}$ and $\sigma_\text{max} > 65504$, 
  overflow/underflow may occur.
- Symmetric storage keeps both factors at $\sqrt{\sigma}$, avoiding 
  extremes.

**Recommendation.** For deployments restricted to FP16, use symmetric 
storage. Otherwise, explicit is preferred for interpretability.

## A.9 Complexity Analysis

For $N_l$ layers, each with four projections:

| Phase | Time | Memory |
|---|---|---|
| Extraction | $\mathcal{O}(N_l d^2)$ | $\mathcal{O}(d^2)$ |
| SVD (per proj) | $\mathcal{O}(d^3)$ | $\mathcal{O}(d^2)$ |
| Truncation | $\mathcal{O}(N_l r d^2)$ | $\mathcal{O}(d^2)$ |
| Total | $\mathcal{O}(N_l d^3)$ | $\mathcal{O}(d^2)$ |

---