<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal proofs and derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Energy-Thresholded Rank Characterization

**Theorem.** Given singular values $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_d \geq 0$, 
define $\mathcal{E}(r) = \sum_{i=1}^r \sigma_i^2 / \sum_{j=1}^d \sigma_j^2$. For 
$\tau \in (0, 1)$, the minimum rank $r^*$ such that the relative Frobenius 
reconstruction error is $\leq \sqrt{1 - \tau}$ is:

$$
r^*(\tau) = \min\{r : \mathcal{E}(r) \geq \tau\}
\tag{A.1}
$$

**Proof.** By the Eckart–Young theorem:

$$
\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2 = (1 - \mathcal{E}(r)) \|W\|_F^2
\tag{A.2}
$$

So $\|W - W_r\|_F / \|W\|_F \leq \sqrt{1 - \tau}$ iff $1 - \mathcal{E}(r) \leq 1 - \tau$ 
iff $\mathcal{E}(r) \geq \tau$. The minimum such $r$ is $r^*(\tau)$. $\square$

## A.2 Symmetric Factorization Identity

**Proposition.** Let $U_r = U_{:, :r} \Sigma^{1/2}$ and $V_r = \Sigma^{1/2} V_{:r, :}^\top$. 
Then:

$$
U_r V_r = U_{:, :r} \Sigma V_{:r, :}^\top = W_r
\tag{A.3}
$$

**Proof.** Direct matrix multiplication:

$$
U_r V_r = U_{:, :r} \Sigma^{1/2} \cdot \Sigma^{1/2} V_{:r, :}^\top = U_{:, :r} \Sigma V_{:r, :}^\top \qquad \square
$$

## A.3 Factor Magnitude Balance

**Proposition.** $\|U_r\|_2 = \|V_r\|_2 = \sqrt{\sigma_1}$ where $\sigma_1$ is 
the largest singular value.

**Proof.** For symmetric SVD:

$$
U_r = U_{:, :r} \Sigma^{1/2} \Rightarrow U_r^\top U_r = \Sigma^{1/2} U_{:, :r}^\top U_{:, :r} \Sigma^{1/2} = \Sigma
\tag{A.4}
$$

So $\|U_r\|_2 = \sqrt{\|\Sigma\|_2} = \sqrt{\sigma_1}$.

Similarly for $V_r$. $\square$

This balanced magnitude is crucial for FP16 stability.

## A.4 Zero-Initialized Adapter Forward

**Claim.** With $B = 0$:

$$
\text{Proj}_P(x) = (x V_r^\top) U_r^\top + (x A^\top) \cdot 0 + b = W_r x + b
\tag{A.5}
$$

**Proof.** Substitute $B = 0$; the second term vanishes. $\square$

## A.5 Adapter Gradient at Initialization

**Claim.** At $B = 0$, $\partial \mathcal{L} / \partial A = 0$.

**Proof.** Forward: $y = f(U, V, x) + (x A^\top) B^\top$. So:

$$
\frac{\partial y}{\partial A} = \frac{\partial}{\partial A} \left[ (x A^\top) B^\top \right] = B x
\tag{A.6}
$$

At $B = 0$: $\partial y / \partial A = 0$. By chain rule, $\partial \mathcal{L} / \partial A = 0$. $\square$

**Consequence.** First gradient step updates only $B$. Standard LoRA property.

## A.6 Parameter Reduction Formula

For a square $d \times d$ projection:

$$
\text{Original params} = d^2
\tag{A.7}
$$

$$
\text{Compressed params} = r(d + d) + r_L(d + d) + d = 2d(r + r_L) + d
\tag{A.7}
$$

Reduction ratio:

$$
\rho = \frac{d^2}{2d(r + r_L) + d} \approx \frac{d}{2(r + r_L)}
\tag{A.8}
$$

At $d = 768$, $r = 128$, $r_L = 16$:

$$
\rho \approx \frac{768}{288} = 2.67
\tag{A.9}
$$

## A.7 Reconstruction Error Bound

For truncated SVD at rank $r^*$:

$$
\frac{\|W - W_{r^*}\|_F}{\|W\|_F} = \sqrt{1 - \mathcal{E}(r^*)} \leq \sqrt{1 - \tau}
\tag{A.10}
$$

At $\tau = 0.92$: bound = $\sqrt{0.08} \approx 0.283$.

## A.8 T5 Bias Preservation

The T5 relative attention bias is defined as:

$$
b(i, j) = \text{Embed}[\text{bucket}(j - i)]
\tag{A.11}
$$

This is a separate module that we preserve by copying the entire `nn.Embedding`. 
The factorized $Q, K, V$ projections do not affect the bias.

**Claim.** After injection, the T5 bias is added to attention scores exactly 
as before.

**Proof.** The `compute_bias` method is unchanged; the `relative_attention_bias` 
module is the same. $\square$

## A.9 Causality Preservation

**Claim.** For decoder models, output at position $i$ depends only on inputs 
at positions $\leq i$.

**Proof sketch.** Factorized $Q, K, V$ are computed from the same input as 
the original. RoPE preserves relative position. Causal masking applies 
$-\infty$ mask for $j > i$. Attention output is a weighted sum over valid 
positions. Adapters don't affect causality. $\square$

## A.10 Approximation Error vs Task Error

**Statement.** No formal bound on downstream task error is provided. The gap 
between reconstruction error (bounded above) and task accuracy is 
empirical. In practice, task accuracy drops correlate strongly with 
reconstruction error at $\tau \in [0.90, 0.95]$.
