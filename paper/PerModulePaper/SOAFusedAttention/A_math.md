# Appendix A. Mathematical Derivations

## A.1 Truncated SVD

For $W \in \mathbb{R}^{m \times n}$, the SVD is:

$$
W = U \Sigma V^\top
\tag{A.1}
$$

The best rank-$k$ approximation in Frobenius norm (Eckart-Young):

$$
W_k = U_{:, :k} \Sigma_{:k, :k} V_{:k, :}^\top
\tag{A.2}
$$

with error:

$$
\|W - W_k\|_F = \sqrt{\sum_{i > k} \sigma_i^2}
\tag{A.3}
$$

## A.2 Factorization

We split (A.2) into two factors:

$$
\text{down} = V_{:k, :}^\top \in \mathbb{R}^{k \times n}
\tag{A.4}
$$

$$
\text{up} = U_{:, :k} \Sigma_{:k, :k} \in \mathbb{R}^{m \times k}
\tag{A.5}
$$

Then:

$$
\text{up} \cdot \text{down} = U_{:, :k} \Sigma_{:k, :k} V_{:k, :}^\top = W_k
\tag{A.6}
$$

## A.3 Parameter Count

For each factorized projection:

$$
\text{params} = k \cdot n + m \cdot k = k(m + n)
\tag{A.7}
$$

vs baseline $m \cdot n$. Reduction factor:

$$
\frac{m \cdot n}{k(m + n)}
\tag{A.8}
$$

For square $m = n = d$:

$$
\frac{d^2}{2kd} = \frac{d}{2k}
\tag{A.9}
$$

## A.4 LoRA Composition

With LoRA attached to the up-projection:

$$
\text{out} = (W_{up} + \Delta W_{up}) \cdot \text{down}(x)
\tag{A.10}
$$

where:

$$
\Delta W_{up} = \frac{\alpha}{r_L} B A
\tag{A.11}
$$

So the effective weight is:

$$
W_\text{eff} = (W_{up} + \frac{\alpha}{r_L} B A) \cdot W_\text{down}
\tag{A.12}
$$

## A.5 GQA Parameter Reduction

For GQA with $H_{kv} < H$:

$$
\text{K params baseline} = d \cdot (H_{kv} \cdot d_h) = d^2 \cdot (H_{kv}/H)
\tag{A.13}
$$

$$
\text{K params ours} = k(d + H_{kv} d_h)
\tag{A.14}
$$

For $H_{kv} = H/4$ and $k = d/6$:

$$
\text{reduction} = \frac{d^2 (H_{kv}/H)}{k(d + H_{kv} d_h)} \approx \frac{d}{k} \cdot \frac{1}{1 + H_{kv} d_h / d}
\tag{A.15}
$$

## A.6 Effect of Rank on Error

The Frobenius error after rank-$k$ truncation is:

$$
\text{err}(k) = \sqrt{\sum_{i > k} \sigma_i^2} / \|W\|_F
\tag{A.16}
$$

For transformer attention weights, this decays rapidly with $k$.

## A.7 Symmetric vs Asymmetric Split

Our implementation uses asymmetric split ($\Sigma$ entirely in the up-projection):

$$
\text{down} = V_k^\top, \quad \text{up} = U_k \Sigma_k
\tag{A.17}
$$

An alternative symmetric split:

$$
\text{down} = \Sigma_k^{1/2} V_k^\top, \quad \text{up} = U_k \Sigma_k^{1/2}
\tag{A.18}
$$

Both give the same $UV$, but asymmetric has:
- Slightly better numerical behavior in FP16.
- All scaling concentrated in one factor.

---
