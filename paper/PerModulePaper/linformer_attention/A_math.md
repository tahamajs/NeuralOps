# Appendix A. Mathematical Derivations

## A.1 Linformer Projection

Given $K \in \mathbb{R}^{L \times d_h}$ and $E \in \mathbb{R}^{k \times L}$:

$$
\tilde{K} = E K \in \mathbb{R}^{k \times d_h}
\tag{A.1}
$$

Similarly $\tilde{V} = F V$.

## A.2 Attention

$$
\text{scores} = \frac{Q \tilde{K}^\top}{\sqrt{d_h}} \in \mathbb{R}^{L \times k}
\tag{A.2}
$$

$$
\text{attn} = \text{softmax}(\text{scores})
\tag{A.3}
$$

$$
\text{out} = \text{attn} \tilde{V} \in \mathbb{R}^{L \times d_h}
\tag{A.4}
$$

## A.3 Complexity

| Operation | FLOPs |
|---|---|
| $E K$ | $k L d_h$ |
| $Q \tilde{K}^\top$ | $L k d_h$ |
| Softmax | $L k$ |
| $\text{attn} \tilde{V}$ | $L k d_h$ |
| **Total** | $\mathcal{O}(L k d_h)$ |

## A.4 Low-Rank Approximation Error

If $P = \text{softmax}(QK^\top/\sqrt{d_h})$ has singular values $\sigma_1 \geq \sigma_2 \geq \dots$:

$$
\|P - P_k\|_F = \sqrt{\sum_{i > k} \sigma_i^2}
\tag{A.5}
$$

For fixed $k$, this is the minimum approximation error over rank-$k$ matrices.

## A.5 Diagonal Init

With $E = [I; \epsilon]$ (identity block + noise):

$$
E K = \begin{pmatrix} K_{1:\min(k,L)} \\ \epsilon K \end{pmatrix}
\tag{A.6}
$$

For $k \geq L$, the top block is exactly $K$, so $\tilde{K} = K$ (exact).

## A.6 Causal Fallback Proof

**Claim.** For $L > k$, the fallback preserves causality.

**Proof.** Fallback uses dense attention with triangular mask. Standard 
causal attention is causally correct. $\square$

## A.7 Interpolation for Long Sequences

For $L > L_\text{max}$, we interpolate:

$$
E_\text{ext}[i] = \text{interp}(E[i], L)
\tag{A.7}
$$

using linear interpolation. This is a heuristic — the interpolation does 
not preserve the identity-init property.

---