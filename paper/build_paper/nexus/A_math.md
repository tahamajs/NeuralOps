# Appendix A. Mathematical Derivations

## A.1 Block Partitioning

Given sequence length $N$ and block size $B_s$, the number of blocks is:

$$
M = \lceil N / B_s \rceil
\tag{A.1}
$$

Token $t$ is in block $b_t = \lfloor t / B_s \rfloor$.

## A.2 XAttention Scoring

For stride $\sigma$:

$$
\tilde{Q} = Q[:, :, ::\sigma, :]
\tag{A.2}
$$

Block score:

$$
S_{b_i, b_j} = \text{mean}(\tilde{Q}_{[b_i]} \tilde{K}_{[b_j]}^\top)
\tag{A.3}
$$

## A.3 Softpick

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{x_i/T} - 1)}{\sum_j \text{ReLU}(e^{x_j/T} - 1)}
\tag{A.4}
$$

**Claim.** For $x_i < 0$, output is 0.

**Proof.** $e^{x_i} < 1 \Rightarrow e^{x_i} - 1 < 0 \Rightarrow \text{ReLU} = 0$. $\square$

**Concern.** If all $x_j < 0$: denominators 0, sum 0, division by $10^{-8}$ → 
near-zero output.

## A.4 Stripe Mask

Token $t$ is in stripe group $s_t = \lfloor t / S_w \rfloor$.

Mask:

$$
\text{mask}_{ij} = \mathbb{1}[|s_i - s_j| \leq 1]
\tag{A.5}
$$

## A.5 RFF Rectification

$$
\phi(x) = \frac{1}{\sqrt{F}} \exp(W_\text{rff} \hat{x})
\tag{A.6}
$$

where $\hat{x} = x / \|x\|_2$.

Linear attention:

$$
\text{Out}_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{A.7}
$$

## A.6 Complexity

Encoder path:
- XAttention: $\mathcal{O}((N/\sigma)^2 d_h)$
- Block attention: $\mathcal{O}(N \cdot k \cdot B_s \cdot d_h)$
- Total: sub-quadratic.

Causal path: $\mathcal{O}(N^2 d_h)$ (dense).

---