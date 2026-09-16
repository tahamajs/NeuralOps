# Appendix A. Mathematical Derivations

## A.1 Energy-Based Rank Selection

For a weight matrix $W$ with SVD $W = U \Sigma V^\top$, the energy captured 
by rank $r$ is:

$$
E(r) = \frac{\sum_{i \leq r} \sigma_i^2}{\sum_{i} \sigma_i^2}
\tag{A.1}
$$

We select $r^* = \min \{ r : E(r) \geq \eta \}$.

**Claim.** The Frobenius error after rank-$r^*$ truncation is bounded by:

$$
\|W - W_{r^*}\|_F \leq \sqrt{1 - \eta} \cdot \|W\|_F
\tag{A.2}
$$

**Proof.** By definition of $E(r^*)$:

$$
\sum_{i > r^*} \sigma_i^2 = (1 - E(r^*)) \|W\|_F^2 \leq (1 - \eta) \|W\|_F^2
\tag{A.3}
$$

Taking square roots gives (A.2). $\square$

## A.2 Split-Head Complexity

For an encoder with $H_\text{local}$ local heads and $H_\text{global}$ 
global heads:

$$
\text{FLOPs} = B \cdot L \cdot d_h \cdot (H_\text{local} \cdot W + H_\text{global} \cdot F)
\tag{A.4}
$$

For fixed $W, F, H_\text{local}, H_\text{global}$, this is linear in $L$.

## A.3 Local Window Attention

For query at position $i$, the window mask is:

$$
\text{mask}_{ij} = \mathbb{1}[(i - W/2) \leq j \leq (i + W/2)]
\tag{A.5}
$$

Causal variant:

$$
\text{mask}_{ij}^\text{causal} = \mathbb{1}[i - W/2 \leq j \leq i]
\tag{A.6}
$$

## A.4 Kernel Linear Attention

RFF feature map:

$$
\phi(x) = \frac{1}{\sqrt{F}} \exp(W_\text{rff} x)
\tag{A.7}
$$

Attention output:

$$
\text{Out}_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{A.8}
$$

**Zero-sum keys.** Subtract mean:

$$
\tilde{k}_j = k_j - \bar{k}
\tag{A.9}
$$

Then $\phi(\tilde{k}_j)$ is centered around 1.

## A.5 Softpick Fix

The V5 stable softmax:

$$
\text{softpick}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
\tag{A.10}
$$

This is always numerically stable (uses `F.softmax` which subtracts max).

The V4 (broken) version:

$$
\text{softpick}_\text{V4}(x)_i = \text{ReLU}\!\left(e^{(x_i - m)/T} - e^{-m/T}\right)
\tag{A.11}
$$

**Failure case.** If $m < 0$, then $e^{-m/T} > 1$, and 
$\text{ReLU}(e^{(x_i - m)/T} - e^{-m/T}) = 0$ for all $i$ (since 
$x_i \leq m$).

## A.6 Causality Preservation

**Claim.** For decoders, AEGIS-V5 preserves causality via the standard 
causal mask.

**Proof.** The decoder path uses $Q K^\top$ with a triangular mask 
$\text{mask}_{ij} = \mathbb{1}[j \leq i]$. Softmax assigns zero weight to 
$-\infty$ entries. Thus, output at position $i$ depends only on inputs 
$j \leq i$. $\square$

---