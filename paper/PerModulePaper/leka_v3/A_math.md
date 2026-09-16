# Appendix A. Mathematical Derivations

## A.1 Kernel Feature Map

For $x \in \mathbb{R}^{d_h}$:

$$
\phi(x) = \frac{1}{\sqrt{d_\text{feat}}} \begin{pmatrix} \cos(W_1 x) \\ \sin(W_1 x) \end{pmatrix}
\tag{A.1}
$$

with $W_1 \in \mathbb{R}^{d_h \times d_\text{feat}/2}$.

**Kernel approximation:**

$$
\phi(x)^\top \phi(y) = \frac{1}{d_\text{feat}} \sum_{k=1}^{d_\text{feat}/2} [\cos(w_k \cdot x) \cos(w_k \cdot y) + \sin(w_k \cdot x) \sin(w_k \cdot y)]
\tag{A.2}
$$

Using $\cos(a)\cos(b) + \sin(a)\sin(b) = \cos(a - b)$:

$$
\phi(x)^\top \phi(y) = \frac{1}{d_\text{feat}} \sum_k \cos(w_k \cdot (x - y))
\tag{A.3}
$$

By Bochner's theorem, this approximates $e^{-\|x-y\|^2/2}$, the Gaussian kernel.

## A.2 Landmark Attention

For query $q_i$ and landmark set $\mathcal{L}$:

$$
\text{Out}_\text{land}[i] = \frac{\sum_{j \in \mathcal{L}} \phi(q_i)^\top \phi(k_j) \cdot V_j}{\sum_{j \in \mathcal{L}} \phi(q_i)^\top \phi(k_j)}
\tag{A.4}
$$

Define:

$$
S = \sum_{j \in \mathcal{L}} \phi(k_j) V_j^\top, \quad Z = \sum_{j \in \mathcal{L}} \phi(k_j)
\tag{A.5}
$$

Then:

$$
\text{Out}_\text{land}[i] = \frac{\phi(q_i)^\top S}{\phi(q_i)^\top Z}
\tag{A.6}
$$

**Complexity:** $S$ and $Z$ cost $\mathcal{O}(|\mathcal{L}| \cdot d_\text{feat} \cdot d_h)$ 
once; then per query is $\mathcal{O}(d_\text{feat} \cdot d_h)$.

## A.3 Causal Mask Correctness

**Claim.** Query at position $i$ attends only to landmark positions $\leq i$.

**Proof.** Mask sets $\text{mask}_{ij} = -\infty$ when $\mathcal{L}_j > i$. 
Softmax assigns zero weight. $\square$

## A.4 Landmark Scorer Bound

**Claim.** The scorer cost is $\mathcal{O}(L \cdot d_\text{feat} \cdot 64)$ 
for a 2-layer MLP.

**Proof.** First layer: $\mathbb{R}^{2 d_\text{feat}} \to \mathbb{R}^{64}$: 
$2 d_\text{feat} \cdot 64$ FLOPs. Second layer: $64 \to 1$: 64 FLOPs. Total 
per token: $128 d_\text{feat} + 64$. For $L$ tokens: $\mathcal{O}(L d_\text{feat})$. $\square$

## A.5 W_corr Analysis

**Purpose.** $W_\text{corr}$ lets each head mix landmark values before 
attention:

$$
V_\mathcal{L}^\text{adj} = V_\mathcal{L} W_\text{corr}
\tag{A.7}
$$

**Effect.** If $W_\text{corr} = I$, no mixing. If learned, it can project 
landmark values onto important subspaces.

## A.6 Complexity Summary

| Operation | Complexity |
|---|---|
| Q/K/V projections | $\mathcal{O}(L d^2)$ |
| Feature map | $\mathcal{O}(L d_h d_\text{feat})$ |
| Scorer | $\mathcal{O}(L d_\text{feat})$ |
| TopK | $\mathcal{O}(L \log L_\text{land})$ |
| Landmark attention | $\mathcal{O}(L L_\text{land} d_h)$ |
| Local attention | $\mathcal{O}(L W d_h)$ |
| **Total** | $\mathcal{O}(L(L_\text{land} + W) d_h)$ |

For $L_\text{land}, W \sim \mathcal{O}(1)$: **linear in $L$**.

---