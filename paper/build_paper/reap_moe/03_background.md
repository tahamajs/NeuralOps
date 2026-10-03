# 3. Background

## 3.1 Mixture-of-Experts

A standard MoE layer computes:

$$
y = \sum_{i=1}^{E} g_i(x) \cdot E_i(x)
\tag{3.1}
$$

where:
- $E_i$ is the $i$-th expert (a small neural network).
- $g_i(x)$ is the gating weight, with $\sum_i g_i(x) = 1$.

**Sparse MoE.** To save compute, only the top-$k$ experts are activated:

$$
y = \sum_{i \in \text{Top-}k(g(x))} \tilde{g}_i(x) \cdot E_i(x)
\tag{3.2}
$$

where $\tilde{g}$ is $g$ restricted to the top-$k$ and renormalized.

## 3.2 Router Design

**Linear router.** The gate is typically:

$$
g(x) = \text{softmax}(W_r \cdot \text{LN}(x))
\tag{3.3}
$$

where $W_r \in \mathbb{R}^{E \times d}$.

**Noisy gating.** Shazeer et al. (2017) add Gaussian noise during training:

$$
g(x) = \text{softmax}(W_r x + \epsilon), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
\tag{3.4}
$$

This provides exploration and prevents premature router collapse.

**Top-$k$ selection.** From $g(x) \in \mathbb{R}^E$, select the top-$k$ experts 
and renormalize.

## 3.3 Load-Balancing Loss

Without regularization, the router tends to concentrate on a few experts. 
The standard fix is the auxiliary loss:

$$
\mathcal{L}_\text{aux} = E \cdot \sum_{i=1}^{E} \rho_i \cdot \pi_i
\tag{3.5}
$$

where:
- $\rho_i = \frac{1}{N} \sum_{j=1}^{N} \mathbb{1}[i \in \text{Top-}k(g(x_j))]$ is 
  the fraction of tokens routed to expert $i$.
- $\pi_i = \frac{1}{N} \sum_{j=1}^{N} g_i(x_j)$ is the mean gate probability.

**Interpretation.** $\mathcal{L}_\text{aux} = 1$ when both $\rho$ and $\pi$ are 
uniform. It is larger when either is concentrated.

## 3.4 SVD Factorization

For a weight matrix $W \in \mathbb{R}^{m \times n}$, the rank-$r$ SVD 
approximation is:

$$
W \approx U_r \Sigma_r V_r^\top
\tag{3.6}
$$

with $U_r \in \mathbb{R}^{m \times r}$, $\Sigma_r \in \mathbb{R}^{r \times r}$, 
$V_r \in \mathbb{R}^{n \times r}$. Parameter count decreases from $mn$ to 
$r(m + n)$.

**In REAP-MoE.** Each expert is a two-layer MLP. Both layers are SVD-factorized 
with rank $r = \min(d_\text{in}, d_\text{out}) / 4$.

## 3.5 Transformer FFN Structure

Standard transformer FFN:

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x)
\tag{3.7}
$$

with $W_1 \in \mathbb{R}^{4d \times d}$ and $W_2 \in \mathbb{R}^{d \times 4d}$. 
Parameter count: $8d^2$.

REAP-MoE replaces this with $E$ experts, each using SVD:

$$
\text{FFN}_\text{MoE}(x) = \sum_{i \in \text{Top-}k} g_i \cdot E_i(x)
\tag{3.8}
$$

where each $E_i$ is an SVD-factorized two-layer MLP.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $E$ | Number of experts |
| $k$ | Top-$k$ |
| $d$ | Hidden dim |
| $W_r$ | Router weight |
| $g_i(x)$ | Gate weight for expert $i$ |
| $\rho_i$ | Expert utilization |
| $\pi_i$ | Mean gate probability |
| $\mathcal{L}_\text{aux}$ | Auxiliary loss |
| $r$ | SVD rank |
| $\sigma$ | Noise std |

---