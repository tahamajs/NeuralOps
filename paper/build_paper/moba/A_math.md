# Appendix A. Mathematical Derivations

## A.1 Decay Parameterization

$$
\gamma_i = \sigma(-\exp(A_{\log, i}))
\tag{A.1}
$$

Since $\sigma(z) \in (0, 1)$:
- $\gamma_i \to 0$ as $A_{\log, i} \to +\infty$
- $\gamma_i \to 1$ as $A_{\log, i} \to -\infty$

## A.2 Causal Decay Weights

For query $i$ and key $j$, define:

$$
d_{ij} = \max(0, i - j)
\tag{A.2}
$$

$$
\ell_{ij}^{(d)} = d_{ij} \cdot \log \gamma_d
\tag{A.3}
$$

For $j \leq i$: $d_{ij} = i - j \geq 0$. Since $\log \gamma_d < 0$ 
(for $\gamma_d < 1$):

$$
\ell_{ij}^{(d)} = (i - j) \log \gamma_d \leq 0
\tag{A.4}
$$

with equality when $j = i$. This means $w_{ii}^{(d)} = 1$ (before 
normalization), and weights decay as $j$ moves away from $i$.

For $j > i$: we mask with $-\infty$ (causal).

## A.3 Softmax Over Distance

$$
w_{ij}^{(d)} = \frac{\exp(\ell_{ij}^{(d)})}{\sum_{j'} \exp(\ell_{ij'}^{(d)})}
\tag{A.5}
$$

**Claim.** $w_{ij}^{(d)} \propto \gamma_d^{i - j}$.

**Proof.** $\exp(\ell_{ij}^{(d)}) = \exp((i-j) \log \gamma_d) = \gamma_d^{i-j}$. 
$\square$

This is a geometric decay over distance.

## A.4 Bidirectional Mode

For encoders, we compute both directions:

$$
\text{Out} = 0.5 \cdot (W^\text{fwd} P_1 + W^\text{bwd} P_1)
\tag{A.6}
$$

where $W^\text{fwd}$ is lower-triangular and $W^\text{bwd}$ is upper-triangular.

## A.5 Preservation Gate

$$
\alpha = \sigma(g)
\tag{A.7}
$$

Final:

$$
\text{Final} = \alpha \cdot \text{Out} + (1 - \alpha) \cdot X
\tag{A.8}
$$

At init $g = 2.0$:

$$
\alpha = \sigma(2.0) \approx 0.8808
\tag{A.9}
$$

## A.6 Gated Fusion

$$
\text{Out} = \text{SiLU}(p_1) \odot p_2
\tag{A.10}
$$

where:

$$
\text{SiLU}(x) = x \cdot \sigma(x)
\tag{A.11}
$$

## A.7 Complexity Analysis

**Weight tensor:** $[L, L, d]$ → $\mathcal{O}(L^2 d)$.

**Time:** Softmax over $L \times L \times d$ is $\mathcal{O}(L^2 d)$.

**Total:** $\mathcal{O}(L^2 d)$.

**This is quadratic, not linear.**

## A.8 Causality Proof

**Claim.** For decoders, output at position $i$ depends only on inputs at 
$j \leq i$.

**Proof.** The causal mask sets $w_{ij} = 0$ for $j > i$. So:

$$
\text{Out}_i = \sum_{j \leq i} w_{ij} \cdot p_1[j]
\tag{A.12}
$$

Only inputs $j \leq i$ contribute. $\square$

---