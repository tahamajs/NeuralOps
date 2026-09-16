<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Local Window Complexity

**Claim.** Sliding-window attention with window $w$ has $\mathcal{O}(L w d_h)$ 
time and $\mathcal{O}(L w)$ memory.

**Proof.** For each query, $2w + 1$ keys are attended (or $w + 1$ for causal). 
The score matrix has shape $(L, 2w+1)$, requiring $\mathcal{O}(L w)$ memory. 
The matmuls require $\mathcal{O}(L w d_h)$ time. $\square$

## A.2 Chunked RFF Causality

**Claim.** The chunked prefix scan computes exact causal linear attention.

**Proof.** For chunk $[a, b)$:

$$
O_i = \frac{\sum_{j \le i} \phi(Q_i)^\top \phi(K_j) V_j}{\sum_{j \le i} \phi(Q_i)^\top \phi(K_j)}
\tag{A.1}
$$

Split the sum into $j \in [a, i]$ (intra) and $j \in [0, a)$ (inter):

$$
O_i = \frac{\phi(Q_i)^\top \left(\sum_{j \in [a, i]} \phi(K_j) V_j + \sum_{j < a} \phi(K_j) V_j\right)}{\phi(Q_i)^\top \left(\sum_{j \in [a, i]} \phi(K_j) + \sum_{j < a} \phi(K_j)\right)}
\tag{A.2}
$$

The inter-chunk terms are exactly $S$ and $Z$ carried from previous chunks. 
The intra-chunk terms are computed within the chunk. $\square$

## A.3 Orthogonal Projection Variance

**Claim.** Orthogonal $\Omega$ reduces the variance of $\phi(q)^\top \phi(k)$ 
compared to i.i.d. Gaussian.

**Sketch.** The variance of a Monte Carlo estimator decreases when samples 
are orthogonal rather than independent. This is the standard PORF argument 
(Choromanski et al., 2021). $\square$

## A.4 Gate Initialization

Given $g = \sigma(w^\top x + b)$ with $w = 0$ and $b = 3$:

$$
g_i = \sigma(3) = \frac{1}{1 + e^{-3}} \approx 0.9526
\tag{A.3}
$$

At initialization, the model is a **convex combination with weight 0.95 on 
local and 0.05 on global**. The global stream contributes minimally at start.

## A.5 Non-Negativity of Feature Map

$\phi(x) = \mathrm{elu}(\Omega^\top x) + 1$:

- $\mathrm{elu}(y) \geq -1$ for all $y$ (with $\alpha = 1$).
- Therefore $\phi(x) \geq 0$.

This ensures the denominator in the RFF attention is non-negative.

## A.6 Fusion Error Bound

**Claim.** For gate $g$, the fusion error relative to local is:

$$
\|O - O^\text{local}\| \leq (1 - g) \cdot \|O^\text{global} - O^\text{local}\|
\tag{A.4}
$$

**Proof.** $O - O^\text{local} = (1 - g)(O^\text{global} - O^\text{local})$. 
Taking norms gives (A.4). $\square$

## A.7 Cumsum Memory

For the cumulative state $S \in \mathbb{R}^{B \times H \times m \times d_h}$:

$$
\text{Memory} = B H m d_h \cdot \text{bytes}
\tag{A.5}
$$

For $B=8, H=12, m=128, d_h=64$, FP32: $8 \times 12 \times 128 \times 64 \times 4 = 3.1$ MB per layer. **Practical.**

---