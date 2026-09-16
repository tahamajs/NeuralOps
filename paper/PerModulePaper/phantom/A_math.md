<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 CUR Decomposition Bound

**Theorem (Drineas et al., 2008).** Let $W \in \mathbb{R}^{m \times n}$ with 
rank $r^*$. Sample $r$ columns with probability proportional to leverage 
scores $\pi_j$ and $r$ rows with probability proportional to $\rho_i$. Then:

$$
\|W - C U^+ R\|_F \leq (1 + \epsilon) \|W - W_{r^*}\|_F
\tag{A.1}
$$

with probability $\geq 1 - \delta$ provided $r = \mathcal{O}(r^* / \epsilon^2 \log(1/\delta))$.

**Deterministic version.** If we select the top-$r$ rows and columns 
deterministically, the bound holds with high probability for uniform 
leverage distributions (Papailiopoulos et al., 2014).

## A.2 Leverage Score Computation

Given SVD $W = U \Sigma V^\top$:

$$
\pi_j = \sum_{k=1}^{\min(m, n)} V_{j, k}^2 = \|V_{j, :}\|_2^2
\tag{A.2}
$$

$$
\rho_i = \sum_{k=1}^{\min(m, n)} U_{i, k}^2 = \|U_{i, :}\|_2^2
\tag{A.3}
$$

**Normalization.** $\sum_j \pi_j = r^* = \sum_i \rho_i$.

## A.3 Angular LSH Collision Probability

**Theorem (Charikar, 2002).** For unit vectors $x, y \in \mathbb{R}^d$ with 
angle $\theta = \arccos(x^\top y)$:

$$
\Pr_{a \sim \mathcal{N}(0, I)}[\text{sign}(a^\top x) = \text{sign}(a^\top y)] = 1 - \frac{\theta}{\pi}
\tag{A.4}
$$

**Proof.** The distribution of $a^\top x$ is $\mathcal{N}(0, \|x\|^2)$ and 
similarly for $y$. The joint distribution is bivariate Gaussian with 
correlation $x^\top y = \cos \theta$. The probability of the same sign is 
the probability of both being on the same side of the origin:

$$
\Pr[\text{same sign}] = 1 - \frac{\theta}{\pi}
$$

$\square$

## A.4 Softpick Sparsity Analysis

For a score vector $x \in \mathbb{R}^L$ with maximum $m$, softpick gives:

$$
\text{softpick}_i = \text{ReLU}(e^{(x_i - m)/\tau} - e^{-m/\tau})
\tag{A.5}
$$

This is non-zero iff $x_i > m - \tau \log(2)$. The fraction of non-zero 
entries is:

$$
\text{sparsity} = \frac{1}{L} \#\{i : x_i > m - \tau \log 2\}
\tag{A.6}
$$

For peaked distributions (typical of trained attention), this is 5-15%.

## A.5 Softpick Fallback

When $\sum_j \text{softpick}_j < \epsilon$, we substitute softmax:

$$
p_i = \frac{e^{x_i / \tau}}{\sum_j e^{x_j / \tau}}
\tag{A.7}
$$

**Trigger condition.** This occurs iff $m < 0$ and the spread 
$m - x_\text{min}$ is small. In practice, this happens for rows where all 
scores are dominated by masking (e.g., padding).

## A.6 Multi-Round Averaging

For $R$ independent rounds with individual outputs $O^{(r)}$, the average 
is:

$$
O = \frac{1}{R} \sum_{r=1}^{R} O^{(r)}
\tag{A.8}
$$

**Variance.** If each $O^{(r)}$ has variance $\sigma^2$ and they are 
independent:

$$
\mathrm{Var}(O) = \frac{\sigma^2}{R}
\tag{A.9}
$$

So multi-round averaging reduces variance linearly with $R$.

---