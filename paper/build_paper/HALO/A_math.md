<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Prefix Sum Correctness

**Claim.** The cumulative sum formulation computes exact causal linear 
attention.

**Proof.** For position $i$, causal linear attention is:

$$
O_i = \frac{\sum_{j \le i} \phi(Q_i)^\top \phi(K_j) V_j}{\sum_{j \le i} \phi(Q_i)^\top \phi(K_j)}
\tag{A.1}
$$

Factoring out $\phi(Q_i)$:

$$
O_i = \frac{\phi(Q_i)^\top \sum_{j \le i} \phi(K_j) V_j^\top}{\phi(Q_i)^\top \sum_{j \le i} \phi(K_j)}
\tag{A.2}
$$

The cumulative sums $\text{cum}^{(i)} = \sum_{j \le i} \phi(K_j) V_j^\top$ 
and $\text{den}^{(i)} = \sum_{j \le i} \phi(K_j)$ are computed via `cumsum`. 
$\square$

## A.2 Non-Negativity of ELU+1

**Claim.** $\phi(x) = \mathrm{elu}(x) + 1 > 0$ for all $x \in \mathbb{R}$.

**Proof.** 

- For $x > 0$: $\mathrm{elu}(x) = x > 0$, so $\phi(x) = x + 1 > 1 > 0$.
- For $x \le 0$: $\mathrm{elu}(x) = e^x - 1 \in (-1, 0]$, so 
  $\phi(x) = e^x \in (0, 1]$.

In both cases, $\phi(x) > 0$. $\square$

## A.3 Gate Gradient

The gradient of the loss w.r.t. the gate logit $z = w^\top x + b$:

$$
\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial g} \cdot g(1 - g)
\tag{A.3}
$$

where:

$$
\frac{\partial \mathcal{L}}{\partial g} = \nabla_O \mathcal{L} \cdot (O^\text{dense} - O^\text{lin})
\tag{A.4}
$$

## A.4 Fusion Interpolation

**Claim.** The fusion $O = g O^\text{dense} + (1 - g) O^\text{lin}$ is a 
convex combination for $g \in [0, 1]$.

**Proof.** Trivial from $g + (1 - g) = 1$ and $0 \le g \le 1$. $\square$

## A.5 Cumsum Memory

For a batch of $B$ sequences of length $L$ with $H$ heads and head dim $d_h$:

$$
\text{Memory} = B H L d_h^2 \cdot \text{bytes}
\tag{A.5}
$$

For $B = 8, H = 12, L = 512, d_h = 64$: $8 \times 12 \times 512 \times 64^2 \times 2 = 4$ GB. 

**This is prohibitive.** We need chunked cumsum for practical long-context 
inference.

## A.6 Approximation Error

**Claim.** The fusion error is bounded by:

$$
\|O - O^\text{dense}\| \leq (1 - g) \cdot \|O^\text{lin} - O^\text{dense}\|
\tag{A.6}
$$

**Proof.** $O - O^\text{dense} = (1 - g)(O^\text{lin} - O^\text{dense})$. 
Taking norms gives (A.6). $\square$

---
