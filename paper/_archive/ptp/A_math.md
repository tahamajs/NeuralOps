<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 [CLS] Retention Proof

**Claim.** Under score clamping $s_0 = 10^4$, token 0 is always in 
$\mathcal{T}$.

**Proof.** All scores except $s_0$ are in $(-\infty, 10^4)$. When we 
compute top-$k$ with $k \geq 1$, position 0 has the strictly largest 
score (assuming ties are broken in favor of earlier indices). Therefore 
$0 \in \mathcal{T}$. $\square$

## A.2 Causality Preserved

**Claim.** For decoders, PTP attention satisfies causality.

**Proof.** For query $i$ and pruned key $j \in \mathcal{T}$:
- If original position $\mathcal{T}[b, j] > i$, then the causal mask is 
  $-\infty$, and $P_{ij} = 0$.
- Otherwise, $\mathcal{T}[b, j] \leq i$, and attention is allowed.

Therefore $O_i$ depends only on tokens with original position $\leq i$. 
$\square$

## A.3 Approximation Error

**Claim.** The difference between PTP and dense attention output is:

$$
\|O - O_\text{dense}\|_F \leq 2 \max_j \|V_j\|_F \cdot \sum_{j \notin \mathcal{T}} P_j
\tag{A.1}
$$

**Proof sketch.** Write $O = \sum_{j \in \mathcal{T}} P_j V_j$ and 
$O_\text{dense} = \sum_{j=1}^L P_j V_j$. The difference is 
$\sum_{j \notin \mathcal{T}} P_j V_j$. Taking norms:

$$
\|O - O_\text{dense}\|_F = \left\|\sum_{j \notin \mathcal{T}} P_j V_j\right\|_F \leq \sum_{j \notin \mathcal{T}} P_j \|V_j\|_F
$$

Using $\|V_j\|_F \leq \max_j \|V_j\|_F$:

$$
\|O - O_\text{dense}\|_F \leq \max_j \|V_j\|_F \cdot \sum_{j \notin \mathcal{T}} P_j \qquad \square
$$

## A.4 Complexity

**PTP attention:**

| Operation | Complexity |
|---|---|
| Scoring | $\mathcal{O}(B L d \cdot 64)$ |
| Top-$k$ | $\mathcal{O}(B L \log k)$ |
| Gather | $\mathcal{O}(B H k d_h)$ |
| Attention | $\mathcal{O}(B H L k d_h)$ |

**Reduction factor** from dense: $\rho = k / L$.

## A.5 Optimal Keep Ratio

Assume retained attention mass $m(\rho)$ satisfies $m(1) = 1$ and 
$m(\rho)$ is concave. Then task accuracy is approximately 
$\text{Acc}(\rho) \approx \text{Acc}(1) \cdot m(\rho)$.

We seek:

$$
\rho^* = \arg\max_\rho \left[\text{Acc}(1) m(\rho) - \lambda \rho\right]
\tag{A.2}
$$

Setting derivative to zero:

$$
\text{Acc}(1) m'(\rho) = \lambda
\tag{A.3}
$$

For $m(\rho) = \sqrt{\rho}$ (typical), $\rho^* = (\text{Acc}(1) / (2\lambda))^2$.

## A.6 Score Network Gradient

For loss $\mathcal{L}$ and score $s_i = \sigma(g_i)$ where 
$g_i = \text{MLP}(x_i)$:

$$
\frac{\partial \mathcal{L}}{\partial g_i} = \frac{\partial \mathcal{L}}{\partial s_i} \cdot \sigma(g_i)(1 - \sigma(g_i))
\tag{A.4}
$$

The gradient flows through the top-$k$ operation only via the selected 
tokens (a straight-through estimator is implicitly used for the 
non-selected ones).

## A.7 Comparison to Random Pruning

**Claim.** Random pruning of $k$ tokens has expected accuracy:

$$
\mathbb{E}[\text{Acc}_\text{random}] \approx \text{Acc}(1) \cdot \frac{k}{L}
\tag{A.5}
$$

**Observation.** PTP significantly exceeds this baseline.

## A.8 Numerical Stability

Score clamping bounds $s_i \in [-10^4, 10^4]$, avoiding overflow in 
sigmoid (which is bounded anyway) and preserving numerical stability 
in `topk`.

---